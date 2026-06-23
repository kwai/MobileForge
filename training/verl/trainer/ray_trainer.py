# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import _insert_hint_into_text, process_image
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_per_group_reward_metrics,
    compute_rollout_group_metrics,
    compute_rollout_group_metrics_from_scores,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        self.steps_per_epoch = len(train_dataloader)
        print(f"Steps per epoch: {self.steps_per_epoch}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        detected_tasks_path = os.path.join(folder_path, "detected_tasks.json")
        detected_tasks_info = {
            "detected_perfect_task_ids": list(getattr(self, "detected_perfect_task_ids", set())),
            "detected_wrong_task_ids": list(getattr(self, "detected_wrong_task_ids", set())),
        }
        with open(detected_tasks_path, "w") as f:
            json.dump(detected_tasks_info, f)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        # Safe default: always initialise to empty sets; checkpoint load will
        # overwrite them when applicable.
        self.detected_perfect_task_ids = set()
        self.detected_wrong_task_ids = set()

        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            # No checkpoint: initialise empty sets (they will be populated during training)
            self.detected_perfect_task_ids = set()
            self.detected_wrong_task_ids = set()
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        detected_tasks_path = os.path.join(load_checkpoint_path, "detected_tasks.json")
        if os.path.exists(detected_tasks_path):
            with open(detected_tasks_path) as f:
                detected_info = json.load(f)
            # Only restore task sets if the corresponding once-only mode is enabled.
            # If it was disabled when the checkpoint was saved, the set was empty.
            # If it was enabled but the user now changed to a different mode, stale data
            # shouldn't leak either — only restore when the flag is currently True.
            if self.config.data.exclude_zv_perfect_once:
                self.detected_perfect_task_ids = set(detected_info.get("detected_perfect_task_ids", []))
            if self.config.data.exclude_zv_wrong_once:
                self.detected_wrong_task_ids = set(detected_info.get("detected_wrong_task_ids", []))
            print(f"[Checkpoint] Restored detected task sets: "
                  f"perfect={len(self.detected_perfect_task_ids)}, "
                  f"wrong={len(self.detected_wrong_task_ids)}")
        else:
            # No detected_tasks.json — either old checkpoint or never saved.
            # Keep current (empty) sets; they will be rebuilt during training.
            pass

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples

        val_generations_to_log:
            - 0: 不记录
            - -1: 全量记录
            - >0: 记录指定数量的样本
        """
        if self.config.trainer.val_generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # -1 表示全量记录，否则截取指定数量
        if self.config.trainer.val_generations_to_log > 0:
            samples = samples[: self.config.trainer.val_generations_to_log]
        # val_generations_to_log == -1 时记录所有样本

        self.logger.log_generation(samples, self.global_step)

    def _maybe_log_train_generations(self, batch: DataProto) -> None:
        """Log training samples to a separate file

        train_generations_to_log:
            - 0: 不记录
            - -1: 全量记录
            - >0: 记录指定数量的样本
        """
        if self.config.trainer.train_generations_to_log == 0:
            return

        try:
            # Extract data from batch
            input_ids = batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            labels = batch.non_tensor_batch.get("ground_truth", [""] * len(input_texts))
            if hasattr(labels, "tolist"):
                labels = labels.tolist()

            # 获取原始 assistant response（用于对比）
            raw_responses = batch.non_tensor_batch.get("raw_response", [""] * len(input_texts))
            if hasattr(raw_responses, "tolist"):
                raw_responses = raw_responses.tolist()

            # Create samples list (包含 raw_response)
            samples = list(zip(input_texts, output_texts, labels, scores, raw_responses))

            # Apply limit if not -1
            if self.config.trainer.train_generations_to_log > 0:
                samples = samples[: self.config.trainer.train_generations_to_log]

            # Write to train_generations.log file
            train_log_path = os.path.join(self.config.trainer.save_checkpoint_path, "train_generations.log")
            with open(train_log_path, "a") as f:
                f.write(f"\n{'=' * 60}\nStep {self.global_step}\n{'=' * 60}\n")
                for inp, out, lab, score, raw_resp in samples:
                    f.write(f"[prompt] {inp}\n")
                    f.write(f"[output] {out}\n")
                    f.write(f"[raw_response] {raw_resp}\n")
                    f.write(f"[ground_truth] {lab}\n")
                    f.write(f"[score] {score}\n\n")
        except Exception as e:
            print(f"Warning: Failed to log train generations: {e}")

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        # Collect group keys for per-app/per-action metrics
        val_app_names = []
        val_action_types = []
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            # Collect group keys for per-group metrics
            if "app_name" in test_batch.non_tensor_batch:
                val_app_names.extend(test_batch.non_tensor_batch["app_name"].tolist())
            if "action_type" in test_batch.non_tensor_batch:
                val_action_types.extend(test_batch.non_tensor_batch["action_type"].tolist())

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}

        # compute validation rollout group metrics (all-fail, all-success, avg success rate)
        val_group_metrics = {}
        if repeat_times > 1 and len(sample_scores) > 0:
            scores_array = np.array(sample_scores)
            _atw = self.config.worker.reward.reward_function_kwargs.get("action_type_weight", 0.5)
            val_group_metrics = compute_rollout_group_metrics_from_scores(
                scores=scores_array,
                n=repeat_times,
                success_threshold=0.9,
                prefix="val",
                action_type_weight=_atw,
            )

        # compute per-app and per-action-type validation reward metrics
        val_per_group_metrics = {}
        if val_app_names or val_action_types:
            group_keys = {}
            if val_app_names:
                group_keys["app_name"] = np.array(val_app_names, dtype=object)
            if val_action_types:
                group_keys["action_type"] = np.array(val_action_types, dtype=object)
            val_per_group_metrics = compute_per_group_reward_metrics(
                reward_metrics=dict(reward_metrics_lst),
                group_keys=group_keys,
                prefix="val",
            )

        print("Finish validation.")
        return {
            "val/reward_score": self.val_reward_score,
            **val_reward_metrics,
            **val_length_metrics,
            **val_group_metrics,
            **val_per_group_metrics,
        }

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _rebuild_with_hint(self, mini_batch: DataProto, hint_text: str, images) -> DataProto:
        """
        Re-tokenize prompts by injecting EVALUATION HINTS into the raw user text.

        This is used for adaptive hint retry: when a group is zero-variance without hints,
        we re-tokenize the same prompts but with hints injected to see if hints provide
        enough gradient signal for training.

        Args:
            mini_batch: DataProto with the original (no-hint) batch, already has
                ground_truth, uid, app_name, action_type, etc.
            hint_text: The EVALUATION HINTS content extracted from the raw sample.
            images: List of PIL.Image objects (same for all samples in batch).

        Returns:
            A new DataProto with re-tokenized prompts (with hints injected).
        """
        cfg = self.config.data
        batch_size = len(mini_batch.batch)

        # Reconstruct messages from conversations stored in mini_batch.non_tensor_batch
        # The conversations are in the original sample data; we need to re-extract
        # and re-tokenize with hints.
        # We re-use the stored sample data by going through the dataset's tokenization.
        # Since we already consumed the batch_dict, we need to re-tokenize from scratch.
        # Strategy: decode the existing raw_prompt_ids back to text, inject hint, re-tokenize.

        # Decode the raw prompt (without hint) back to text
        raw_prompt_ids = mini_batch.non_tensor_batch.get("raw_prompt_ids")
        if raw_prompt_ids is None:
            raise ValueError("_rebuild_with_hint requires raw_prompt_ids in non_tensor_batch")

        # Get tokenizer/processor from self
        tokenizer = self.tokenizer
        processor = self.processor

        # Decode to get the prompt text
        if isinstance(raw_prompt_ids, np.ndarray):
            raw_prompt_ids = raw_prompt_ids.tolist()
        prompt_text = tokenizer.decode(raw_prompt_ids, skip_special_tokens=True)

        # Inject hint into prompt text
        text_with_hint = _insert_hint_into_text(prompt_text, hint_text)

        # Re-tokenize with hint
        if images:
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": text_with_hint}],
                add_generation_prompt=True,
                tokenize=False,
            )
            processed_images = [process_image(img, cfg.min_pixels, cfg.max_pixels) for img in images]
            model_inputs = processor(
                processed_images, [prompt], add_special_tokens=False, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        else:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text_with_hint}],
                add_generation_prompt=True,
                tokenize=False,
            )
            model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # Handle position_ids
        if processor is not None and hasattr(processor, "image_processor"):
            proc_name = processor.image_processor.__class__.__name__
            if "Qwen2VLImageProcessor" in proc_name or "Qwen3VLImageProcessor" in proc_name:
                if "Qwen3VLProcessor" in processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )
                text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        # Pad/truncate to max_prompt_length
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=cfg.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )

        # Build raw_prompt_ids for vLLM
        new_raw_prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False, truncation=True, max_length=cfg.max_prompt_length
        )

        # Broadcast to all samples in batch (they all share the same prompt)
        input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Update mini_batch with new tokenized data
        mini_batch.batch["input_ids"] = input_ids
        mini_batch.batch["attention_mask"] = attention_mask
        mini_batch.batch["position_ids"] = position_ids
        mini_batch.non_tensor_batch["raw_prompt_ids"] = np.array([new_raw_prompt_ids] * batch_size, dtype=object)

        if images:
            mini_batch.non_tensor_batch["multi_modal_data"] = np.array(
                [{"images": images}] * batch_size, dtype=object
            )
        else:
            mini_batch.non_tensor_batch.pop("multi_modal_data", None)

        return mini_batch

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        cfg = self.config.data
        use_zv_filter = cfg.exclude_zv_perfect or cfg.exclude_zv_wrong

        # ── retry configuration ─────────────────────────────────────────────────
        use_retry_wrong = cfg.exclude_zv_wrong and cfg.retry_zv_wrong
        max_retry_wrong = cfg.retry_zv_wrong_max_attempts
        use_retry_perfect = cfg.exclude_zv_perfect and cfg.retry_zv_perfect
        max_retry_perfect = cfg.retry_zv_perfect_max_attempts

        # ── adaptive hint configuration ──────────────────────────────────────────
        # adaptive_hint: first rollout without hint; if zv, retry with hint
        adaptive_hint_enabled = cfg.adaptive_hint
        adaptive_hint_only_zv = cfg.adaptive_hint_only_zv
        adaptive_hint_retry_wrong = cfg.adaptive_hint_retry_wrong
        adaptive_hint_retry_perfect = cfg.adaptive_hint_retry_perfect
        adaptive_hint_max_attempts = cfg.adaptive_hint_max_attempts

        # pending queue: each entry is (batch_dict, remaining_attempts, retry_cat, hint_text, images)
        # hint_text: the EVALUATION HINTS content extracted from the raw sample
        # images: list of PIL.Image objects for multimodal input
        pending_retry: deque[tuple[dict, int, str, str, Any]] = deque()
        kept_samples: list[DataProto] = []

        # ── raw pre-filter stats (before any zv filtering) ────────────────────────
        raw_stats: dict[str, int | float] = {
            "total_prompts": 0,
            "total_rollouts": 0,
            "non_zv": 0, "perfect": 0, "wrong": 0, "type_only": 0, "partial": 0,
        }
        # ── post-filter stats (after zv filtering, before retry re-add) ─────────
        post_filter_stats: dict[str, int | float] = {
            "total_prompts": 0, "total_rollouts": 0,
            "non_zv": 0, "perfect": 0, "wrong": 0, "type_only": 0, "partial": 0,
            "dropped_perfect": 0, "dropped_wrong": 0,
            "skipped_once": 0,
            "retry_wrong_rescued": 0, "retry_wrong_exhausted": 0,
            "retry_perfect_rescued": 0, "retry_perfect_exhausted": 0,
            # adaptive hint stats
            "adaptive_hint_wrong_rescued": 0,
            "adaptive_hint_wrong_exhausted": 0,
            "adaptive_hint_perfect_rescued": 0,
            "adaptive_hint_perfect_exhausted": 0,
        }
        retry_wrong_history: list[int] = []
        retry_perfect_history: list[int] = []
        adaptive_hint_history: list[int] = []  # attempts used per adaptive hint retry

        # ── Safety guard: prevent infinite loop when all prompts are zv ─────────────
        # total_prompts_processed tracks how many prompts (including retries) we have
        # scanned.  If it grows far beyond rollout_batch_size without keeping any samples,
        # the dataset likely consists entirely of zv_wrong / zv_perfect prompts whose
        # retry budgets have been exhausted.
        total_prompts_processed = 0

        # ── once-only mode: set of task_ids already detected as zv ─────────────
        # These are instance variables persisted across fit() calls and checkpoints.
        detected_perfect = getattr(self, "detected_perfect_task_ids", set())
        detected_wrong = getattr(self, "detected_wrong_task_ids", set())
        once_perfect_enabled = cfg.exclude_zv_perfect_once and cfg.exclude_zv_perfect
        once_wrong_enabled = cfg.exclude_zv_wrong_once and cfg.exclude_zv_wrong

        print("Start generating batch...")
        n = self.config.worker.rollout.n
        rollout_batch_size = cfg.rollout_batch_size
        while True:
            # ── Safety guard: abort if dataset is too sparse ─────────────────────────
            # When kept_samples is still empty after scanning many prompts (including
            # retries), the dataset probably contains only zv_wrong / zv_perfect
            # prompts whose retry budgets have been exhausted.  Without this guard the
            # loop would spin forever.
            if kept_samples:
                # Normal case: we have at least some kept samples → use them.
                total_rollouts = sum(len(s) for s in kept_samples)
                current_prompts = total_rollouts // n
                if current_prompts >= rollout_batch_size:
                    break
            else:
                # kept_samples is empty: check how many prompts we've scanned so far.
                # Allow 10x the requested size before giving up, then raise so the
                # user can fix the data / retry configuration.
                if total_prompts_processed >= rollout_batch_size * 10:
                    raise RuntimeError(
                        f"[_make_batch_data] Safety exit: scanned {total_prompts_processed} "
                        f"prompts but kept_samples is still empty. The dataset may contain "
                        f"only zv_wrong / zv_perfect prompts with retry budgets "
                        f"exhausted. Hints: (1) increase --retry_zv_wrong_max_attempts, "
                        f"(2) temporarily disable --exclude_zv_wrong, "
                        f"(3) relax --filter_sr_min / --filter_sr_max."
                    )
            # ── 1. Source selection ──────────────────────────────────────────────
            remaining = 0
            retry_cat = ""  # "" | "wrong" | "perfect" | "adaptive_hint"
            hint_text = ""  # extracted EVALUATION HINTS text
            images = None    # list of PIL.Image for multimodal
            if pending_retry:
                batch_dict, remaining, retry_cat, hint_text, images = pending_retry.popleft()
            else:
                # Pull fresh samples, skipping detected once-only tasks.
                while True:
                    try:
                        batch_dict = next(self.data_iterator)
                    except StopIteration:
                        self.data_iterator = iter(self.train_dataloader)
                        batch_dict = next(self.data_iterator)

                    # collate_fn converts task_id to np.array(object) of shape (batch_size,)
                    task_ids_raw = batch_dict.get("task_id", np.array([], dtype=object))
                    # Each element is a Python str (collate_fn stores raw objects);
                    # .item() only applies to 0-d numpy scalars, not to strings.
                    task_ids = (
                        [str(t.item()) if hasattr(t, "item") else str(t) for t in task_ids_raw]
                        if task_ids_raw.ndim == 1 and task_ids_raw.dtype == object
                        else []
                    )

                    # Skip entire batch only when ALL task_ids are once-only detected.
                    # If some are detected and some aren't, the batch proceeds (detected ones
                    # will be dropped at zv-classification stage; newly detected ones are
                    # added to the set for future epochs).
                    all_wrong_skipped = (
                        once_wrong_enabled
                        and all(tid in detected_wrong for tid in task_ids)
                    )
                    all_perfect_skipped = (
                        once_perfect_enabled
                        and all(tid in detected_perfect for tid in task_ids)
                    )
                    if not all_wrong_skipped and not all_perfect_skipped:
                        break

                # fresh sample: set retry budgets based on which excludes are active
                remaining = max_retry_wrong if cfg.exclude_zv_wrong else 0
                if cfg.exclude_zv_perfect:
                    remaining = max(remaining, max_retry_perfect)
                retry_cat = ""
                # Extract hint_text and images from fresh batch for adaptive hint retry
                if adaptive_hint_enabled:
                    hint_arr = batch_dict.get("hint_text", np.array([], dtype=object))
                    images_arr = batch_dict.get("_images_for_hint", np.array([], dtype=object))
                    if hint_arr.ndim == 1 and hint_arr.dtype == object and len(hint_arr) > 0:
                        # All samples in batch share the same hint; take the first non-empty
                        hint_text = next((str(h) for h in hint_arr if h), "")
                    else:
                        hint_text = ""
                    if images_arr.ndim == 1 and images_arr.dtype == object and len(images_arr) > 0:
                        images = images_arr[0]  # shared images across batch
                    else:
                        images = None
                else:
                    hint_text = ""
                    images = None

            # ── 2. Build DataProto and generate rollout ──────────────────────────
            meta_info = {
                "min_pixels": cfg.min_pixels,
                "max_pixels": cfg.max_pixels,
                "video_fps": cfg.video_fps,
            }
            mini_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            mini_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(mini_batch.batch))], dtype=object
            )

            # ── 2a. Adaptive hint: re-tokenize with hints injected ────────────────
            # If retrying with hint (hint_text is non-empty), we need to re-tokenize
            # the prompts by injecting the hint into the raw text.
            if hint_text and images is not None:
                mini_batch = self._rebuild_with_hint(mini_batch, hint_text, images)
                # After rebuild, the batch dict is consumed; we use mini_batch directly.
                gen_batch = mini_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                )
            else:
                gen_batch = mini_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
                )

            gen_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                baseline_batch = deepcopy(gen_batch)
                baseline_batch.meta_info["temperature"] = 0
                baseline_batch.meta_info["n"] = 1
                baseline_output = self.actor_rollout_ref_wg.generate_sequences(baseline_batch)
                mini_batch = mini_batch.union(baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(mini_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                mini_batch.pop(batch_keys=list(baseline_output.batch.keys()))
                mini_batch.batch["reward_baselines"] = reward_baseline_tensor
                del baseline_batch, baseline_output

            mini_batch = mini_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            mini_batch = mini_batch.union(gen_output)

            # ── 3. online_filtering (unchanged) ─────────────────────────────────
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(mini_batch))
                mini_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    metrics.setdefault(k, []).extend(v)
                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = mini_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)
                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low
                    and avg_score < self.config.algorithm.filter_high
                ]
                kept_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if not kept_idxs:
                    raise RuntimeError("No sample kept after online filtering.")
                mini_batch = mini_batch[kept_idxs]

            # ── 4. zv classification + filter ──────────────────────────────────
            n = self.config.worker.rollout.n
            num_prompts = len(mini_batch) // n
            raw_stats["total_prompts"] += num_prompts
            raw_stats["total_rollouts"] += num_prompts * n
            total_prompts_processed += num_prompts

            action_type_weight = self.config.worker.reward.reward_function_kwargs.get("action_type_weight", 0.5)
            dist_lo = action_type_weight * 0.5
            dist_hi = action_type_weight * 1.1

            if use_zv_filter:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(mini_batch))
                mini_batch.batch["token_level_scores"] = reward_tensor

                overall_list = reward_metrics.get("overall", reward_tensor.sum(dim=-1).cpu().tolist())
                uids = mini_batch.non_tensor_batch["uid"]
                uid2scores_map: dict[str, list[float]] = defaultdict(list)
                for uid, score in zip(uids, overall_list):
                    uid2scores_map[uid].append(score)

                local_kept_idxs: list[int] = []
                attempts_for_wrong: dict[int, int] = {}
                attempts_for_perfect: dict[int, int] = {}

                for p_idx in range(num_prompts):
                    uid = uids[p_idx * n]
                    scores = uid2scores_map.get(uid, [])
                    if len(scores) != n:
                        continue
                    std_s = float(np.std(scores, ddof=0))
                    mean_s = float(np.mean(scores))

                    # ── classify ──────────────────────────────────────────────
                    if std_s >= 1e-4:
                        cat = "non_zv"
                    elif mean_s >= 0.99:
                        cat = "perfect"
                    elif mean_s < dist_lo:
                        cat = "wrong"
                    elif mean_s <= dist_hi:
                        cat = "type_only"
                    else:
                        cat = "partial"

                    raw_stats[cat] += 1

                    # ── decide drop / retry ───────────────────────────────────────
                    drop_perfect = cfg.exclude_zv_perfect and cat == "perfect"
                    drop_wrong = cfg.exclude_zv_wrong and cat == "wrong"
                    drop = drop_perfect or drop_wrong

                    if drop:
                        # Determine which retry budget applies and whether this is
                        # a fresh or already-retried sample.
                        is_wrong = cat == "wrong"
                        is_retrying = (retry_cat == "wrong" and is_wrong) or (retry_cat == "perfect" and not is_wrong)
                        budget = remaining if is_retrying else (max_retry_wrong if is_wrong else max_retry_perfect)

                        if is_wrong:
                            can_retry = use_retry_wrong and budget > 0 and not is_retrying
                        else:
                            can_retry = use_retry_perfect and budget > 0 and not is_retrying

                        if can_retry:
                            # Fresh sample → retry. Already-retried → drop (prevents loops).
                            attempts_used = budget
                            if is_wrong:
                                attempts_for_wrong[p_idx] = attempts_used
                            else:
                                attempts_for_perfect[p_idx] = attempts_used
                            pending_retry.append((batch_dict, budget - 1, cat, hint_text, images))
                            if is_wrong:
                                post_filter_stats["retry_wrong_rescued"] += 1
                            else:
                                post_filter_stats["retry_perfect_rescued"] += 1
                        else:
                            # Drop: record task_id for once-only mode + stats.
                            # batch_dict["task_id"] is np.array(object) of shape (batch_size,);
                            # p_idx indexes which sample this group corresponds to.
                            tid_arr = batch_dict.get("task_id", np.array([], dtype=object))
                            if tid_arr.size == 1:
                                task_id = str(tid_arr.item()) if hasattr(tid_arr, "item") else str(tid_arr)
                            elif tid_arr.ndim == 1 and tid_arr.size > p_idx:
                                task_id = str(tid_arr[p_idx].item()) if hasattr(tid_arr[p_idx], "item") else str(tid_arr[p_idx])
                            else:
                                task_id = ""
                            if drop_wrong:
                                if once_wrong_enabled:
                                    self.detected_wrong_task_ids.add(task_id)
                                post_filter_stats["dropped_wrong"] += 1
                                if is_retrying and retry_cat == "wrong":
                                    post_filter_stats["retry_wrong_exhausted"] += 1
                            else:
                                if once_perfect_enabled:
                                    self.detected_perfect_task_ids.add(task_id)
                                post_filter_stats["dropped_perfect"] += 1
                                if is_retrying and retry_cat == "perfect":
                                    post_filter_stats["retry_perfect_exhausted"] += 1
                            post_filter_stats["total_prompts"] += 1
                            post_filter_stats["total_rollouts"] += n
                    else:
                        local_kept_idxs.extend(range(p_idx * n, (p_idx + 1) * n))
                        post_filter_stats[cat] += 1
                        post_filter_stats["total_prompts"] += 1
                        post_filter_stats["total_rollouts"] += n

                retry_wrong_history.extend(attempts_for_wrong.values())
                retry_perfect_history.extend(attempts_for_perfect.values())

                if local_kept_idxs:
                    kept_samples.append(mini_batch[local_kept_idxs])

                # ── 4b. Adaptive hint: retry wrong groups with hints ─────────────────
                # Independent of use_zv_filter: always run when adaptive_hint is enabled.
                # reuse reward from §4 if available; otherwise compute it here.
                reward_for_adaptive: Optional[Any] = reward_tensor if use_zv_filter else None
                reward_metrics_for_adaptive: Optional[dict] = reward_metrics if use_zv_filter else None
                if adaptive_hint_enabled and reward_for_adaptive is None:
                    reward_for_adaptive, reward_metrics_for_adaptive = ray.get(
                        self.reward_fn.compute_reward.remote(mini_batch)
                    )

                if adaptive_hint_enabled and reward_for_adaptive is not None:
                    overall_list2 = reward_metrics_for_adaptive.get(
                        "overall", reward_for_adaptive.sum(dim=-1).cpu().tolist()
                    )
                    uid2scores2: dict[str, list[float]] = defaultdict(list)
                    for uid, score in zip(uids, overall_list2):
                        uid2scores2[uid].append(score)

                    adaptive_kept_idxs: list[int] = []
                    adaptive_retry_attempts: dict[int, int] = {}
                    is_hint_retrying = retry_cat == "adaptive_hint"

                    for p_idx in range(num_prompts):
                        uid = uids[p_idx * n]
                        scores2 = uid2scores2.get(uid, [])
                        if len(scores2) != n:
                            continue
                        std_s2 = float(np.std(scores2, ddof=0))
                        mean_s2 = float(np.mean(scores2))

                        # Classify: only "wrong" triggers hint retry (already-correct = no need)
                        if std_s2 >= 1e-4:
                            cat2 = "non_zv"
                        elif mean_s2 >= 0.99:
                            cat2 = "perfect"
                        elif mean_s2 < dist_lo:
                            cat2 = "wrong"
                        elif mean_s2 <= dist_hi:
                            cat2 = "type_only"
                        else:
                            cat2 = "partial"

                        # Skip if already kept by zv filter
                        if p_idx * n in local_kept_idxs:
                            continue

                        # Hint retry only for "wrong" groups (or type_only/partial if enabled)
                        is_wrong_cat = cat2 == "wrong"
                        is_partial_cat = cat2 in ("type_only", "partial")
                        should_hint_retry = is_wrong_cat or is_partial_cat
                        if not should_hint_retry:
                            # non_zv → keep; perfect → skip (no need to retry)
                            if cat2 == "non_zv":
                                adaptive_kept_idxs.extend(range(p_idx * n, (p_idx + 1) * n))
                            continue

                        has_hint = bool(hint_text)
                        if not has_hint:
                            # No hint available: keep if non_zv (already handled above)
                            continue

                        can_retry = not is_hint_retrying  # no loops
                        budget = (remaining if is_hint_retrying else adaptive_hint_max_attempts)
                        can_retry = can_retry and (budget > 0)
                        if is_wrong_cat:
                            can_retry = can_retry and adaptive_hint_retry_wrong
                        elif is_partial_cat:
                            can_retry = can_retry and (
                                adaptive_hint_retry_wrong or adaptive_hint_retry_perfect
                            )

                        if can_retry:
                            adaptive_retry_attempts[p_idx] = budget
                            pending_retry.append((
                                batch_dict,
                                budget - 1,
                                "adaptive_hint",
                                hint_text,
                                images,
                            ))
                            if is_hint_retrying:
                                if is_wrong_cat:
                                    post_filter_stats["adaptive_hint_wrong_exhausted"] += 1
                                else:
                                    post_filter_stats["adaptive_hint_perfect_exhausted"] += 1
                            else:
                                if is_wrong_cat:
                                    post_filter_stats["adaptive_hint_wrong_rescued"] += 1
                                else:
                                    post_filter_stats["adaptive_hint_perfect_rescued"] += 1
                        else:
                            # Exhausted or no hint: already counted (no double-count)
                            pass

                    adaptive_hint_history.extend(adaptive_retry_attempts.values())
                    if adaptive_kept_idxs:
                        kept_samples.append(mini_batch[adaptive_kept_idxs])

            else:
                # No zv filter: all rollouts kept (non-wrong/non-partial groups)
                # Wrong/partial groups are queued for hint retry, not added to kept_samples yet
                post_filter_stats["total_prompts"] += num_prompts
                post_filter_stats["total_rollouts"] += num_prompts * n
                post_filter_stats["non_zv"] += num_prompts

                # ── 4b (no zv filter). Adaptive hint still applies ───────────────
                if adaptive_hint_enabled:
                    reward_for_adaptive, reward_metrics_for_adaptive = ray.get(
                        self.reward_fn.compute_reward.remote(mini_batch)
                    )
                    # Get uids from mini_batch (not available from §4 since zv filter is off)
                    uids_no_zv = mini_batch.non_tensor_batch["uid"]
                    overall_list2 = reward_metrics_for_adaptive.get(
                        "overall", reward_for_adaptive.sum(dim=-1).cpu().tolist()
                    )
                    uid2scores2: dict[str, list[float]] = defaultdict(list)
                    for uid, score in zip(uids_no_zv, overall_list2):
                        uid2scores2[uid].append(score)

                    adaptive_kept_idxs: list[int] = []
                    adaptive_retry_attempts: dict[int, int] = {}
                    is_hint_retrying = retry_cat == "adaptive_hint"

                    for p_idx in range(num_prompts):
                        uid = uids_no_zv[p_idx * n]
                        scores2 = uid2scores2.get(uid, [])
                        if len(scores2) != n:
                            continue
                        std_s2 = float(np.std(scores2, ddof=0))
                        mean_s2 = float(np.mean(scores2))

                        if std_s2 >= 1e-4:
                            cat2 = "non_zv"
                        elif mean_s2 >= 0.99:
                            cat2 = "perfect"
                        elif mean_s2 < dist_lo:
                            cat2 = "wrong"
                        elif mean_s2 <= dist_hi:
                            cat2 = "type_only"
                        else:
                            cat2 = "partial"

                        is_wrong_cat = cat2 == "wrong"
                        is_partial_cat = cat2 in ("type_only", "partial")
                        should_hint_retry = is_wrong_cat or is_partial_cat
                        if not should_hint_retry:
                            if cat2 == "non_zv":
                                adaptive_kept_idxs.extend(range(p_idx * n, (p_idx + 1) * n))
                            continue

                        has_hint = bool(hint_text)
                        if not has_hint:
                            # No hint: keep this group (no point retrying without hint)
                            adaptive_kept_idxs.extend(range(p_idx * n, (p_idx + 1) * n))
                            continue

                        can_retry = not is_hint_retrying
                        budget = (remaining if is_hint_retrying else adaptive_hint_max_attempts)
                        can_retry = can_retry and (budget > 0)
                        if is_wrong_cat:
                            can_retry = can_retry and adaptive_hint_retry_wrong
                        elif is_partial_cat:
                            can_retry = can_retry and (
                                adaptive_hint_retry_wrong or adaptive_hint_retry_perfect
                            )

                        if can_retry:
                            adaptive_retry_attempts[p_idx] = budget
                            pending_retry.append((
                                batch_dict,
                                budget - 1,
                                "adaptive_hint",
                                hint_text,
                                images,
                            ))
                            if is_hint_retrying:
                                if is_wrong_cat:
                                    post_filter_stats["adaptive_hint_wrong_exhausted"] += 1
                                else:
                                    post_filter_stats["adaptive_hint_perfect_exhausted"] += 1
                            else:
                                if is_wrong_cat:
                                    post_filter_stats["adaptive_hint_wrong_rescued"] += 1
                                else:
                                    post_filter_stats["adaptive_hint_perfect_rescued"] += 1
                        else:
                            # Exhausted: keep without hint
                            adaptive_kept_idxs.extend(range(p_idx * n, (p_idx + 1) * n))

                    adaptive_hint_history.extend(adaptive_retry_attempts.values())
                    if adaptive_kept_idxs:
                        kept_samples.append(mini_batch[adaptive_kept_idxs])
                else:
                    # No adaptive hint: keep everything
                    kept_samples.append(mini_batch)

            # ── 5. Stopping condition ───────────────────────────────────────────
            total_rollouts = sum(len(s) for s in kept_samples)
            current_prompts = total_rollouts // n
            rollout_batch_size = cfg.rollout_batch_size

            if current_prompts >= rollout_batch_size:
                final_batch = DataProto.concat(kept_samples)
                final_batch = final_batch[: rollout_batch_size * n]

                # ── 6. Log comprehensive zv metrics to wandb ─────────────────
                _m = metrics  # shorthand

                total_raw_p = max(raw_stats["total_prompts"], 1)
                total_raw_r = max(raw_stats["total_rollouts"], 1)
                total_pf_p = max(post_filter_stats["total_prompts"], 1)
                total_pf_r = max(post_filter_stats["total_rollouts"], 1)

                # Raw pre-filter group composition
                _m["batch/zv_raw/group_count/total"]           = raw_stats["total_prompts"]
                _m["batch/zv_raw/group_count/non_zv"]          = raw_stats["non_zv"]
                _m["batch/zv_raw/group_count/perfect"]          = raw_stats["perfect"]
                _m["batch/zv_raw/group_count/wrong"]           = raw_stats["wrong"]
                _m["batch/zv_raw/group_count/type_only"]       = raw_stats["type_only"]
                _m["batch/zv_raw/group_count/partial"]        = raw_stats["partial"]
                _m["batch/zv_raw/group_ratio/non_zv"]          = raw_stats["non_zv"]         / total_raw_p
                _m["batch/zv_raw/group_ratio/perfect"]         = raw_stats["perfect"]        / total_raw_p
                _m["batch/zv_raw/group_ratio/wrong"]           = raw_stats["wrong"]          / total_raw_p
                _m["batch/zv_raw/group_ratio/type_only"]       = raw_stats["type_only"]      / total_raw_p
                _m["batch/zv_raw/group_ratio/partial"]         = raw_stats["partial"]        / total_raw_p
                _m["batch/zv_raw/zero_var_ratio"]              = (
                    raw_stats["perfect"] + raw_stats["wrong"] + raw_stats["type_only"] + raw_stats["partial"]
                ) / total_raw_p
                _m["batch/zv_raw/has_gradient_ratio"]          = raw_stats["non_zv"] / total_raw_p

                # Post-filter group composition
                _m["batch/zv_filter/group_count/total"]        = post_filter_stats["total_prompts"]
                _m["batch/zv_filter/group_count/non_zv"]       = post_filter_stats["non_zv"]
                _m["batch/zv_filter/group_count/perfect"]      = post_filter_stats["perfect"]
                _m["batch/zv_filter/group_count/wrong"]        = post_filter_stats["wrong"]
                _m["batch/zv_filter/group_count/type_only"]    = post_filter_stats["type_only"]
                _m["batch/zv_filter/group_count/partial"]     = post_filter_stats["partial"]
                _m["batch/zv_filter/group_ratio/non_zv"]      = post_filter_stats["non_zv"]   / total_pf_p
                _m["batch/zv_filter/group_ratio/perfect"]      = post_filter_stats["perfect"]  / total_pf_p
                _m["batch/zv_filter/group_ratio/wrong"]       = post_filter_stats["wrong"]    / total_pf_p
                _m["batch/zv_filter/group_ratio/type_only"]    = post_filter_stats["type_only"]/ total_pf_p
                _m["batch/zv_filter/group_ratio/partial"]     = post_filter_stats["partial"]  / total_pf_p
                _m["batch/zv_filter/zero_var_ratio"]          = (
                    post_filter_stats["perfect"] + post_filter_stats["wrong"]
                    + post_filter_stats["type_only"] + post_filter_stats["partial"]
                ) / total_pf_p
                _m["batch/zv_filter/has_gradient_ratio"]       = post_filter_stats["non_zv"] / total_pf_p

                # Effective rollout ratios (rollouts in non-zero-variance groups / total rollouts)
                _m["batch/zv_filter/effective_rollout_ratio"]  = (
                    post_filter_stats["non_zv"] * n
                ) / total_pf_r

                # Dropped by category
                _m["batch/zv_filter/dropped/perfect"]         = post_filter_stats["dropped_perfect"]
                _m["batch/zv_filter/dropped/wrong"]           = post_filter_stats["dropped_wrong"]
                _m["batch/zv_filter/dropped/total"]           = (
                    post_filter_stats["dropped_perfect"] + post_filter_stats["dropped_wrong"]
                )
                _m["batch/zv_filter/dropped/perfect_ratio"]  = (
                    post_filter_stats["dropped_perfect"] / total_raw_p
                )
                _m["batch/zv_filter/dropped/wrong_ratio"]     = (
                    post_filter_stats["dropped_wrong"] / total_raw_p
                )
                _m["batch/zv_filter/dropped/total_ratio"]     = (
                    (post_filter_stats["dropped_perfect"] + post_filter_stats["dropped_wrong"]) / total_raw_p
                )

                # Retry stats — wrong
                _m["batch/zv_retry/wrong/requested"]          = post_filter_stats["retry_wrong_rescued"]
                _m["batch/zv_retry/wrong/exhausted"]          = post_filter_stats["retry_wrong_exhausted"]
                if retry_wrong_history:
                    _m["batch/zv_retry/wrong/attempts_used/mean"] = float(np.mean(retry_wrong_history))
                    _m["batch/zv_retry/wrong/attempts_used/max"]  = float(np.max(retry_wrong_history))
                else:
                    _m["batch/zv_retry/wrong/attempts_used/mean"] = 0.0
                    _m["batch/zv_retry/wrong/attempts_used/max"]  = 0.0

                # Retry stats — perfect
                _m["batch/zv_retry/perfect/requested"]        = post_filter_stats["retry_perfect_rescued"]
                _m["batch/zv_retry/perfect/exhausted"]        = post_filter_stats["retry_perfect_exhausted"]
                if retry_perfect_history:
                    _m["batch/zv_retry/perfect/attempts_used/mean"] = float(np.mean(retry_perfect_history))
                    _m["batch/zv_retry/perfect/attempts_used/max"]  = float(np.max(retry_perfect_history))
                else:
                    _m["batch/zv_retry/perfect/attempts_used/mean"] = 0.0
                    _m["batch/zv_retry/perfect/attempts_used/max"]  = 0.0

                # Adaptive hint stats
                _m["batch/adaptive_hint/wrong_rescued"]      = post_filter_stats["adaptive_hint_wrong_rescued"]
                _m["batch/adaptive_hint/wrong_exhausted"]     = post_filter_stats["adaptive_hint_wrong_exhausted"]
                _m["batch/adaptive_hint/perfect_rescued"]     = post_filter_stats["adaptive_hint_perfect_rescued"]
                _m["batch/adaptive_hint/perfect_exhausted"]   = post_filter_stats["adaptive_hint_perfect_exhausted"]
                total_adaptive = (
                    post_filter_stats["adaptive_hint_wrong_rescued"]
                    + post_filter_stats["adaptive_hint_wrong_exhausted"]
                    + post_filter_stats["adaptive_hint_perfect_rescued"]
                    + post_filter_stats["adaptive_hint_perfect_exhausted"]
                )
                _m["batch/adaptive_hint/total"]              = total_adaptive
                if adaptive_hint_history:
                    _m["batch/adaptive_hint/attempts_used/mean"] = float(np.mean(adaptive_hint_history))
                    _m["batch/adaptive_hint/attempts_used/max"]  = float(np.max(adaptive_hint_history))
                else:
                    _m["batch/adaptive_hint/attempts_used/mean"] = 0.0
                    _m["batch/adaptive_hint/attempts_used/max"]  = 0.0

                _m["batch/zv_retry/queue_size_at_end"]        = len(pending_retry)

                # Once-only skipped tasks
                _m["batch/zv_filter/skipped_once"]            = post_filter_stats["skipped_once"]

                # Prompt count comparison
                _m["batch/zv_filter/prompts_raw"]             = raw_stats["total_prompts"]
                _m["batch/zv_filter/prompts_after_filter"]    = post_filter_stats["total_prompts"]
                _m["batch/zv_filter/rollouts_raw"]            = raw_stats["total_rollouts"]
                _m["batch/zv_filter/rollouts_after_filter"]  = post_filter_stats["total_rollouts"]
                _m["batch/zv_filter/prompt_retention_ratio"]  = total_pf_p / total_raw_p

                print(
                    f"[zv-metrics] raw: {dict(raw_stats)}  |  "
                    f"post_filter: {dict(post_filter_stats)}  |  "
                    f"retry_queue={len(pending_retry)}"
                )
                return final_batch

            print(f"{current_prompts=} < {rollout_batch_size=}")
            max_try = self.config.trainer.max_try_make_batch
            if max_try > 0 and (len(kept_samples) + len(pending_retry)) >= max_try:
                raise RuntimeError(
                    f"max_try_make_batch={max_try} reached. "
                    f"Collected {current_prompts}/{rollout_batch_size}."
                )


    def _log_with_progress(self, data: dict[str, Any], step: int) -> None:
        """Log metrics with training progress info injected into the dict."""
        # Compute current epoch (1-indexed for human readability)
        epoch = (step // max(self.steps_per_epoch, 1)) + 1
        progress = {
            "global_step": step,
            "total_steps": self.training_steps,
            "epoch": epoch,
            "total_epochs": self.config.trainer.total_epochs,
        }
        self.logger.log(data={**progress, **data}, step=step)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self._log_with_progress(val_metrics, self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics_reduced = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics_reduced)

                        # compute per-app and per-action-type reward metrics
                        if "app_name" in batch.non_tensor_batch and "action_type" in batch.non_tensor_batch:
                            per_group_metrics = compute_per_group_reward_metrics(
                                reward_metrics=reward_metrics,
                                group_keys={
                                    "app_name": batch.non_tensor_batch["app_name"],
                                    "action_type": batch.non_tensor_batch["action_type"],
                                },
                                prefix="reward",
                            )
                            metrics.update(per_group_metrics)

                    # ── Always compute and log batch/zv_raw/* zv metrics ─────────────────
                    # This is unconditional: we want to monitor has_gradient_ratio
                    # even when zv_filter is disabled (use_zv_filter=False).
                    _token_scores = batch.batch.get("token_level_scores")
                    if _token_scores is not None:
                        _scores_cpu = _token_scores.detach().float().cpu()
                        _seq_scores = _scores_cpu.sum(dim=-1)  # (total_rollouts,)
                        _n = self.config.worker.rollout.n
                        _total_rollouts = _seq_scores.shape[0]
                        _total_prompts = _total_rollouts // _n

                        # Collect scores per group
                        _uids = batch.non_tensor_batch.get("uid")
                        _uid2scores: dict[str, list[float]] = defaultdict(list)
                        if _uids is not None and len(_uids) == _total_rollouts:
                            for _uid, _sc in zip(_uids, _seq_scores.tolist()):
                                _uid2scores[str(_uid)].append(_sc)
                        else:
                            # Fallback: use index-based grouping
                            for _i in range(_total_prompts):
                                _uid = f"p{_i}"
                                _uid2scores[_uid] = _seq_scores[_i * _n:(_i + 1) * _n].tolist()

                        _atw = self.config.worker.reward.reward_function_kwargs.get("action_type_weight", 0.5)
                        _dist_lo = _atw * 0.5
                        _dist_hi = _atw * 1.1

                        _zv_raw = {"total_prompts": 0, "total_rollouts": 0, "non_zv": 0,
                                    "perfect": 0, "wrong": 0, "type_only": 0, "partial": 0}
                        _zv_raw["total_prompts"] = _total_prompts
                        _zv_raw["total_rollouts"] = _total_rollouts

                        for _sc_list in _uid2scores.values():
                            if len(_sc_list) < _n:
                                continue
                            _sa = np.array(_sc_list, dtype=np.float64)
                            _std_s = float(_sa.std(ddof=0))
                            _mean_s = float(_sa.mean())
                            if _std_s >= 1e-4:
                                _zv_raw["non_zv"] += 1
                            elif _mean_s >= 0.99:
                                _zv_raw["perfect"] += 1
                            elif _mean_s < _dist_lo:
                                _zv_raw["wrong"] += 1
                            elif _mean_s <= _dist_hi:
                                _zv_raw["type_only"] += 1
                            else:
                                _zv_raw["partial"] += 1

                        _tp = max(_zv_raw["total_prompts"], 1)
                        _tr = max(_zv_raw["total_rollouts"], 1)
                        metrics["batch/zv_raw/has_gradient_ratio"] = _zv_raw["non_zv"] / _tp
                        metrics["batch/zv_raw/zero_variance_ratio"] = (
                            _zv_raw["perfect"] + _zv_raw["wrong"] + _zv_raw["type_only"] + _zv_raw["partial"]
                        ) / _tp
                        metrics["batch/zv_raw/group_count/total"] = _zv_raw["total_prompts"]
                        metrics["batch/zv_raw/group_count/non_zv"] = _zv_raw["non_zv"]
                        metrics["batch/zv_raw/group_count/perfect"] = _zv_raw["perfect"]
                        metrics["batch/zv_raw/group_count/wrong"] = _zv_raw["wrong"]
                        metrics["batch/zv_raw/group_count/type_only"] = _zv_raw["type_only"]
                        metrics["batch/zv_raw/group_count/partial"] = _zv_raw["partial"]
                        metrics["batch/zv_raw/group_ratio/non_zv"] = _zv_raw["non_zv"] / _tp
                        metrics["batch/zv_raw/group_ratio/perfect"] = _zv_raw["perfect"] / _tp
                        metrics["batch/zv_raw/group_ratio/wrong"] = _zv_raw["wrong"] / _tp
                        metrics["batch/zv_raw/group_ratio/type_only"] = _zv_raw["type_only"] / _tp
                        metrics["batch/zv_raw/group_ratio/partial"] = _zv_raw["partial"] / _tp

                    # Log training generations if enabled
                    self._maybe_log_train_generations(batch)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            # compute rollout group metrics (all-fail / zero-variance / score_dist …)
            if self.config.worker.rollout.n > 1:
                _atw = self.config.worker.reward.reward_function_kwargs.get("action_type_weight", 0.5)
                metrics.update(compute_rollout_group_metrics(
                    batch=batch,
                    n=self.config.worker.rollout.n,
                    success_threshold=0.9,
                    prefix="rollout",
                    action_type_weight=_atw,
                ))

            # Training reward for wandb when scores were precomputed in _make_batch_data
            # (zv / online_filter): fit() skips compute_reward.remote, so reward/* was
            # never merged.  critic/score/* still logs the same mean but users look for
            # reward/ or val-style names.
            token_scores = batch.batch.get("token_level_scores")
            if token_scores is not None:
                seq = token_scores.sum(-1).detach().float()
                if seq.numel() > 0:
                    mean_r = seq.mean().item()
                    metrics.setdefault("reward/overall", mean_r)
                    if seq.numel() > 1:
                        metrics.setdefault("reward/overall_std", seq.std(unbiased=False).item())
                    metrics.setdefault("reward/overall_max", seq.max().item())
                    metrics.setdefault("reward/overall_min", seq.min().item())
                    metrics.setdefault("train/reward_score", mean_r)
                    metrics.setdefault("train/overall_reward", mean_r)

            self._log_with_progress(metrics, self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self._log_with_progress(val_metrics, self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
