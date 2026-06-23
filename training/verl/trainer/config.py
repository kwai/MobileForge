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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    """system prompt file path (jinja2 template or plain text), will be added as system role message"""
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    # MobileForge config
    use_mobileforge_format: bool = False
    """Use MobileForge data format. Train and val sets use separate data paths."""
    positive_only: bool = True
    """Keep only positive samples (impact=positive steps only)"""

    # ── 必须启用的筛选 ──
    filter_loop_threshold: int = 7
    """死循环剔除阈值: 某 attempt 中连续相同 action >= 此值时剔除该 attempt（<=0 表示不启用）"""

    # ── 可选筛选（设为 false / 0 / -1 则不启用） ──
    filter_best_trajectory: bool = False
    """是否只保留每个任务的最优轨迹（优先成功, 其次 positive 步骤占比最高）"""
    filter_infeasible_k: int = 0
    """Infeasible 任务剔除: 同一任务中 infeasible 投票 >= k 则整体剔除（0=不启用）"""
    filter_sr_min: float = -1.0
    """SR 范围筛选下界: 保留 avg_sr >= sr_min 的任务（<0 表示不启用）"""
    filter_sr_max: float = -1.0
    """SR 范围筛选上界: 保留 avg_sr <= sr_max 的任务（<0 表示不启用）"""
    filter_keep_hard_task_best_path: bool = False
    """Hard task 最优路径保留: 对所有尝试均失败（avg_sr=0）的任务，从各 attempt
    中选取从第 1 步开始连续 step_success=True 最长的那个 attempt，并截取其成功前缀
    步骤追加到训练集中。这些步骤会绕过 SR 过滤器，确保 hard task 也能贡献训练信号。"""
    
    # 数据预处理选项
    remove_evaluation_hints: bool = False
    """是否从 user prompt 中删除 EVALUATION HINTS FROM PREVIOUS ATTEMPTS 块"""

    # ── 自适应 Hint 策略 ──────────────────────────────────────────────────────
    adaptive_hint: bool = False
    """
    自适应 Hint 策略：
    - 默认 false：使用 remove_evaluation_hints 的设置决定是否包含 hints
    - true：首次 rollout 先不带 hint；如果 group 为 zero-variance，则重新带 hint rollout
    仅当 remove_evaluation_hints=false 时生效（因为 adaptive_hint 本质上是"动态决定是否用 hint"）
    """
    adaptive_hint_only_zv: bool = True
    """仅对 zero-variance 组启用 hint retry（保留 non_zv 组不带 hint 的自主探索）"""
    adaptive_hint_retry_wrong: bool = True
    """zv_wrong 组是否允许 retry with hint"""
    adaptive_hint_retry_perfect: bool = True
    """zv_perfect 组是否允许 retry with hint"""
    adaptive_hint_max_attempts: int = 2
    """带 hint 的最大重试次数（包含首次不带 hint），建议 2~3"""

    # ── Zero-variance 样本过滤 / 重试策略 ─────────────────────────────────────
    # 分类依据：group 内所有 rollout 的 reward std < zero_var_eps → advantage≈0
    #   zv_perfect : 均分 ≥ 0.99
    #   zv_wrong   : 均分 < action_type_weight * 0.5
    #   zv_type_only: 均分 ∈ [action_type_weight*0.5, action_type_weight*1.1]
    # 若启用了过滤，对应 group 的所有 rollout 在本次训练中被剔除（advantage=0，不参与梯度更新）
    exclude_zv_perfect: bool = False
    """剔除 zv_perfect 组（完全饱和，组内无对比信号）"""
    exclude_zv_wrong: bool = False
    """剔除 zv_wrong 组（动作类型完全错误，模型完全无法完成）"""

    # ── Once-only 模式（仅在对应 exclude=true 时生效）────────────────────────
    # 检测到 zv_perfect/wrong 的 task_id 后，在后续所有 epoch 中永久跳过该 task
    exclude_zv_perfect_once: bool = False
    """对 zv_perfect 采用 once-only 模式：epoch1 检测后，后续 epoch 永久跳过该 task"""
    exclude_zv_wrong_once: bool = False
    """对 zv_wrong 采用 once-only 模式：epoch1 检测后，后续 epoch 永久跳过该 task"""

    # ── zv_wrong 重试策略 ─────────────────────────────────────────────────────
    retry_zv_wrong: bool = False
    """对 zv_wrong 组进行重新采样（而非直接剔除），可与 exclude_zv_wrong_once 配合使用"""
    retry_zv_wrong_max_attempts: int = 3
    """zv_wrong 重试最大次数（包含首次采样），建议值 2~5"""

    # ── zv_perfect 重试策略 ────────────────────────────────────────────────────
    retry_zv_perfect: bool = False
    """对 zv_perfect 组进行重新采样（而非直接剔除），可与 exclude_zv_perfect_once 配合使用"""
    retry_zv_perfect_max_attempts: int = 3
    """zv_perfect 重试最大次数（包含首次采样），建议值 2~5"""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.system_prompt = get_abs_path(self.system_prompt, prompt="System prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        # Validation: once-only mode requires the corresponding exclude flag
        if self.exclude_zv_perfect_once and not self.exclude_zv_perfect:
            raise ValueError("exclude_zv_perfect_once=True requires exclude_zv_perfect=True")
        if self.exclude_zv_wrong_once and not self.exclude_zv_wrong:
            raise ValueError("exclude_zv_wrong_once=True requires exclude_zv_wrong=True")
        # Adaptive hint takes over wrong/partial group handling; incompatible with exclude_zv_wrong
        if self.adaptive_hint and self.exclude_zv_wrong:
            raise ValueError(
                "exclude_zv_wrong=True is incompatible with adaptive_hint=True. "
                "Adaptive hint handles wrong groups by retrying with hints; "
                "set exclude_zv_wrong=False when using adaptive_hint=True."
            )
        if self.adaptive_hint and self.retry_zv_wrong:
            raise ValueError(
                "retry_zv_wrong=True is incompatible with adaptive_hint=True. "
                "Adaptive hint already handles wrong groups by retrying with hints; "
                "set retry_zv_wrong=False when using adaptive_hint=True."
            )
        # When adaptive_hint is enabled, also disable perfect exclusion (adaptive hint doesn't handle perfect)
        if self.adaptive_hint and self.exclude_zv_perfect:
            raise ValueError(
                "exclude_zv_perfect=True is incompatible with adaptive_hint=True. "
                "Adaptive hint only retries wrong/partial groups with hints, not perfect groups. "
                "Set exclude_zv_perfect=False when using adaptive_hint=True."
            )


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation, -1 means log all"""
    train_generations_to_log: int = 0
    """number of generations to log for training, -1 means log all"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
