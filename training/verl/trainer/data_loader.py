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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import MobileForgeRLHFDataset, RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    """
    Create training and validation data loaders.

    Supports two modes:
    1. Standard mode: uses RLHFDataset, requires separate train_files and val_files
    2. MobileForge mode: uses MobileForgeRLHFDataset for both train and val,
       with separate data paths (train_files and val_files).
       Both sets use the same MobileForge format and reward function.
    """

    if config.use_mobileforge_format:
        # MobileForge mode: both train and val use MobileForge format with independent paths
        print("[MobileForge] Using MobileForge data format")
        print(f"[MobileForge] Train data: {config.train_files}")
        print(f"[MobileForge] Val data:   {config.val_files}")
        print(f"[MobileForge] Positive only: {config.positive_only}")

        # Filtering parameters (shared between train and val for consistency)
        filter_kwargs = dict(
            filter_loop_threshold=config.filter_loop_threshold,
            filter_best_trajectory=config.filter_best_trajectory,
            filter_infeasible_k=config.filter_infeasible_k,
            filter_sr_min=config.filter_sr_min,
            filter_sr_max=config.filter_sr_max,
            filter_keep_hard_task_best_path=config.filter_keep_hard_task_best_path,
        )
        print(f"[MobileForge] Filter params: {filter_kwargs}")

        print("[MobileForge] Loading training set...")
        train_dataset = MobileForgeRLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            positive_only=config.positive_only,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            remove_evaluation_hints=config.remove_evaluation_hints,
            **filter_kwargs,
        )

        # Validation set: independent data path, no filtering (keep original distribution)
        print("[MobileForge] Loading validation set (no filtering)...")
        val_dataset = MobileForgeRLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            positive_only=False,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            remove_evaluation_hints=config.remove_evaluation_hints,
        )
    else:
        # Standard mode: use separate train and val files
        train_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )

        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            system_prompt=config.system_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

    # Create training dataloader
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # Create validation dataloader
    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1, f"Train dataloader is empty! Dataset size: {len(train_dataset)}"
    assert len(val_dataloader) >= 1, f"Val dataloader is empty! Dataset size: {len(val_dataset)}"
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
