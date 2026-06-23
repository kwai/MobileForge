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

import base64
import json
import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # 加载 system prompt（支持 jinja2 模板或纯文本）
        self.system_prompt = None
        if system_prompt:
            with open(system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages = []

        # 添加 system message（如果配置了 system_prompt）
        if self.system_prompt:
            system_prompt_template = Template(self.system_prompt.strip())
            # 支持在 system prompt 中使用模板变量（虽然通常不需要）
            system_content = system_prompt_template.render(content="", **example)
            messages.append({"role": "system", "content": system_content})

        # 构建 user message
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # 传递完整的 example 数据到模板，支持使用 instruction, history 等字段
            # 同时保持 content 变量的兼容性
            prompt_str = format_prompt.render(content=prompt_str, **example)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages.append({"role": "user", "content": content_list})
        else:
            messages.append({"role": "user", "content": prompt_str})

        return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            # 支持单个图像（非列表）的情况
            if not isinstance(images, (list, tuple)):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        image_size = None  # 用于 GUI-R1 数据集的坐标缩放（原始图片尺寸）
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            # 支持单个图像（非列表）的情况
            if not isinstance(images, (list, tuple)):
                images = [images]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            # 先获取原始图片尺寸（用于 GUI-R1 数据集的坐标转换）
            # gt_bbox 是基于原始图片尺寸的像素坐标，需要用原始尺寸进行 0-1000 归一化
            if len(images) > 0:
                first_image = images[0]
                if isinstance(first_image, str):
                    with Image.open(first_image) as img:
                        image_size = (img.width, img.height)
                elif isinstance(first_image, dict) and "bytes" in first_image:
                    with Image.open(BytesIO(first_image["bytes"])) as img:
                        image_size = (img.width, img.height)
                elif isinstance(first_image, bytes):
                    with Image.open(BytesIO(first_image)) as img:
                        image_size = (img.width, img.height)
                elif hasattr(first_image, "width") and hasattr(first_image, "height"):
                    image_size = (first_image.width, first_image.height)

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids

        # 支持 GUI-R1 数据集的特殊格式：
        # 优先检查是否有 gt_bbox 和 gt_action（GUI-R1 格式）
        # 如果有则组合构建 ground_truth，否则使用 answer_key
        if "gt_bbox" in example and "gt_action" in example:
            # GUI-R1 数据集格式：需要根据图片大小缩放 gt_bbox
            gt_bbox = list(example.pop("gt_bbox", [0, 0]))  # 转为列表以便修改
            gt_action = example.pop("gt_action", "click")
            gt_input_text = example.pop("gt_input_text", "")

            # 记录原始坐标是否为归一化格式（0-1 范围）
            # 需要检查所有坐标值（2个或4个）是否都在 0-1 范围内
            # GUI-R1 数据的坐标通常是像素坐标（非归一化）
            is_normalized_coord = len(gt_bbox) >= 2 and all(0 <= coord <= 1 for coord in gt_bbox)

            # bbox_valid：检查坐标是否有效（非负数表示有效）
            bbox_valid = len(gt_bbox) >= 2 and all(coord >= 0 for coord in gt_bbox)

            # 根据处理后的图片大小缩放 gt_bbox（如果坐标是归一化的 0-1 范围）
            if image_size is not None and len(gt_bbox) >= 2 and is_normalized_coord:
                scale_x, scale_y = image_size
                gt_bbox[0] *= scale_x
                gt_bbox[1] *= scale_y
                if len(gt_bbox) > 2:
                    gt_bbox[2] *= scale_x
                if len(gt_bbox) > 3:
                    gt_bbox[3] *= scale_y

            # 构建 ground_truth JSON，包含图片大小信息用于奖励函数中的坐标转换
            gt = {
                "action": gt_action,
                "gt_bbox": gt_bbox,
                "input_text": gt_input_text,
                # 添加图片大小信息，供奖励函数进行 0-1000 坐标转换
                "image_size": list(image_size) if image_size else None,
                # 标记坐标是否有效（[-0.1, -0.1] 等特殊坐标无效）
                "bbox_valid": bbox_valid,
                # 标记坐标是否为归一化格式（用于奖励函数中的方向判断）
                # GUI-R1 数据为 False，MobileForge 数据为 True
                "is_normalized": False,  # GUI-R1 格式默认不是归一化的
            }
            example["ground_truth"] = json.dumps(gt)
        elif self.answer_key in example:
            # 使用 answer_key 指定的字段
            example["ground_truth"] = example.pop(self.answer_key)
        else:
            # 如果都没有，设置为空字符串避免报错
            example["ground_truth"] = ""

        return example


def _remove_evaluation_hints(text: str) -> str:
    """
    从文本中删除 EVALUATION HINTS FROM PREVIOUS ATTEMPTS 块
    
    Hint 块格式：
    ================================================================================
    EVALUATION HINTS FROM PREVIOUS ATTEMPTS
    ================================================================================
    ... hint 内容 ...
    ================================================================================
    END OF EVALUATION HINTS
    ================================================================================
    
    Args:
        text: 原始文本
        
    Returns:
        删除 hint 块后的文本
    """
    import re
    
    # 匹配完整的 hint 块（包括开始和结束标记）
    # 使用非贪婪匹配来处理可能的多个块
    pattern = (
        r"={10,}\s*\n"  # 开始分隔线（10个或更多=号）
        r"\s*EVALUATION HINTS FROM PREVIOUS ATTEMPTS\s*\n"  # 标题
        r"={10,}\s*\n"  # 分隔线
        r".*?"  # hint 内容（非贪婪）
        r"={10,}\s*\n"  # 分隔线
        r"\s*END OF EVALUATION HINTS\s*\n"  # 结束标记
        r"={10,}\s*"  # 结束分隔线
    )
    
    # 删除 hint 块
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # 清理可能产生的多余空行
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    
    return cleaned_text.strip()


def _extract_hint_text(text: str) -> str:
    """
    从文本中提取 EVALUATION HINTS FROM PREVIOUS ATTEMPTS 块的内容（不含标记）

    Hint 块格式：
    ================================================================================
    EVALUATION HINTS FROM PREVIOUS ATTEMPTS
    ================================================================================
    ... hint 内容 ...
    ================================================================================
    END OF EVALUATION HINTS
    ================================================================================

    Args:
        text: 原始文本

    Returns:
        hint 内容字符串（不含标记），如果没有 hint 块则返回空字符串
    """
    import re

    pattern = (
        r"={10,}\s*\n"
        r"\s*EVALUATION HINTS FROM PREVIOUS ATTEMPTS\s*\n"
        r"={10,}\s*\n"
        r"(.*?)"
        r"={10,}\s*\n"
        r"\s*END OF EVALUATION HINTS\s*\n"
        r"={10,}\s*"
    )

    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _insert_hint_into_text(text: str, hint: str) -> str:
    """
    将 hint 内容重新插入到文本末尾（用于 retry with hint）

    Hint 被插入到 user 消息的末尾、"END OF PROMPT" 标记之前。

    Args:
        text: 原始文本（已不含 hint）
        hint: hint 内容

    Returns:
        重新插入 hint 后的文本
    """
    import re

    # 查找 END OF PROMPT 标记的位置
    end_pattern = r"(={5,}\s*END OF PROMPT\s*={5,})"
    match = re.search(end_pattern, text)

    if match:
        # 在 END OF PROMPT 之前插入 hint
        insert_pos = match.start()
        hint_block = (
            "\n\n"
            + "=" * 10 + "\n"
            + "EVALUATION HINTS FROM PREVIOUS ATTEMPTS\n"
            + "=" * 10 + "\n"
            + hint + "\n"
            + "=" * 10 + "\n"
            + "END OF EVALUATION HINTS\n"
            + "=" * 10
        )
        return text[:insert_pos] + hint_block + text[insert_pos:]
    else:
        # 如果没有 END OF PROMPT，追加到末尾
        hint_block = (
            "\n\n"
            + "=" * 10 + "\n"
            + "EVALUATION HINTS FROM PREVIOUS ATTEMPTS\n"
            + "=" * 10 + "\n"
            + hint + "\n"
            + "=" * 10 + "\n"
            + "END OF EVALUATION HINTS\n"
            + "=" * 10
        )
        return text + hint_block


def _extract_raw_response(conversations: list) -> str:
    """
    从 MobileForge 数据的 conversations 中提取原始的 assistant response

    用于在训练日志中显示完整的 assistant 响应（包含 thinking, tool_call, conclusion）
    注意：取 **最后一个** assistant 消息（多轮对话时前面的是历史动作）
    """
    raw_response = ""
    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", [])
            if isinstance(content, str):
                raw_response = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                raw_response = "".join(text_parts)
    return raw_response


def _parse_ground_truth_from_conversations(conversations: list) -> dict:
    """
    从 MobileForge 数据的 conversations 中解析 ground_truth 结构

    直接使用 r1gui_qwen3vl_system.jinja 中定义的动作空间，不做任何动作名称映射：
    click, long_press, swipe, type, answer, system_button, wait, terminate

    返回的 ground_truth 格式：
    {
        "action": "click",          # 动作类型（保持原始名称）
        "gt_bbox": [x, y],          # coordinate（0-1000 归一化）
        "gt_bbox2": [x2, y2],       # coordinate2（仅 swipe 使用）
        "input_text": "",           # 文本内容（type, answer）
        "button": "",               # 按钮名称（system_button: Back/Home/Menu/Enter）
        "direction": "",            # 滑动方向（swipe: up/down/left/right）
        "status": "",               # 终止状态（terminate: success/failure）
        "time": 0,                  # 时间参数（long_press, wait）
        "is_normalized": True,      # 坐标已归一化标记
        "bbox_valid": True          # 坐标是否有效
    }
    """
    import re as _re

    # 默认的空 ground_truth
    gt = {
        "action": "",
        "gt_bbox": [],
        "gt_bbox2": [],
        "input_text": "",
        "button": "",
        "direction": "",
        "status": "",
        "time": 0,
        "is_normalized": True,  # MobileForge 数据的坐标是 0-1000 归一化的
        "bbox_valid": True,
    }

    # 从 assistant 消息中提取文本
    # 注意：取 **最后一个** assistant 消息（即目标动作）
    # 因为 conversations 可能包含多轮对话，前面的 assistant 消息是历史动作
    assistant_text = ""
    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", [])
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                assistant_text = "".join(text_parts)
            # 不 break，继续遍历以找到最后一个 assistant 消息

    if not assistant_text:
        return gt

    # 解析 <tool_call> 标签中的 JSON
    tool_call_match = _re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", assistant_text, _re.DOTALL)
    if not tool_call_match:
        return gt

    try:
        tool_call = json.loads(tool_call_match.group(1))
        args = tool_call.get("arguments", tool_call)

        # 提取 action（保持原始名称，不做映射）
        action = args.get("action", "")
        gt["action"] = action

        # 提取 coordinate
        coord = args.get("coordinate", [])
        if coord and len(coord) >= 2:
            gt["gt_bbox"] = [coord[0], coord[1]]

        # swipe: 提取 coordinate2 和方向（保持 action="swipe"，不映射为 scroll）
        if action == "swipe":
            coord2 = args.get("coordinate2", [])
            if coord2 and len(coord2) >= 2:
                gt["gt_bbox2"] = [coord2[0], coord2[1]]
            # 从坐标计算方向（手指移动方向）
            if coord and coord2 and len(coord) >= 2 and len(coord2) >= 2:
                dx = coord2[0] - coord[0]
                dy = coord2[1] - coord[1]
                if abs(dx) > abs(dy):
                    gt["direction"] = "right" if dx > 0 else "left"
                else:
                    gt["direction"] = "down" if dy > 0 else "up"
            elif args.get("direction"):
                gt["direction"] = args.get("direction", "")

        # type: 提取文本
        if action == "type":
            gt["input_text"] = args.get("text", "")

        # answer: 提取文本
        if action == "answer":
            gt["input_text"] = args.get("text", "")

        # system_button: 提取按钮名称（保持 action="system_button"，不映射为 press_*）
        if action == "system_button":
            gt["button"] = args.get("button", "")

        # terminate: 提取状态（保持 action="terminate"，不映射为 complete）
        if action == "terminate":
            gt["status"] = args.get("status", "")

        # long_press: 提取时间参数
        if action == "long_press":
            time_val = args.get("time", 0)
            if time_val:
                gt["time"] = time_val

        # wait: 提取时间参数
        if action == "wait":
            time_val = args.get("time", 0)
            if time_val:
                gt["time"] = time_val

        return gt
    except (json.JSONDecodeError, TypeError, KeyError):
        return gt


# ────────────────────────────────────────────────────────────
# 训练数据加载时的筛选工具函数
# 与 data_analyzer/filters.py 的逻辑保持一致
# ────────────────────────────────────────────────────────────


def _build_task_index(all_samples: List[dict]) -> dict:
    """
    将扁平的 step-level 样本列表组织为
      { task_id: { attempt_id: [step_sample, ...] } }
    同时计算每个 task 的 avg_sr 等聚合信息。
    """
    tasks: dict = {}
    for s in all_samples:
        meta = s.get("metadata", {})
        tid = s.get("task_id", meta.get("task_id", ""))
        aid = meta.get("attempt_id", s.get("attempt_id", ""))
        if tid not in tasks:
            tasks[tid] = {"attempts": {}, "attempt_stats": {}}
        if aid not in tasks[tid]["attempts"]:
            # Attempt 是否成功以 final_result 判定（1=success, 0=failure；缺失按失败）
            final_result = meta.get("final_result", 0)
            attempt_success = int(final_result or 0) == 1
            tasks[tid]["attempts"][aid] = {
                "steps": [],
                "overall_success": attempt_success,
                "final_result": final_result,
                "task_feasible": meta.get("task_feasible", None),
                "max_consecutive_same_actions": meta.get("max_consecutive_same_actions", 0),
                "avg_sr": meta.get("avg_sr", 0.0),
            }
        tasks[tid]["attempts"][aid]["steps"].append(s)

    # 计算任务级 avg_sr（按 final_result=1 统计成功 attempt）
    for tid, tdata in tasks.items():
        atts = tdata["attempts"]
        total = len(atts)
        ok = sum(1 for a in atts.values() if a.get("overall_success"))
        tdata["avg_sr"] = ok / total if total > 0 else 0.0
    return tasks


def _flatten_tasks(tasks: dict) -> List[dict]:
    """将 task_index 扁平化回 step-level 样本列表"""
    result = []
    for tdata in tasks.values():
        for adata in tdata["attempts"].values():
            result.extend(adata["steps"])
    return result


def _filter_loop_attempts(tasks: dict, k: int) -> Tuple[dict, int]:
    """
    剔除死循环 attempt（连续 >= k 次相同 action）
    与 data_analyzer/filters.py DataFilter.remove_loops 逻辑一致
    """
    removed = 0
    for tid in list(tasks.keys()):
        tdata = tasks[tid]
        to_del = []
        for aid, adata in tdata["attempts"].items():
            if adata.get("max_consecutive_same_actions", 0) >= k:
                to_del.append(aid)
                removed += 1
        for aid in to_del:
            del tdata["attempts"][aid]
        # 移除空任务
        if not tdata["attempts"]:
            del tasks[tid]
    return tasks, removed


def _filter_best_trajectory(tasks: dict) -> Tuple[dict, int]:
    """
    每个任务仅保留最优 attempt:
    1. 优先保留成功 attempt（final_result==1，即 overall_success=True 的）
    2. 其次保留 positive 步骤占比最高的
    与 data_analyzer/filters.py DataFilter.best_trajectory 逻辑一致
    """
    removed = 0
    for tid in list(tasks.keys()):
        tdata = tasks[tid]
        atts = tdata["attempts"]
        if len(atts) <= 1:
            continue
        scored = []
        for aid, adata in atts.items():
            steps = adata["steps"]
            pos = sum(
                1 for s in steps if s.get("metadata", {}).get("impact") == "positive" or s.get("is_positive", False)
            )
            ratio = pos / len(steps) if steps else 0
            scored.append((aid, adata["overall_success"], ratio))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_aid = scored[0][0]
        for aid in [s[0] for s in scored[1:]]:
            del atts[aid]
            removed += 1
    # 移除空任务
    tasks = {k: v for k, v in tasks.items() if v["attempts"]}
    return tasks, removed


def _extract_hard_task_best_path(tasks: dict) -> Tuple[List[dict], int]:
    """
    针对"全部失败"的 hard task，从各 attempt 中选出从头部开始连续
    step_success=True 最长的那个 attempt，并截取其成功前缀步骤作为额外训练数据。

    定义：
    - hard task: 该任务所有 attempt 均失败（final_result!=1 → avg_sr == 0）
    - 最长成功路径: 对某个 attempt 按 step_number 排序后，从第 1 步开始连续
      step_success=True 的最长前缀长度（遇到第一个 step_success=False 即截止）
    - best attempt: 最长成功路径最长的 attempt；若相等则取步骤总数最多的 attempt

    返回值：
    - extra_samples: 从 hard tasks 中提取的步骤列表（已截取到成功前缀）
    - hard_task_count: 处理了多少个 hard task
    """
    extra_samples: List[dict] = []
    hard_task_count = 0

    for tid, tdata in tasks.items():
        # 只处理 avg_sr == 0.0 的 hard task（所有 attempt 均失败）
        if tdata.get("avg_sr", 0.0) > 0.0:
            continue

        atts = tdata["attempts"]
        if not atts:
            continue

        hard_task_count += 1
        best_aid = None
        best_prefix_len = -1
        best_total_steps = -1

        for aid, adata in atts.items():
            steps = sorted(adata["steps"], key=lambda s: s.get("metadata", {}).get("step_number", 0))
            # 计算从头部开始连续 step_success=True 的长度
            prefix_len = 0
            for step in steps:
                if step.get("metadata", {}).get("step_success", False):
                    prefix_len += 1
                else:
                    break

            # 选最长前缀；若相等，选步骤总数更多的（包含更多上下文）
            if (prefix_len > best_prefix_len) or (
                prefix_len == best_prefix_len and len(steps) > best_total_steps
            ):
                best_prefix_len = prefix_len
                best_total_steps = len(steps)
                best_aid = aid

        if best_aid is None or best_prefix_len == 0:
            # 没有任何成功步骤，跳过
            continue

        best_steps = sorted(
            atts[best_aid]["steps"],
            key=lambda s: s.get("metadata", {}).get("step_number", 0),
        )
        success_prefix = best_steps[:best_prefix_len]
        extra_samples.extend(success_prefix)

    return extra_samples, hard_task_count


def _filter_infeasible(tasks: dict, k: int) -> Tuple[dict, int]:
    """
    Infeasible 任务剔除: 同一任务中 infeasible 投票 >= k → 整体剔除
    与 data_analyzer/filters.py DataFilter.remove_infeasible 逻辑一致
    """
    removed = 0
    to_del = []
    for tid, tdata in tasks.items():
        inf_count = sum(1 for a in tdata["attempts"].values() if a.get("task_feasible") is False)
        if inf_count >= k:
            to_del.append(tid)
    for tid in to_del:
        removed += len(tasks[tid]["attempts"])
        del tasks[tid]
    return tasks, removed


def _filter_by_sr(tasks: dict, sr_min: float, sr_max: float) -> Tuple[dict, int]:
    """
    SR 范围筛选: 保留 avg_sr ∈ [sr_min, sr_max] 的任务
    与 data_analyzer/filters.py DataFilter.filter_by_sr 逻辑一致
    """
    removed = 0
    to_del = []
    for tid, tdata in tasks.items():
        sr = tdata.get("avg_sr", 0.0)
        if sr < sr_min or sr > sr_max:
            to_del.append(tid)
    for tid in to_del:
        removed += len(tasks[tid]["attempts"])
        del tasks[tid]
    return tasks, removed


def _load_json_files(data_path: str) -> List[dict]:
    """
    Load MobileForge JSON data from a file path or directory.

    Supports:
    1. Single JSON file: /path/to/mobileforge_grpo_xxx.json
    2. Directory: /path/to/session_xxx/ - loads all mobileforge_grpo_*.json files;
       if none found, loads all *.json files (excluding stats/summary files)
    3. Comma-separated paths: /path/to/file1.json,/path/to/file2.json

    Args:
        data_path: File path, directory path, or comma-separated paths

    Returns:
        Merged list of all samples
    """
    import glob as _glob

    all_samples = []

    # Check if comma-separated paths
    if "," in data_path and not os.path.exists(data_path):
        paths = [p.strip() for p in data_path.split(",") if p.strip()]
    else:
        paths = [data_path]

    for path in paths:
        if os.path.isfile(path):
            print(f"[MobileForge] Loading file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_samples.extend(data)
            else:
                print(f"[MobileForge] Warning: {path} does not contain a list, skipping")
        elif os.path.isdir(path):
            # Try mobileforge_grpo_*.json first
            grpo_files = sorted(_glob.glob(os.path.join(path, "mobileforge_grpo_*.json")))
            if not grpo_files:
                # Fallback: load all *.json excluding known non-data files
                exclude_patterns = {"session_summary.json", "README.md"}
                all_json = sorted(_glob.glob(os.path.join(path, "*.json")))
                grpo_files = [
                    f
                    for f in all_json
                    if os.path.basename(f) not in exclude_patterns
                    and "stats" not in os.path.basename(f)
                    and "error_analysis" not in os.path.basename(f)
                ]
            if not grpo_files:
                print(f"[MobileForge] Warning: No JSON data files found in {path}")
                continue
            for fp in grpo_files:
                print(f"[MobileForge] Loading file: {fp}")
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
                else:
                    print(f"[MobileForge] Warning: {fp} does not contain a list, skipping")
        else:
            raise FileNotFoundError(f"[MobileForge] Data path not found: {path}")

    return all_samples


def load_mobileforge_data(
    data_path: str,
    positive_only: bool = True,
    # ── Filtering params (aligned with data_analyzer/filters.py) ──
    filter_loop_threshold: int = 0,
    filter_best_trajectory: bool = False,
    filter_infeasible_k: int = 0,
    filter_sr_min: float = -1.0,
    filter_sr_max: float = -1.0,
    filter_keep_hard_task_best_path: bool = False,
) -> List[dict]:
    """
    Load MobileForge format data and apply filtering.

    Training and validation data should be prepared as separate files
    by mobileforge_data_processor.py and passed independently via
    --data_path (train) and --val_data_path (val).

    Supports loading from:
    - Single JSON file path
    - Directory containing JSON files (from mobileforge_data_processor.py output)
    - Comma-separated file paths

    Filtering pipeline:
    1. Loop attempt removal (filter_loop_threshold >= 1)
    2. Best trajectory selection (filter_best_trajectory)
    3. Infeasible task removal (filter_infeasible_k >= 1)
    4. SR range filtering (filter_sr_min >= 0 && filter_sr_max >= 0)
    4.5 Hard task best path extraction (filter_keep_hard_task_best_path)
        - Before SR filtering, extract the longest consecutive success prefix
          from the best attempt of tasks where all attempts failed (avg_sr=0).
        - These extra steps are merged back after SR filtering.
    5. Positive-only filtering (positive_only)

    Args:
        data_path: Data file path, directory, or comma-separated paths
        positive_only: Whether to load only positive samples
        filter_loop_threshold: Loop removal threshold (0=disabled)
        filter_best_trajectory: Whether to keep only the best trajectory
        filter_infeasible_k: Infeasible vote threshold (0=disabled)
        filter_sr_min: SR filter lower bound (<0=disabled)
        filter_sr_max: SR filter upper bound (<0=disabled)
        filter_keep_hard_task_best_path: For tasks where ALL attempts failed
            (avg_sr=0), extract the longest consecutive success prefix (from
            step 1) of the best attempt and add those steps to training data.
            These extra steps bypass the SR range filter so that hard tasks
            can contribute training signal even when filtered out by sr_min>0.

    Returns:
        List of samples
    """
    # Load raw data (supports file, directory, or comma-separated paths)
    all_samples = _load_json_files(data_path)

    print(f"[MobileForge] Raw sample count: {len(all_samples)}")

    # ── Extract/compute necessary fields for each sample ──
    for sample in all_samples:
        metadata = sample.get("metadata", {})

        if "task_id" not in sample:
            sample["task_id"] = metadata.get("task_id", f"unknown_{id(sample)}")
        if "step_number" not in sample:
            sample["step_number"] = metadata.get("step_number", 0)
        if "is_positive" not in sample:
            impact = metadata.get("impact", "")
            sample["is_positive"] = impact == "positive"
        if "ground_truth" not in sample:
            sample["ground_truth"] = _parse_ground_truth_from_conversations(sample.get("conversations", []))
        if "raw_response" not in sample:
            sample["raw_response"] = _extract_raw_response(sample.get("conversations", []))

    print(f"[MobileForge] Positive samples: {sum(1 for s in all_samples if s.get('is_positive'))}/{len(all_samples)}")

    # ── Steps 1~4: Build task index, execute task/attempt level filtering ──
    need_task_filter = (
        (filter_loop_threshold > 0)
        or filter_best_trajectory
        or (filter_infeasible_k > 0)
        or (filter_sr_min >= 0 and filter_sr_max >= 0)
        or filter_keep_hard_task_best_path
    )

    hard_task_extra_samples: List[dict] = []

    if need_task_filter:
        tasks = _build_task_index(all_samples)
        print(f"[MobileForge] Built task index: {len(tasks)} tasks")

        # 1. Loop attempt removal
        if filter_loop_threshold > 0:
            tasks, n = _filter_loop_attempts(tasks, filter_loop_threshold)
            print(
                f"[MobileForge] Loop removal (k>={filter_loop_threshold}): removed {n} attempts, {len(tasks)} tasks remaining"
            )

        # 2. Best trajectory selection
        if filter_best_trajectory:
            tasks, n = _filter_best_trajectory(tasks)
            print(f"[MobileForge] Best trajectory: removed {n} attempts, {len(tasks)} tasks remaining")

        # 3. Infeasible task removal
        if filter_infeasible_k > 0:
            tasks, n = _filter_infeasible(tasks, filter_infeasible_k)
            print(
                f"[MobileForge] Infeasible removal (k>={filter_infeasible_k}): removed {n} attempts, {len(tasks)} tasks remaining"
            )

        # 4.5 Hard task best path extraction (BEFORE SR filter so hard tasks
        #     are still present in the task index when we compute avg_sr)
        if filter_keep_hard_task_best_path:
            hard_task_extra_samples, n_hard = _extract_hard_task_best_path(tasks)
            print(
                f"[MobileForge] Hard task best path: found {n_hard} hard tasks, "
                f"extracted {len(hard_task_extra_samples)} steps from their best attempts"
            )

        # 4. SR range filtering
        if filter_sr_min >= 0 and filter_sr_max >= 0:
            tasks, n = _filter_by_sr(tasks, filter_sr_min, filter_sr_max)
            print(
                f"[MobileForge] SR range filter [{filter_sr_min}, {filter_sr_max}]: removed {n} attempts, {len(tasks)} tasks remaining"
            )

        # Flatten back to sample list, then merge hard task extra steps
        all_samples = _flatten_tasks(tasks)
        if hard_task_extra_samples:
            all_samples = all_samples + hard_task_extra_samples
            print(
                f"[MobileForge] After filtering + hard task best path: {len(all_samples)} samples "
                f"({len(hard_task_extra_samples)} from hard tasks)"
            )
        else:
            print(f"[MobileForge] After filtering: {len(all_samples)} samples")

    # ── Step 5: Positive-only filtering ──
    before = len(all_samples)
    if positive_only:
        all_samples = [s for s in all_samples if s.get("is_positive", False)]
        print(f"[MobileForge] Positive-only filter: {before} -> {len(all_samples)}")

    # ── Summary ──
    task_ids = list(set(s["task_id"] for s in all_samples))
    print(f"[MobileForge] Final: {len(all_samples)} samples, {len(task_ids)} tasks")

    if len(all_samples) == 0:
        print(f"[MobileForge Error] No samples after filtering!")
        print(f"  - positive_only={positive_only}")
        print(f"  - filter_loop_threshold={filter_loop_threshold}")
        print(f"  - filter_best_trajectory={filter_best_trajectory}")
        print(f"  - filter_infeasible_k={filter_infeasible_k}")
        print(f"  - filter_sr_min={filter_sr_min}, filter_sr_max={filter_sr_max}")
        print(f"  - filter_keep_hard_task_best_path={filter_keep_hard_task_best_path}")

    return all_samples


class MobileForgeRLHFDataset(Dataset):
    """
    MobileForge RLHF Dataset

    Supports:
    - MobileForge format conversation data (conversations field)
    - base64 encoded images
    - Multiple composable training data filtering strategies (aligned with data_analyzer)

    Training and validation sets should be prepared as separate files
    by mobileforge_data_processor.py and loaded independently.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        positive_only: bool = True,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        remove_evaluation_hints: bool = False,
        # ── Filtering params (aligned with data_analyzer/filters.py) ──
        filter_loop_threshold: int = 0,
        filter_best_trajectory: bool = False,
        filter_infeasible_k: int = 0,
        filter_sr_min: float = -1.0,
        filter_sr_max: float = -1.0,
        filter_keep_hard_task_best_path: bool = False,
    ):
        """
        Args:
            data_path: MobileForge data file path, directory, or comma-separated paths
            tokenizer: tokenizer
            processor: multimodal processor
            positive_only: Whether to keep only positive samples
            max_prompt_length: Maximum prompt length
            truncation: Truncation strategy
            format_prompt: Format prompt template file
            system_prompt: System prompt file
            min_pixels: Minimum pixels
            max_pixels: Maximum pixels
            filter_overlong_prompts: Whether to filter overlong prompts
            remove_evaluation_hints: Whether to remove EVALUATION HINTS blocks from user prompts
            filter_loop_threshold: Loop removal threshold (0=disabled)
            filter_best_trajectory: Whether to keep only the best trajectory
            filter_infeasible_k: Infeasible vote threshold (0=disabled)
            filter_sr_min: SR filter lower bound (<0=disabled)
            filter_sr_max: SR filter upper bound (<0=disabled)
            filter_keep_hard_task_best_path: For tasks where ALL attempts failed,
                extract the longest consecutive success prefix of the best attempt
                and add those steps to training data (bypasses SR filter exclusion).
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.remove_evaluation_hints = remove_evaluation_hints

        # Load data and apply filtering
        self.samples = load_mobileforge_data(
            data_path=data_path,
            positive_only=positive_only,
            filter_loop_threshold=filter_loop_threshold,
            filter_best_trajectory=filter_best_trajectory,
            filter_infeasible_k=filter_infeasible_k,
            filter_sr_min=filter_sr_min,
            filter_sr_max=filter_sr_max,
            filter_keep_hard_task_best_path=filter_keep_hard_task_best_path,
        )

        print(f"[MobileForge] Loaded data: positive_only={positive_only}")
        print(
            f"[MobileForge] Filters: loop>={filter_loop_threshold}, best_traj={filter_best_trajectory}, "
            f"infeasible_k={filter_infeasible_k}, sr=[{filter_sr_min},{filter_sr_max}], "
            f"hard_task_best_path={filter_keep_hard_task_best_path}"
        )
        print(f"[MobileForge] Samples: {len(self.samples)}")

        # Load system prompt
        self.system_prompt = None
        if system_prompt:
            with open(system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read()

        # Load format prompt
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # Filter overlong prompts
        if filter_overlong_prompts:
            original_count = len(self.samples)
            self.samples = [s for s in self.samples if self._check_prompt_length(s)]
            print(f"[MobileForge] Overlong prompt filter: {original_count} -> {len(self.samples)}")

    def _check_prompt_length(self, sample: dict) -> bool:
        """检查 prompt 长度是否超过限制"""
        try:
            messages = self._build_messages(sample)
            if self.processor is not None:
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                # 获取图像
                images = self._get_images(sample)
                if images:
                    processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]
                    model_inputs = self.processor(
                        processed_images, [prompt], add_special_tokens=False, return_tensors="pt"
                    )
                    return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length
        except Exception:
            return False

    def _build_messages(self, sample: dict) -> list[dict]:
        """
        从 MobileForge 格式构建消息列表

        MobileForge 数据格式：
        - conversations 中已经包含完整的对话历史（含 action history）
        - 直接使用 conversations 中的 system 和 user 消息
        - 不需要重新用模板构建（因为 action history 已经包含在其中）
        """
        messages = []
        conversations = sample.get("conversations", [])

        # 直接使用 conversations 中的 system 和 user 消息
        # MobileForge 数据中 conversations 已经包含完整的 action history
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", [])

            # 跳过 assistant 消息（这是之前的响应，不是 prompt 的一部分）
            if role == "assistant":
                continue

            if role == "system":
                # 处理 system 消息
                if isinstance(content, str):
                    messages.append({"role": "system", "content": content})
                elif isinstance(content, list):
                    # 提取文本内容
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        messages.append({"role": "system", "content": "".join(text_parts)})

            elif role == "user":
                # 处理 user 消息（保留原始内容，包含 action history 和图像占位符）
                if isinstance(content, str):
                    # 如果需要删除 evaluation hints，先处理文本
                    if self.remove_evaluation_hints:
                        content = _remove_evaluation_hints(content)
                    
                    # 构建带图像的 user message
                    images = self._get_images(sample)
                    if images and "<image>" in content:
                        content_list = []
                        for i, text_part in enumerate(content.split("<image>")):
                            if i != 0:
                                content_list.append({"type": "image"})
                            if text_part:
                                content_list.append({"type": "text", "text": text_part})
                        messages.append({"role": "user", "content": content_list})
                    else:
                        messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # 内容已经是列表格式，提取文本
                    # 注意：MobileForge 数据中 text 已经包含 <image> 占位符
                    # 不需要从 image_url 类型再添加占位符
                    user_text = ""
                    has_image_in_list = False
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type", "")
                            if item_type == "text":
                                user_text += item.get("text", "")
                            elif item_type in ["image", "image_url"]:
                                has_image_in_list = True
                    
                    # 如果需要删除 evaluation hints，处理提取的文本
                    if self.remove_evaluation_hints:
                        user_text = _remove_evaluation_hints(user_text)

                    # 检查 text 中是否已有 <image> 占位符
                    has_image_placeholder = "<image>" in user_text

                    # 构建带图像的 user message
                    images = self._get_images(sample)
                    if images and (has_image_placeholder or has_image_in_list):
                        if has_image_placeholder:
                            # text 中已有占位符，直接使用
                            content_list = []
                            for i, text_part in enumerate(user_text.split("<image>")):
                                if i != 0:
                                    content_list.append({"type": "image"})
                                if text_part:
                                    content_list.append({"type": "text", "text": text_part})
                            messages.append({"role": "user", "content": content_list})
                        else:
                            # text 中没有占位符，在末尾添加图像
                            content_list = [{"type": "text", "text": user_text}]
                            for _ in images:
                                content_list.append({"type": "image"})
                            messages.append({"role": "user", "content": content_list})
                    else:
                        messages.append({"role": "user", "content": user_text})

        return messages

    def _get_images(self, sample: dict) -> List[ImageObject]:
        """从样本中获取图像"""
        images = []

        # 尝试从 conversations 中提取图像
        for conv in sample.get("conversations", []):
            content = conv.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        image_data = None

                        # 支持两种图像格式：
                        # 1. {"type": "image", "image": "base64..."}
                        # 2. {"type": "image_url", "image_url": {"url": "data:image;base64,..."}}
                        if item_type == "image":
                            image_data = item.get("image", "")
                        elif item_type == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                image_data = image_url.get("url", "")
                            elif isinstance(image_url, str):
                                image_data = image_url

                        if image_data:
                            try:
                                img = self._decode_base64_image(image_data)
                                if img:
                                    images.append(img)
                            except Exception:
                                pass

        # 也尝试从 images 字段获取
        if not images and "images" in sample:
            for img_data in sample["images"]:
                try:
                    img = self._decode_base64_image(img_data)
                    if img:
                        images.append(img)
                except Exception:
                    pass

        return images

    def _decode_base64_image(self, image_data: str) -> Optional[ImageObject]:
        """解码图像。

        支持：
        1. 本地图片路径（由 tools/extract_images_to_files.py 生成）
        2. data:image/...;base64,... 字符串
        3. 纯 base64 字符串
        """
        try:
            # Path-backed image format: JSON stores only the local file path to
            # avoid loading huge embedded base64 payloads into memory.
            if os.path.exists(image_data):
                image = Image.open(image_data)
                image.load()
                return image

            # 处理 data:image 格式
            if image_data.startswith("data:image"):
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image.load()
            return image
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = self._build_messages(sample)

        # 获取图像
        images = self._get_images(sample)
        image_size = None

        if images:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]

            if processed_images:
                image_size = (processed_images[0].width, processed_images[0].height)

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # 处理 position_ids (Qwen-VL mrope)
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
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

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        # 构建输出
        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "task_id": sample.get("task_id", ""),
            "step_number": sample.get("step_number", 0),
        }

        if images:
            example["multi_modal_data"] = {"images": images}

        # 构建 ground_truth
        ground_truth = sample.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            # 添加图片大小信息
            if image_size:
                ground_truth["image_size"] = list(image_size)
            example["ground_truth"] = json.dumps(ground_truth)
        elif isinstance(ground_truth, str):
            example["ground_truth"] = ground_truth
        else:
            example["ground_truth"] = ""

        # 添加原始 assistant response（用于日志显示）
        example["raw_response"] = sample.get("raw_response", "")

        # 添加 app_name 和 action_type 用于 per-group metrics
        metadata = sample.get("metadata", {})
        example["app_name"] = metadata.get("app", metadata.get("app_name", "unknown"))
        # action_type 从 ground_truth 中提取
        if isinstance(ground_truth, dict):
            example["action_type"] = ground_truth.get("action", "unknown")
        elif isinstance(ground_truth, str):
            try:
                gt_dict = json.loads(ground_truth)
                example["action_type"] = gt_dict.get("action", "unknown")
            except (json.JSONDecodeError, TypeError):
                example["action_type"] = "unknown"
        else:
            example["action_type"] = "unknown"

        # ── Adaptive Hint: store hint text for retry ──────────────────────────
        # Extract hint from the raw user text in conversations (BEFORE remove_evaluation_hints)
        hint_text = self._extract_hint_text_from_sample(sample)
        example["hint_text"] = hint_text  # "" if no hint exists
        # Store multi_modal_data for adaptive hint re-tokenization
        if images:
            example["_images_for_hint"] = images
        else:
            example["_images_for_hint"] = None

        return example

    def _extract_hint_text_from_sample(self, sample: dict) -> str:
        """从 sample 的 conversations 中提取 user 消息中包含的 EVALUATION HINTS 文本"""
        conversations = sample.get("conversations", [])
        for conv in conversations:
            if conv.get("role") == "user":
                content = conv.get("content", [])
                if isinstance(content, str):
                    return _extract_hint_text(content)
                elif isinstance(content, list):
                    user_text = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_text += item.get("text", "")
                    return _extract_hint_text(user_text)
        return ""
