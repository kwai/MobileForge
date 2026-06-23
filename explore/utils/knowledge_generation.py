"""
在执行完task后，调用 update_trajectory_to_knowledge 即可，如果需要马上用上新的知识库，可以再继续看看 memory.py
"""

import xml.etree.ElementTree as ET
import subprocess
import time
import re
import os
import re
import glob
import imagehash
import openai

# from dotenv import load_dotenv
from PIL import Image
import io
import os
import json
import uuid
import base64
import hashlib

from utils.utils import pil_to_webp_base64, cv2_to_pil
from typing import List, Any, Dict, Union
import cv2
import copy

# 如果版本升级，是否清空知识库重新生成
EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE = (
    os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE", "False").lower() == "false"
)
# 如果版本降级，是否清空知识库重新生成
EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE = (
    os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE", "True").lower() == "true"
)


from utils.memory import KnowledgeStore
from tqdm import tqdm
import math
import operator
from dataclasses import asdict
import numpy as np
from utils.utils import ndarray_to_webp_base64, resize_ndarray_image
from utils.device import UIElement
import imagehash
from PIL import Image
from utils.prompt_templates import KNOWLEDGE_EXTRACTOR
from utils.utils import openai_request
from utils.device import (
    _generate_ui_element_description,
    add_screenshot_label,
    add_ui_element_mark,
)


def pil_image_to_phash(pil_image: Image.Image) -> str:
    """Convert a PIL Image to a perceptual hash.

    Args:
        pil_image (Image.Image): The PIL Image object.

    Returns:
        str: The perceptual hash.
    """

    return str(imagehash.phash(pil_image, hash_size=16, highfreq_factor=8)).upper()


def ndarray_image_to_phash(ndarray_image: np.ndarray) -> str:
    """Convert a NumPy ndarray image to a perceptual hash.

    Args:
        ndarray_image (np.ndarray): The NumPy ndarray image.

    Returns:
        str: The perceptual hash.
    """
    return pil_image_to_phash(Image.fromarray(ndarray_image))


def dot_product(v1: list, v2: list) -> float:
    return sum(map(operator.mul, v1, v2))


def cosine_similarity(v1: list, v2: list) -> float:
    """越接近1越相似"""
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)


def update_trajectory_to_knowledge(
    trajectory_data: list[dict],
    locations: list[tuple[str, int]],
    fusion_memory: KnowledgeStore,
    knowledge_data: dict[str, dict],
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> None:
    """将轨迹数据转换为知识数据，并且更新到知识库（knowledge_data）中。需要在调用完这个函数之后手动保存更新后的knowledge_data，并且重新载入一次fusion memory（因为concat的消息尚未在memory中进行更新）

    Args:
        trajectory_data (list[dict]): 轨迹数据
        locations (list[tuple[str, int]]): 检索出来的index到knowledge_data位置的映射,val分别是package_name和index
        fusion_memory (KnowledgeStore): 知识库
        knowledge_data (dict[str, dict[str, Any]]): 知识数据
        usage (dict[str, int], optional): 本次调用使用的token数，会在这个函数中更新

    Returns:
        None
    """
    tmp_memory = None

    def is_transition_valid(
        before_screenshot: np.ndarray, after_screenshot: np.ndarray
    ) -> bool:
        """判断两个截图之间的转换是否有效"""
        return ndarray_image_to_phash(before_screenshot) != ndarray_image_to_phash(
            after_screenshot
        )

    for d in tqdm(trajectory_data, ncols=80, leave=False, desc="Updating knowledge"):
        before_screenshot = d["before_screenshot"]
        after_screenshot = d["after_screenshot"]
        if not is_transition_valid(before_screenshot, after_screenshot):
            continue
        pil_before_screenshot = Image.fromarray(before_screenshot).convert("RGB")
        task_description = d["goal"]
        numeric_tag_of_element = (
            d["converted_action"].index
            if hasattr(d["converted_action"], "index")
            else None
        )
        if numeric_tag_of_element is None:
            continue
        if d["target_element"] is None:
            continue
        e = UIElement(**d["target_element"])
        x_min, y_min, x_max, y_max = (
            e.bbox_pixels.x_min,
            e.bbox_pixels.y_min,
            e.bbox_pixels.x_max,
            e.bbox_pixels.y_max,
        )
        w, h = pil_before_screenshot.size
        x_min = int(max(x_min, 0))
        y_min = int(max(y_min, 0))
        x_max = int(min(x_max, w))
        y_max = int(min(y_max, h))
        image_patch = pil_before_screenshot.crop((x_min, y_min, x_max, y_max))
        logical_screen_size = (w, h)
        physical_frame_boundary = (0, 0, w, h)
        orientation = 0
        add_ui_element_mark(
            before_screenshot,
            e,
            numeric_tag_of_element,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
        add_screenshot_label(
            before_screenshot,
            "Before",
        )
        add_screenshot_label(
            after_screenshot,
            "After",
        )
        if tmp_memory is not None:
            res = tmp_memory.search(image_patch, k=1, similarity_threshold=0.99)
            if len(res) > 0:  # 说明在本次轨迹中已经被处理过
                continue
        res = fusion_memory.search(image_patch, k=1, similarity_threshold=0.99)
        ui_element_attributes = (
            _generate_ui_element_description(e, numeric_tag_of_element)
            if d["target_element"] is not None
            else "None"
        )
        action = d["converted_action"].json_str()
        package_name = d["top_app_package_name"]
        p = KNOWLEDGE_EXTRACTOR.format(
            task_description=task_description,
            numeric_tag_of_element=numeric_tag_of_element,
            ui_element_attributes=ui_element_attributes,
            action=action,
        )
        low_resolution = os.getenv("LOW_RESOLUTION", "False").lower() == "true"
        if low_resolution:
            before_screenshot = resize_ndarray_image(before_screenshot, 1000)
            after_screenshot = resize_ndarray_image(after_screenshot, 1000)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{ndarray_to_webp_base64(before_screenshot)}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{ndarray_to_webp_base64(after_screenshot)}",
                        },
                    },
                    {"type": "text", "text": p},
                ],
            },
        ]
        rsp_txt = openai_request(
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            timeout=120,
            usage=usage,
        )
        rsp_txt = rsp_txt.strip()
        assert rsp_txt != "", "empty response from MLLM"
        if not rsp_txt.endswith("."):
            rsp_txt += "."
        rsp_txt += " "
        if tmp_memory is None:
            tmp_memory = KnowledgeStore(
                knowledge_items=[{"image": np.array(image_patch), "info": rsp_txt}],
                embedding_pipeline=fusion_memory.embedding_pipeline,
            )
        else:
            tmp_memory.add_knowledge_items(
                [{"image": np.array(image_patch), "info": rsp_txt}]
            )

        add_as_new = True
        rsp_txt_embedding = fusion_memory.embedding_pipeline(rsp_txt)
        for r in res:
            add_as_new = False
            idx = r["index"]
            txt = r["knowledge"]
            txt_embedding = fusion_memory.embedding_pipeline(txt)
            similarity = cosine_similarity(rsp_txt_embedding, txt_embedding)
            if similarity <= 0.1:
                pkg, k_idx = locations[idx]
                knowledge_data[pkg]["knowledge"][k_idx]["info"] += rsp_txt
        if add_as_new:
            d = {
                "attrib": asdict(e),
                "image": np.array(image_patch),
                "info": rsp_txt,
            }
            if "knowledge" not in knowledge_data[package_name]:
                knowledge_data[package_name]["knowledge"] = []
            knowledge_data[package_name]["knowledge"].append(d)
            fusion_memory.add_knowledge_items([d])
            locations.append(
                (package_name, len(knowledge_data[package_name]["knowledge"]) - 1)
            )
