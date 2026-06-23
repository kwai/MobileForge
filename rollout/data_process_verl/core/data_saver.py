#!/usr/bin/env python3
"""
MobileForge数据保存模块

负责将处理后的训练数据保存为不同格式，包括：
1. MobileForge GRPO格式
2. R1V格式
3. 自动创建带时间戳的输出目录
"""

import json
import re
import shutil
import base64
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pandas as pd
import logging
from PIL import Image

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  Evaluation Hints 检测与移除
#  与 MobileForge training/verl/utils/dataset.py 保持一致
# ════════════════════════════════════════════════════════════════

_HINT_BLOCK_PATTERN = re.compile(
    r'={10,}\s*\n'
    r'\s*EVALUATION HINTS FROM PREVIOUS ATTEMPTS\s*\n'
    r'={10,}\s*\n'
    r'.*?'
    r'={10,}\s*\n'
    r'\s*END OF EVALUATION HINTS\s*\n'
    r'={10,}\s*',
    re.DOTALL,
)


def _conversation_has_hints(conversation: List[Dict[str, Any]]) -> bool:
    """检查对话中是否包含 EVALUATION HINTS 块"""
    for msg in conversation:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if _HINT_BLOCK_PATTERN.search(content):
                return True
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    if _HINT_BLOCK_PATTERN.search(item.get("text", "")):
                        return True
    return False


def _remove_hints_from_conversation(
    conversation: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从对话消息中移除 EVALUATION HINTS 块，返回新列表"""
    new_conv: List[Dict[str, Any]] = []
    for msg in conversation:
        if msg.get("role") != "user":
            new_conv.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            cleaned = _HINT_BLOCK_PATTERN.sub("", content)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            new_conv.append({**msg, "content": cleaned})
        elif isinstance(content, list):
            new_items = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    cleaned = _HINT_BLOCK_PATTERN.sub("", item.get("text", ""))
                    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
                    if cleaned:
                        new_items.append({**item, "text": cleaned})
                else:
                    new_items.append(item)
            new_conv.append({**msg, "content": new_items})
        else:
            new_conv.append(msg)
    return new_conv


def _extract_goal_from_guiowl_messages(input_messages: List[Dict[str, Any]]) -> str:
    """
    从 GUIOwl15 的 user message 中提取 goal (task instruction)

    Args:
        input_messages: GUIOwl15 的多轮对话消息列表

    Returns:
        goal 字符串
    """
    for msg in input_messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if "Instruction:" in text:
                            parts = text.split("Instruction:")
                            if len(parts) > 1:
                                goal = parts[1].split("Previous actions:")[0].strip()
                                return goal
    return ""


def _convert_format_guiowl(
    goal: str,
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将 GUIOwl15 的多轮对话格式转换为实际的 LLM call 格式

    原始多轮格式（detailed_model_logs.json 中保存的）：
        [system, user+img(step1), assistant(step1), user+img(step2), assistant(step2), ...]
    
    目标格式（实际的 LLM call）：
        [system, user(text+history+img), assistant]

    Args:
        goal: 任务目标（instruction）
        messages: 多轮对话消息列表（包括 input_messages + 当前 assistant response）

    Returns:
        转换后的单轮格式消息列表
    """
    import copy

    if len(messages) < 2:
        return copy.deepcopy(messages)

    new_messages = [copy.deepcopy(messages[0])]  # Keep system message

    # 提取历史 action（从 assistant responses，跳过最后一个）
    history = []
    assistant_count = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            assistant_count += 1
            # 跳过最后一个 assistant（它是当前的响应）
            if assistant_count <= len([m for m in messages if m.get("role") == "assistant"]) - 1:
                text = msg.get("content", [])
                if isinstance(text, list) and len(text) > 0:
                    text = text[0].get("text", "")
                elif isinstance(text, str):
                    pass
                else:
                    text = ""
                # 提取 Action
                if "Action:" in text:
                    action = text.split("Action:")[-1].split("<tool_call>")[0].strip()
                    history.append(action)

    # 构建 history string
    if history:
        history_string = ""
        for j, h in enumerate(history):
            history_string += f"Step{j+1}: {h}\n"
        history_string = history_string[:-1]  # Remove trailing newline
    else:
        history_string = "No previous action."

    # 构建 user message with history
    user_text = (
        f"Please generate the next move according to the UI screenshot, "
        f"instruction and previous actions.\n\n"
        f"Instruction: {goal}\n\n"
        f"Previous actions:\n{history_string}"
    )

    # 找到最后一个 user message with image
    last_user_with_image = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            has_image = False
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        has_image = True
                        break
            if has_image:
                last_user_with_image = msg

    if last_user_with_image:
        # 构建新的 user message - 找到 image_url item
        image_url_value = None
        for item in last_user_with_image.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_url_dict = item.get("image_url", {})
                if isinstance(image_url_dict, dict):
                    image_url_value = image_url_dict.get("url")
                elif isinstance(image_url_dict, str):
                    image_url_value = image_url_dict
                break

        if image_url_value:
            new_user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url_value}},
                ],
            }
        else:
            # 没有找到图片，使用默认格式
            new_user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        new_messages.append(new_user_msg)
    else:
        # 没有找到带图片的 user message，使用默认格式
        new_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        })

    # 添加当前的 assistant response（最后一个）
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    if last_assistant:
        new_messages.append(copy.deepcopy(last_assistant))

    return new_messages


def _convert_multiturn_to_singleturn(
    input_messages: List[Dict[str, Any]],
    assistant_response: str,
) -> List[Dict[str, Any]]:
    """
    将多轮对话格式转换为单轮格式（用于GUIOwl15数据）

    多轮格式（input_messages）：
        [system, user+img(step1), assistant(step1), user(empty)+img(step2), ...]
    单轮格式（输出）：
        [system, user(当前step), assistant(当前step)]

    Args:
        input_messages: 多轮对话的消息列表
        assistant_response: 当前的 assistant 响应文本

    Returns:
        单轮格式的对话列表
    """
    conversation = []

    for msg in input_messages:
        role = msg.get("role", "")

        # 只保留 system 和最后一个 user 消息
        if role == "system":
            conversation.append(msg)
        elif role == "user":
            # 检查是否是最后一个 user 消息（当前 step 的输入）
            # 找到最后一个包含图片的 user 消息
            content = msg.get("content", [])
            has_image = False
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["image", "image_url"]:
                        has_image = True
                        break
            elif isinstance(content, str) and "<image>" in content:
                has_image = True

            # 保留包含图片的 user 消息（当前 step）
            if has_image:
                conversation.append(msg)

    # 添加当前的 assistant 响应
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_response}],
    })

    return conversation


# 移除PILImageEncoder，现在直接使用标准JSON编码器，因为图像已经转换为base64字符串


class MobileForgeDataSaver:
    """
    MobileForge数据保存器

    负责将处理后的数据保存为训练格式，自动管理输出目录结构
    """

    def __init__(self, base_output_dir: str = "processed_data"):
        """
        初始化数据保存器

        Args:
            base_output_dir: 基础输出目录
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        # 创建带时间戳的子目录
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{self.timestamp}"
        self.session_dir.mkdir(exist_ok=True)

        logger.info(f"初始化数据保存器")
        logger.info(f"基础输出目录: {self.base_output_dir}")
        logger.info(f"会话目录: {self.session_dir}")

    def save_training_data(
        self,
        processed_data: Dict[str, Any],
        format_type: str = "grpo",
        positive_only: bool = False,
        max_steps: int = 0,
        remove_evaluation_hints: bool = False,
    ) -> Dict[str, str]:
        """
        保存训练数据到文件（所有文件都保存在带时间戳的会话目录中）

        Args:
            processed_data: 处理后的训练数据
            format_type: 格式类型（"grpo"用于RL训练，"r1v"用于SFT）
            positive_only: 是否仅保留 impact=positive 的步骤
            max_steps: 最终数据集最大步骤数（<= 0 不限制），超出时按 app+action_type 均衡采样
            remove_evaluation_hints: 是否从 user prompt 中删除 EVALUATION HINTS 块

        Returns:
            保存的文件路径字典
        """
        saved_files = {}

        if format_type == "grpo":
            saved_files.update(
                self._save_grpo_format(
                    processed_data,
                    positive_only=positive_only,
                    max_steps=max_steps,
                    remove_evaluation_hints=remove_evaluation_hints,
                )
            )
        else:
            saved_files.update(self._save_r1v_format(processed_data))

        # 生成使用说明文件
        self._create_usage_readme(saved_files, format_type)

        return saved_files

    def _compute_step_level_stats(
        self, all_grpo_samples: List[Dict], processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算详细的步骤级统计信息"""
        total_samples = len(all_grpo_samples)
        if total_samples == 0:
            return {"error": "No samples to analyze"}

        # 收集各字段的统计
        overall_success_true = 0
        overall_success_false = 0
        impact_positive = 0
        impact_negative = 0
        impact_neutral = 0
        impact_unknown = 0
        bad_step_true = 0
        bad_step_false = 0

        # 按任务统计（用于过滤简单/困难任务）
        task_samples = {}  # task_id -> list of samples

        for sample in all_grpo_samples:
            metadata = sample.get("metadata", {})
            task_id = metadata.get("task_id", "unknown")

            # 收集到任务分组
            if task_id not in task_samples:
                task_samples[task_id] = []
            task_samples[task_id].append(sample)

            # 统计各字段
            if metadata.get("overall_success", False):
                overall_success_true += 1
            else:
                overall_success_false += 1

            impact = metadata.get("impact", "unknown")
            if impact == "positive":
                impact_positive += 1
            elif impact == "negative":
                impact_negative += 1
            elif impact == "neutral":
                impact_neutral += 1
            else:
                impact_unknown += 1

            if sample.get("bad_step", False):
                bad_step_true += 1
            else:
                bad_step_false += 1

        # 获取简单任务、困难任务和异常任务的task_id列表
        processing_stats = processed_data.get("statistics", {})
        easy_task_ids = set(processing_stats.get("easy_task_ids", []))
        hard_task_ids = set(processing_stats.get("hard_task_ids", []))
        error_task_ids = set(processing_stats.get("error_task_ids", []))

        # 过滤后的样本统计（排除简单和困难任务）
        filtered_samples = [
            s
            for s in all_grpo_samples
            if s.get("metadata", {}).get("task_id", "") not in easy_task_ids
            and s.get("metadata", {}).get("task_id", "") not in hard_task_ids
        ]

        filtered_stats = (
            self._compute_filtered_stats(filtered_samples) if filtered_samples else {}
        )

        # 排除异常任务后的样本统计
        clean_samples = [
            s
            for s in all_grpo_samples
            if s.get("metadata", {}).get("task_id", "") not in error_task_ids
        ]

        clean_stats = (
            self._compute_filtered_stats(clean_samples) if clean_samples else {}
        )

        # 计算排除异常任务后的任务级统计
        clean_task_ids = set(task_samples.keys()) - error_task_ids
        clean_easy_task_ids = easy_task_ids - error_task_ids
        clean_hard_task_ids = hard_task_ids - error_task_ids

        # 排除异常任务 + 只保留中等难度任务（排除简单和困难任务）
        excluded_task_ids = error_task_ids | easy_task_ids | hard_task_ids
        medium_clean_samples = [
            s
            for s in all_grpo_samples
            if s.get("metadata", {}).get("task_id", "") not in excluded_task_ids
        ]

        medium_clean_stats = (
            self._compute_filtered_stats(medium_clean_samples)
            if medium_clean_samples
            else {}
        )

        # 计算中等难度+排除异常的任务数
        medium_clean_task_ids = set(task_samples.keys()) - excluded_task_ids

        def pct(count, total):
            return round(count / total * 100, 2) if total > 0 else 0.0

        return {
            "total_samples": total_samples,
            "total_tasks": len(task_samples),
            "all_samples": {
                "overall_success": {
                    "true": overall_success_true,
                    "false": overall_success_false,
                    "true_pct": pct(overall_success_true, total_samples),
                    "false_pct": pct(overall_success_false, total_samples),
                },
                "impact": {
                    "positive": impact_positive,
                    "negative": impact_negative,
                    "neutral": impact_neutral,
                    "unknown": impact_unknown,
                    "positive_pct": pct(impact_positive, total_samples),
                    "negative_pct": pct(impact_negative, total_samples),
                    "neutral_pct": pct(impact_neutral, total_samples),
                    "unknown_pct": pct(impact_unknown, total_samples),
                },
                "bad_step": {
                    "true": bad_step_true,
                    "false": bad_step_false,
                    "true_pct": pct(bad_step_true, total_samples),
                    "false_pct": pct(bad_step_false, total_samples),
                },
            },
            "filtered_samples": filtered_stats,
            "filter_info": {
                "easy_tasks_excluded": len(easy_task_ids),
                "hard_tasks_excluded": len(hard_task_ids),
                "total_excluded_samples": total_samples - len(filtered_samples),
                "remaining_samples": len(filtered_samples),
            },
            "clean_samples": clean_stats,
            "clean_info": {
                "error_tasks_excluded": len(error_task_ids),
                "clean_tasks_count": len(clean_task_ids),
                "clean_easy_tasks": len(clean_easy_task_ids),
                "clean_hard_tasks": len(clean_hard_task_ids),
                "total_excluded_samples": total_samples - len(clean_samples),
                "remaining_samples": len(clean_samples),
            },
            "medium_clean_samples": medium_clean_stats,
            "medium_clean_info": {
                "error_tasks_excluded": len(error_task_ids),
                "easy_tasks_excluded": len(easy_task_ids),
                "hard_tasks_excluded": len(hard_task_ids),
                "total_tasks_excluded": len(excluded_task_ids),
                "medium_clean_tasks_count": len(medium_clean_task_ids),
                "total_excluded_samples": total_samples - len(medium_clean_samples),
                "remaining_samples": len(medium_clean_samples),
            },
        }

    def _compute_filtered_stats(self, samples: List[Dict]) -> Dict[str, Any]:
        """计算过滤后样本的统计信息"""
        total = len(samples)
        if total == 0:
            return {"error": "No samples after filtering"}

        stats = {
            "total_samples": total,
            "overall_success": {"true": 0, "false": 0},
            "impact": {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0},
            "bad_step": {"true": 0, "false": 0},
        }

        for sample in samples:
            metadata = sample.get("metadata", {})

            if metadata.get("overall_success", False):
                stats["overall_success"]["true"] += 1
            else:
                stats["overall_success"]["false"] += 1

            impact = metadata.get("impact", "unknown")
            if impact in stats["impact"]:
                stats["impact"][impact] += 1
            else:
                stats["impact"]["unknown"] += 1

            if sample.get("bad_step", False):
                stats["bad_step"]["true"] += 1
            else:
                stats["bad_step"]["false"] += 1

        def pct(count, total):
            return round(count / total * 100, 2) if total > 0 else 0.0

        # 添加百分比 - 使用 list() 避免在迭代时修改字典
        for key in ["overall_success", "bad_step"]:
            for subkey in list(stats[key].keys()):
                if not subkey.endswith("_pct"):
                    stats[key][f"{subkey}_pct"] = pct(stats[key][subkey], total)

        for key in ["impact"]:
            for subkey in list(stats[key].keys()):
                if not subkey.endswith("_pct"):
                    stats[key][f"{subkey}_pct"] = pct(stats[key][subkey], total)

        return stats

    # ════════════════════════════════════════════════════════════════
    #  Balanced sampling: action_type-first, app-second
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _balanced_by_app(
        samples: List[Dict[str, Any]], quota: int, rng: random.Random
    ) -> List[Dict[str, Any]]:
        """
        Select *quota* samples from *samples*, balancing across apps.

        Strategy: sort apps by count (rarest first), give each app
        an equal share of the remaining quota, take all if the app
        has fewer than its share.
        """
        app_groups: Dict[str, List[Dict]] = defaultdict(list)
        for s in samples:
            app = (s.get("metadata") or {}).get("app") or "Unknown"
            app_groups[app].append(s)

        for app in app_groups:
            rng.shuffle(app_groups[app])

        selected: List[Dict] = []
        remaining = quota
        sorted_apps = sorted(app_groups.keys(), key=lambda a: len(app_groups[a]))
        n_remaining = len(sorted_apps)

        for app in sorted_apps:
            if n_remaining <= 0 or remaining <= 0:
                break
            share = max(remaining // n_remaining, 1) if remaining > 0 else 0
            take = min(len(app_groups[app]), share)
            selected.extend(app_groups[app][:take])
            remaining -= take
            n_remaining -= 1

        return selected

    def _balanced_step_selection(
        self,
        samples: List[Dict[str, Any]],
        max_steps: int,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Two-level balanced sampling: **action_type-first, app-second**.

        Strategy:
        1. Level 1 — global action_type balance: allocate
           ``max_steps // num_action_types`` per action_type.
           Rare types (fewer samples than quota) take all; their
           surplus is redistributed to remaining types.
        2. Level 2 — within each action_type's allocation, call
           ``_balanced_by_app`` to spread samples across apps.

        This ensures action_type counts are as equal as possible
        globally, while maintaining app diversity within each type.

        Returns:
            (selected_samples, stats_dict)
        """
        before_count = len(samples)
        if before_count <= max_steps:
            return samples, {
                "max_steps": max_steps,
                "before_count": before_count,
                "after_count": before_count,
                "num_apps": 0,
                "num_action_types": 0,
                "app_distribution": {},
                "action_type_distribution": {},
                "skipped": True,
            }

        rng = random.Random(seed)

        # ── Collect global groupings ──
        type_groups: Dict[str, List[Dict]] = defaultdict(list)
        app_set: set = set()
        for s in samples:
            at = (s.get("metadata") or {}).get("action_type") or "unknown"
            type_groups[at].append(s)
            app_set.add((s.get("metadata") or {}).get("app") or "Unknown")

        num_types = len(type_groups)
        num_apps = len(app_set)

        # ── Level 1: global action_type quota allocation ──
        selected: List[Dict] = []
        remaining_budget = max_steps
        sorted_types = sorted(type_groups.keys(), key=lambda t: len(type_groups[t]))
        remaining_types = num_types

        for at in sorted_types:
            if remaining_types <= 0 or remaining_budget <= 0:
                break
            type_quota = remaining_budget // remaining_types
            type_quota = max(type_quota, 1) if remaining_budget > 0 else 0
            at_samples = type_groups[at]

            if len(at_samples) <= type_quota:
                # Rare type — take all
                chosen = at_samples
            else:
                # Level 2: within this action_type, balance by app
                chosen = self._balanced_by_app(at_samples, type_quota, rng)

            selected.extend(chosen)
            remaining_budget -= len(chosen)
            remaining_types -= 1

        # Safety truncation
        if len(selected) > max_steps:
            rng.shuffle(selected)
            selected = selected[:max_steps]

        # ── Re-sort (keep task_id → attempt → step order) ──
        import re as _re

        def _sort_key(sample):
            md = sample.get("metadata", {})
            tid = md.get("task_id", "")
            aid = md.get("attempt_id", 0)
            if isinstance(aid, str):
                m = _re.search(r"\d+", aid)
                aid = int(m.group()) if m else 0
            return (tid, aid, md.get("step_number", 0))

        selected.sort(key=_sort_key)

        # ── Compute distribution statistics ──
        # action_type
        at_counter_before: Dict[str, int] = defaultdict(int)
        for s in samples:
            at_counter_before[
                (s.get("metadata") or {}).get("action_type") or "unknown"
            ] += 1
        at_counter_after: Dict[str, int] = defaultdict(int)
        for s in selected:
            at_counter_after[
                (s.get("metadata") or {}).get("action_type") or "unknown"
            ] += 1

        at_dist: Dict[str, Dict] = {}
        for at in sorted(type_groups.keys()):
            at_dist[at] = {
                "available": at_counter_before.get(at, 0),
                "selected": at_counter_after.get(at, 0),
            }

        # app
        app_counter_before: Dict[str, int] = defaultdict(int)
        for s in samples:
            app_counter_before[
                (s.get("metadata") or {}).get("app") or "Unknown"
            ] += 1
        app_counter_after: Dict[str, int] = defaultdict(int)
        for s in selected:
            app_counter_after[
                (s.get("metadata") or {}).get("app") or "Unknown"
            ] += 1

        app_dist: Dict[str, Dict] = {}
        for app in sorted(app_set):
            app_dist[app] = {
                "available": app_counter_before.get(app, 0),
                "selected": app_counter_after.get(app, 0),
            }

        stats = {
            "max_steps": max_steps,
            "before_count": before_count,
            "after_count": len(selected),
            "num_apps": num_apps,
            "num_action_types": num_types,
            "app_distribution": app_dist,
            "action_type_distribution": at_dist,
        }

        return selected, stats

    def _save_grpo_format(
        self,
        processed_data: Dict[str, Any],
        positive_only: bool = False,
        max_steps: int = 0,
        remove_evaluation_hints: bool = False,
    ) -> Dict[str, str]:
        """保存GRPO格式的训练数据（步骤级标注版本，支持重试去重）"""
        logger.info(
            f"保存GRPO格式训练数据（步骤级标注版本，支持重试去重, "
            f"positive_only={positive_only}, max_steps={max_steps}, "
            f"remove_evaluation_hints={remove_evaluation_hints}）..."
        )

        all_grpo_samples = []
        retry_deduplication_stats = {
            "total_steps_before_dedup": 0,
            "total_steps_after_dedup": 0,
            "deduplicated_steps": 0,
        }

        # Hint 统计
        hint_stats = {
            "total_steps_scanned": 0,       # 扫描过的所有步骤（含被筛掉的）
            "steps_with_hint_total": 0,      # 原始数据中带 hint 的步骤数
            "steps_with_hint_kept": 0,       # 最终保留的样本中原本带 hint 的步骤数
            "hints_removed": 0,              # 实际被移除了 hint 块的步骤数
        }

        # 获取 error_task_ids 和 error_attempt_ids 用于标记
        error_task_ids = set(
            processed_data.get("statistics", {}).get("error_task_ids", [])
        )
        error_attempt_ids = set(
            processed_data.get("statistics", {}).get("error_attempt_ids", [])
        )

        # 处理所有样本（不再区分原始的正负样本）
        for sample in processed_data["all_samples"]:
            if sample.get("data_source") == "detailed_model_logs_step_labeled":
                task_id = sample["task_id"]
                attempt_id = sample.get("attempt", "unknown")

                step_details = sample.get("metadata", {}).get("step_details", [])
                retry_deduplication_stats["total_steps_after_dedup"] += len(
                    step_details
                )

                for step_detail in step_details:
                    step_number = step_detail.get("step_number", 1)
                    step_label = step_detail.get("step_label", "unknown")
                    impact = step_detail.get("impact", "unknown")

                    # 统计去重信息
                    if step_detail.get("is_deduplicated", False):
                        retry_deduplication_stats["deduplicated_steps"] += 1
                        total_attempts = step_detail.get("total_attempts", 1)
                        retry_deduplication_stats["total_steps_before_dedup"] += (
                            total_attempts
                        )
                    else:
                        retry_deduplication_stats["total_steps_before_dedup"] += 1

                    input_messages = step_detail.get("input_messages", [])
                    raw_response = step_detail.get("raw_response", "")

                    if not input_messages or not raw_response:
                        continue

                    # 检查是否是 GUIOwl15 数据
                    agent_name = sample.get("agent_name", "")
                    is_guiowl15 = agent_name == "GUIOwl15"

                    if is_guiowl15:
                        # GUIOwl15: 需要调用 _convert_format_guiowl 转换为实际的 LLM call 格式
                        goal = _extract_goal_from_guiowl_messages(input_messages)
                        # _convert_format_guiowl 需要完整的 messages（包括当前的 assistant）
                        full_messages = input_messages + [{
                            "role": "assistant",
                            "content": [{"type": "text", "text": raw_response}]
                        }]
                        converted_messages = _convert_format_guiowl(goal, full_messages)
                        
                        # 从转换后的消息中分离 input_messages 和 assistant
                        if converted_messages:
                            # 最后一个是 assistant
                            assistant_msg = converted_messages[-1]
                            input_only = converted_messages[:-1]
                        else:
                            input_only = []
                            assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": raw_response}]}

                        # 标准化处理 - GUIOwl15 需要保留完整的 image_url 数据
                        normalized_input_messages = []
                        for msg in input_only:
                            normalized_msg = self._normalize_message_content(msg, keep_image_url=True)
                            normalized_input_messages.append(normalized_msg)
                        normalized_assistant = self._normalize_message_content(assistant_msg)

                        conversation = normalized_input_messages + [normalized_assistant]
                    else:
                        # 其他模型：直接标准化处理
                        # 保留 image_url/base64 数据，避免只留下 <image> 占位符而丢失真实图像。
                        normalized_input_messages = []
                        for msg in input_messages:
                            normalized_msg = self._normalize_message_content(msg, keep_image_url=True)
                            normalized_input_messages.append(normalized_msg)

                        # 创建标准化的assistant响应
                        assistant_response = {
                            "role": "assistant",
                            "content": [{"type": "text", "text": raw_response}],
                        }
                        normalized_assistant = self._normalize_message_content(
                            assistant_response
                        )

                        conversation = normalized_input_messages + [normalized_assistant]

                    # ── Hint 检测与可选移除 ──
                    has_hint = _conversation_has_hints(conversation)
                    hint_stats["total_steps_scanned"] += 1
                    if has_hint:
                        hint_stats["steps_with_hint_total"] += 1
                    if remove_evaluation_hints and has_hint:
                        conversation = _remove_hints_from_conversation(conversation)
                        hint_stats["hints_removed"] += 1

                    # 基于impact判断是否为bad_step（impact != 'positive' 则为bad_step）
                    bad_step = impact != "positive"

                    # DART-GUI Distribution Alignment: extract rollout_log_probs
                    # We store both the token-level logprobs and the sum for flexibility
                    rollout_logprobs = step_detail.get("logprobs", None)
                    rollout_token_logprobs_sum = step_detail.get("token_logprobs_sum", None)
                    
                    # 提取任务可行性评估信息（来自 final_decision.json）
                    sample_metadata = sample.get("metadata", {})

                    grpo_sample = {
                        "conversations": conversation,
                        "bad_step": bad_step,
                        # DART-GUI: rollout_log_probs for distribution alignment
                        # This is the sum of log probabilities during data collection (rollout)
                        "rollout_log_probs_sum": rollout_token_logprobs_sum,
                        # Store detailed token-level logprobs for advanced use cases
                        "rollout_logprobs": rollout_logprobs,
                        "metadata": {
                            "task_id": task_id,
                            "attempt_id": attempt_id,
                            "step_number": step_number,
                            "step_label": step_label,
                            "reasonableness": step_detail.get(
                                "reasonableness", "unknown"
                            ),
                            "reasonableness_explanation": step_detail.get(
                                "reasonableness_explanation", ""
                            ),
                            "impact": impact,
                            "step_success": step_detail.get("success", False),
                            "overall_success": sample["success"],
                            "final_result": sample.get("evaluation_result", -1),
                            "is_error_task": task_id
                            in error_task_ids,  # 该任务是否包含 error_trajectories
                            "is_error_attempt": f"{task_id}|{attempt_id}"
                            in error_attempt_ids,  # 该 attempt 是否为 error
                            "action_type": step_detail.get("action_type"),
                            "action": step_detail.get("action"),  # 完整 action (来自 log.json)
                            "tokens": step_detail.get("tokens", {}),
                            "api_duration": step_detail.get("api_duration", 0),
                            "retry_count": step_detail.get("retry_count", 1),
                            "image_mapping_fixed": True,
                            "step_level_annotation": True,  # 标记为步骤级标注
                            # 新增：重试去重信息
                            "is_deduplicated": step_detail.get(
                                "is_deduplicated", False
                            ),
                            "total_attempts": step_detail.get("total_attempts", 1),
                            "retry_deduplication_applied": True,
                            "attempt_stats": sample["attempt_stats"],
                            # 任务级 SR
                            "avg_sr": sample.get("avg_sr", 0.0),
                            # App 信息
                            "app": sample_metadata.get("app", "Unknown"),
                            # 死循环检测
                            "max_consecutive_same_actions": sample_metadata.get(
                                "max_consecutive_same_actions", 0
                            ),
                            "loop_detail": sample_metadata.get("loop_detail", ""),
                            # Eval Hint 相关
                            "has_eval_hint": sample_metadata.get("has_eval_hint", False),
                            "has_hints_input": sample_metadata.get("has_hints_input", False),
                            # 对话中是否原本包含 EVALUATION HINTS 块
                            "has_hint_in_conversation": has_hint,
                            # DART-GUI: 记录logprobs相关统计
                            "num_completion_tokens_with_logprobs": step_detail.get(
                                "num_completion_tokens_with_logprobs", None
                            ),
                            "has_rollout_logprobs": rollout_logprobs is not None,
                            # 任务可行性评估（来自 final_decision.json）
                            "task_feasible": sample_metadata.get("task_feasible", None),
                            "task_feasible_reason": sample_metadata.get("task_feasible_reason", ""),
                            "task_barriers": sample_metadata.get("task_barriers", []),
                        },
                    }

                    # 只收集有效标注的样本（正样本或负样本）
                    if step_label in ["positive", "negative"]:
                        # positive_only 模式下仅保留 impact=positive 的步骤
                        if positive_only and impact != "positive":
                            continue
                        all_grpo_samples.append(grpo_sample)
                        if has_hint:
                            hint_stats["steps_with_hint_kept"] += 1
                    # 忽略step_label == 'unknown'的步骤

        # 按照任务ID、attempt序号、step序号进行升序排列
        def sort_key(sample):
            metadata = sample.get("metadata", {})
            task_id = metadata.get("task_id", "")
            # 处理 attempt_id，可能是 "attempt_1" 或者数字
            attempt_id = metadata.get("attempt_id", 0)
            if isinstance(attempt_id, str):
                # 从 "attempt_1" 中提取数字
                import re

                match = re.search(r"\d+", attempt_id)
                attempt_id = int(match.group()) if match else 0
            step_number = metadata.get("step_number", 0)
            return (task_id, attempt_id, step_number)

        all_grpo_samples.sort(key=sort_key)

        # ── 可选：按 app + action_type 均衡采样到 max_steps ──
        balanced_selection_stats = None
        if max_steps > 0 and len(all_grpo_samples) > max_steps:
            all_grpo_samples, balanced_selection_stats = self._balanced_step_selection(
                all_grpo_samples, max_steps
            )
            logger.info(
                f"均衡采样: {balanced_selection_stats['before_count']} → "
                f"{balanced_selection_stats['after_count']} steps"
            )

        # 统计正负样本数量（基于 bad_step，即 impact != 'positive'）
        positive_count = sum(1 for s in all_grpo_samples if not s["bad_step"])
        negative_count = sum(1 for s in all_grpo_samples if s["bad_step"])

        # 计算详细的步骤级统计信息
        step_level_stats = self._compute_step_level_stats(
            all_grpo_samples, processed_data
        )

        # 保存文件
        saved_files = {}

        # 合并保存所有样本到一个文件
        combined_file = self.session_dir / f"mobileforge_grpo_{self.timestamp}.json"
        with open(combined_file, "w", encoding="utf-8") as f:
            json.dump(all_grpo_samples, f, indent=2, ensure_ascii=False)
        saved_files["grpo_data"] = str(combined_file)
        saved_files["step_level_stats"] = step_level_stats  # 添加统计信息供外部使用
        saved_files["hint_stats"] = hint_stats
        if balanced_selection_stats is not None:
            saved_files["balanced_selection_stats"] = balanced_selection_stats
        logger.info(
            f"已保存GRPO数据: {len(all_grpo_samples)} 个样本 (正样本: {positive_count}, 负样本: {negative_count})"
        )

        # 保存统计信息
        grpo_stats = {
            "session_timestamp": self.timestamp,
            "annotation_method": "step_level_impact_based_with_retry_deduplication",
            "bad_step_criteria": "impact != positive (包括 negative, neutral, unknown)",
            "total_samples": len(all_grpo_samples),
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "original_positive_trajectories": len(processed_data["positive_samples"]),
            "original_negative_trajectories": len(processed_data["negative_samples"]),
            "processing_statistics": processed_data.get("statistics", {}),
            "step_level_breakdown": {
                "positive_labeled_steps": positive_count,
                "negative_labeled_steps": negative_count,
                "total_valid_steps": len(all_grpo_samples),
            },
            "step_level_stats": step_level_stats,
            "retry_deduplication_stats": retry_deduplication_stats,
            "annotation_details": {
                "method": "基于evaluation_summary.json中的step_reasonableness_analysis",
                "bad_step_criteria": "impact != positive",
                "positive_criteria": "impact == positive",
                "negative_criteria": "impact != positive (negative/neutral/unknown)",
                "unknown_steps_handling": "步骤级标注为unknown的步骤被忽略",
                "retry_handling": "重试步骤经过去重处理，每个step_number只保留一个代表性执行",
                "sorting": "按任务ID、attempt序号、step序号升序排列",
            },
            "fix_notes": {
                "image_mapping_strategy": "UI-TARS时间正序映射，最旧图像在前",
                "png_naming_convention": "step N保存(N-1).png",
                "max_images_retained": 5,
                "mapping_fixed": True,
                "step_level_annotation": True,
                "retry_deduplication_applied": True,
                "deduplication_strategy": "优先选择成功的执行，其次选择最新的重试",
                "output_format": "正负样本合并为单一JSON文件",
            },
        }

        stats_file = self.session_dir / f"mobileforge_grpo_stats_{self.timestamp}.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(grpo_stats, f, indent=2)
        saved_files["grpo_stats"] = str(stats_file)
        logger.info(f"已保存GRPO统计信息")

        return saved_files

    def _save_r1v_format(self, processed_data: Dict[str, Any]) -> Dict[str, str]:
        """保存R1V格式的训练数据"""
        logger.info("保存R1V格式训练数据...")

        saved_files = {}

        # 正样本
        positive_file = (
            self.session_dir / f"mobileforge_positive_samples_{self.timestamp}.json"
        )
        with open(positive_file, "w", encoding="utf-8") as f:
            json.dump(
                processed_data["positive_samples"], f, indent=2, ensure_ascii=False
            )
        saved_files["positive_r1v"] = str(positive_file)
        logger.info(f"已保存R1V正样本: {len(processed_data['positive_samples'])} 个")

        # 负样本
        negative_file = (
            self.session_dir / f"mobileforge_negative_samples_{self.timestamp}.json"
        )
        with open(negative_file, "w", encoding="utf-8") as f:
            json.dump(
                processed_data["negative_samples"], f, indent=2, ensure_ascii=False
            )
        saved_files["negative_r1v"] = str(negative_file)
        logger.info(f"已保存R1V负样本: {len(processed_data['negative_samples'])} 个")

        # 统计信息
        r1v_stats = {
            "session_timestamp": self.timestamp,
            "processing_statistics": processed_data.get("statistics", {}),
            "positive_samples_count": len(processed_data["positive_samples"]),
            "negative_samples_count": len(processed_data["negative_samples"]),
        }

        stats_file = (
            self.session_dir / f"mobileforge_processing_stats_{self.timestamp}.json"
        )
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(r1v_stats, f, indent=2)
        saved_files["r1v_stats"] = str(stats_file)
        logger.info(f"已保存R1V统计信息")

        return saved_files

    def _create_usage_readme(
        self, saved_files: Dict[str, str], format_type: str
    ) -> None:
        """在会话目录中创建使用说明文件"""
        try:
            if format_type == "grpo":
                # 获取训练文件的相对路径
                grpo_file = saved_files.get("grpo_data", "")

                readme_content = f"""# MobileForge训练数据 - 会话 {self.timestamp}

## 📁 文件说明

本会话生成的训练数据文件：

- `{Path(grpo_file).name}`: MobileForge GRPO训练数据（包含正负样本）
- `mobileforge_grpo_stats_{self.timestamp}.json`: 统计信息
- `session_summary.json`: 会话汇总

## 📊 数据格式

JSON文件中每个样本包含：
- `conversations`: 对话内容
- `bad_step`: 是否为负样本（true=负样本，false=正样本）
- `metadata`: 元数据，包含任务ID、attempt序号、step序号等

数据按以下顺序升序排列：
1. 任务ID (task_id)
2. Attempt序号 (attempt_id)
3. Step序号 (step_number)

## 🚀 MobileForge训练使用

使用本会话生成的数据进行GRPO训练：

```bash
cd /path/to/MobileForge
torchrun --nproc_per_node=8 src/open_r1/grpo_gui.py \\
    --dataset_name {Path(grpo_file).absolute()} \\
    --model_name_or_path your_model_path \\
    --output_dir ./grpo_output \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 16 \\
    --save_steps 100 \\
    --logging_steps 10
```

## 🔧 图像映射修复

本版本已修复UI-TARS多轮对话中的图像错位问题：

- ✅ **正确的时间序列映射**: UI-TARS保存screenshot为`{{step-1}}.png`，即step N保存(N-1).png
- ✅ **多轮对话处理**: UI-TARS只保留最近5张图像，按时间正序排列
- ✅ **占位符替换**: 确保对话中的图像按正确的时间顺序排列（最旧的图像在前）
- ✅ **数据一致性**: 训练数据中的图像与实际执行时的截图完全一致

## 🏷️ 步骤级标注机制

本版本采用**步骤级合理性标注**，实现了更精确的正负样本划分：

- ✅ **步骤级精度**: 基于`evaluation_summary.json`中的`step_reasonableness_analysis`
- ✅ **合理性判断**: 
  - **正样本** (bad_step=false): `reasonable_steps`列表中的步骤
  - **负样本** (bad_step=true): `unreasonable_steps`列表中的步骤
- ✅ **详细解释**: 每个步骤都包含合理性解释和影响分析
- ✅ **质量保证**: 忽略标注为unknown的步骤，确保训练数据质量

## 📊 数据质量保证

- 所有对话步骤都经过图像映射修复
- **步骤级标注**确保正负样本的精确性
- 保留完整的合理性分析元数据用于追溯和分析
- 支持细粒度的模型训练和评估
- 数据按任务ID → Attempt序号 → Step序号升序排列
"""

            else:  # r1v format
                positive_file = saved_files.get("positive_r1v", "")
                negative_file = saved_files.get("negative_r1v", "")

                readme_content = f"""# MobileForge训练数据 - 会话 {self.timestamp}

## 📁 文件说明

本会话生成的R1V格式训练数据：

- `{Path(positive_file).name}`: R1V格式正样本
- `{Path(negative_file).name}`: R1V格式负样本  
- `mobileforge_processing_stats_{self.timestamp}.json`: 统计信息
- `session_summary.json`: 会话汇总

## 🚀 使用说明

R1V格式数据可用于SFT训练或其他自定义训练流程。

## 🔧 图像映射修复

本版本已修复UI-TARS多轮对话中的图像错位问题，确保数据质量。
"""

            readme_file = self.session_dir / "README.md"
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(readme_content)

            logger.info(f"已创建使用说明文件: {readme_file}")

        except Exception as e:
            logger.error(f"创建使用说明时出错: {e}")

    def save_error_analysis(self, processed_data: Dict[str, Any]) -> str:
        """
        保存错误分析JSON文件，包含所有error_trajectories和error_task信息

        Args:
            processed_data: 处理后的数据字典

        Returns:
            保存的文件路径
        """
        try:
            stats = processed_data.get("statistics", {})
            all_samples = processed_data.get("all_samples", [])

            # 收集错误任务的详细信息
            error_tasks = {}
            error_trajectories = []

            # 从统计信息中获取error_task_ids和error_attempt_ids
            error_task_ids = stats.get("error_task_ids", [])
            error_attempt_ids = stats.get("error_attempt_ids", [])

            # 遍历所有样本，提取错误相关信息
            for sample in all_samples:
                task_id = sample.get("task_id")
                attempt = sample.get("attempt")
                evaluation_result = sample.get("evaluation_result")

                # 只处理error的轨迹（evaluation_result not in [0, 1]）
                if evaluation_result not in [0, 1]:
                    # 记录错误轨迹
                    trajectory_info = {
                        "task_id": task_id,
                        "attempt": attempt,
                        "evaluation_result": evaluation_result,
                        "success": sample.get("success"),
                        "total_steps": len(sample.get("steps", [])),
                        "agent_name": sample.get("agent_name"),
                        "task_description": sample.get("task_description", ""),
                    }
                    error_trajectories.append(trajectory_info)

                    # 按任务ID组织错误信息
                    if task_id not in error_tasks:
                        error_tasks[task_id] = {
                            "task_id": task_id,
                            "task_description": sample.get("task_description", ""),
                            "error_attempts": [],
                            "total_error_attempts": 0,
                        }

                    error_tasks[task_id]["error_attempts"].append(
                        {
                            "attempt": attempt,
                            "agent_name": sample.get("agent_name"),
                            "evaluation_result": evaluation_result,
                            "success": sample.get("success"),
                            "total_steps": len(sample.get("steps", [])),
                        }
                    )
                    error_tasks[task_id]["total_error_attempts"] += 1

            # 构建完整的错误分析报告
            error_analysis = {
                "summary": {
                    "timestamp": self.timestamp,
                    "total_tasks": stats.get("total_tasks", 0),
                    "processed_tasks": stats.get("processed_tasks", 0),
                    "error_task_count": len(error_task_ids),
                    "error_trajectory_count": len(error_trajectories),
                    "error_task_percentage": (
                        len(error_task_ids) / stats.get("processed_tasks", 1) * 100
                        if stats.get("processed_tasks", 0) > 0
                        else 0
                    ),
                },
                "error_task_ids": error_task_ids,
                "error_attempt_ids": error_attempt_ids,
                "error_tasks": list(error_tasks.values()),
                "error_trajectories": error_trajectories,
                "statistics": {
                    "total_error_trajectories": stats.get("error_trajectories", 0),
                    "successful_trajectories": stats.get("successful_trajectories", 0),
                    "failed_trajectories": stats.get("failed_trajectories", 0),
                },
            }

            # 保存为JSON文件
            error_file = self.session_dir / f"error_analysis_{self.timestamp}.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ 错误分析已保存: {error_file}")
            logger.info(f"  - 错误任务数: {len(error_task_ids)}")
            logger.info(f"  - 错误轨迹数: {len(error_trajectories)}")

            return str(error_file)

        except Exception as e:
            logger.error(f"保存错误分析时出错: {e}")
            import traceback

            traceback.print_exc()
            return ""

    def save_session_summary(
        self, processed_data: Dict[str, Any], step_level_stats: Dict[str, Any] = None
    ) -> str:
        """保存会话汇总信息"""
        stats = processed_data["statistics"]

        summary = {
            "session_info": {
                "timestamp": self.timestamp,
                "session_dir": str(self.session_dir),
                "base_output_dir": str(self.base_output_dir),
            },
            "processing_summary": {
                "total_tasks": stats["total_tasks"],
                "processed_tasks": stats["processed_tasks"],
                "successful_trajectories": stats["successful_trajectories"],
                "failed_trajectories": stats["failed_trajectories"],
                "error_trajectories": stats["error_trajectories"],
                "image_mapping_fixes": stats["image_mapping_fixes"],
                "placeholder_replacements": stats["placeholder_replacements"],
            },
            "task_difficulty_distribution": {
                "easy_tasks": stats.get("easy_tasks", 0),
                "pass1": stats.get("pass1", 0),
                "pass2": stats.get("pass2", 0),
                "pass3": stats.get("pass3", 0),
                "hard_tasks": stats.get("hard_tasks", 0),
            },
            "trajectory_level_samples": {
                "positive_samples": len(processed_data["positive_samples"]),
                "negative_samples": len(processed_data["negative_samples"]),
                "total_samples": len(processed_data["all_samples"]),
            },
            "error_tasks_info": {
                "error_task_count": len(stats.get("error_task_ids", [])),
            },
        }

        # 添加步骤级统计
        if step_level_stats:
            all_stats = step_level_stats.get("all_samples", {})
            summary["step_level_stats"] = {
                "total_step_samples": step_level_stats.get("total_samples", 0),
                "bad_step": all_stats.get("bad_step", {}),
                "impact": all_stats.get("impact", {}),
                "overall_success": all_stats.get("overall_success", {}),
            }

            # 添加过滤后统计（排除简单和困难任务）
            filter_info = step_level_stats.get("filter_info", {})
            filtered_stats = step_level_stats.get("filtered_samples", {})
            summary["filtered_stats"] = {
                "filter_info": filter_info,
                "stats": filtered_stats,
            }

            # 添加排除异常任务后的统计
            clean_info = step_level_stats.get("clean_info", {})
            clean_stats = step_level_stats.get("clean_samples", {})
            summary["clean_stats"] = {
                "clean_info": clean_info,
                "stats": clean_stats,
            }

            # 添加排除异常任务+只保留中等难度任务的统计
            medium_clean_info = step_level_stats.get("medium_clean_info", {})
            medium_clean_stats = step_level_stats.get("medium_clean_samples", {})
            summary["medium_clean_stats"] = {
                "medium_clean_info": medium_clean_info,
                "stats": medium_clean_stats,
            }

        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"已保存会话汇总: {summary_file}")
        return str(summary_file)

    def _normalize_message_content(
        self, message: Dict[str, Any], keep_image_url: bool = False
    ) -> Dict[str, Any]:
        """标准化消息内容格式，确保content字段始终为list格式

        Args:
            message: 原始消息
            keep_image_url: 如果为 True，保留 image_url 不转换为 <image> 占位符
                           用于 GUIOwl15 等需要保留完整图像数据的情况
        """
        normalized_msg = message.copy()
        content = message.get("content", [])

        # 如果content是字符串，转换为标准list格式
        if isinstance(content, str):
            normalized_msg["content"] = [{"type": "text", "text": content}]
            return normalized_msg
        # 其他情况（不是list），创建默认的text内容
        elif not isinstance(content, list):
            normalized_msg["content"] = [{"type": "text", "text": str(content)}]
            return normalized_msg

        # content 已经是 list
        if keep_image_url:
            # 保留完整的 image_url/base64 数据，但规范 content 列表结构。
            # 注意：如果 text 中已经包含 <image> 占位符，不再额外追加新的 <image>，
            # 否则会出现占位符数量大于真实图片数量的问题。
            normalized_content = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        normalized_content.append({"type": "text", "text": item.get("text", "")})
                    elif item_type in ("image", "image_url"):
                        normalized_content.append(item)
                    elif item:
                        normalized_content.append({"type": "text", "text": str(item)})
                elif isinstance(item, str) and item.strip():
                    normalized_content.append({"type": "text", "text": item})

            if not normalized_content:
                normalized_content = [{"type": "text", "text": ""}]

            normalized_msg["content"] = normalized_content
            return normalized_msg

        # 检查是否有单独的 image_url item（需要转换的情况）
        text_parts = []
        image_count = 0
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(item.get("text", ""))
                elif item_type in ("image", "image_url"):
                    image_count += 1
            elif isinstance(item, str) and item.strip():
                text_parts.append(item)

        if image_count > 0:
            # Qwen3VL 等格式：image_url 单独在 content 列表中
            # 需要合并到 text 并添加 <image> 占位符
            combined_text = "".join(text_parts)
            # 如果 text 中已有 <image> 占位符，不再重复追加；
            # 否则追加占位符，每个占位符对应一个图像。
            existing_placeholders = combined_text.count("<image>")
            if existing_placeholders < image_count:
                combined_text += "<image>" * (image_count - existing_placeholders)
            normalized_msg["content"] = [{"type": "text", "text": combined_text}]
        else:
            # text 中已包含 <image> 占位符，保持原样
            normalized_msg["content"] = content

        return normalized_msg

    def get_session_info(self) -> Dict[str, str]:
        """获取当前会话信息"""
        return {
            "timestamp": self.timestamp,
            "session_dir": str(self.session_dir),
            "base_output_dir": str(self.base_output_dir),
        }
