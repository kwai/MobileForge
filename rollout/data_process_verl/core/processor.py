#!/usr/bin/env python3
"""
MobileForge数据处理器核心模块

包含主要的数据处理逻辑，负责：
1. 图像映射修复
2. 对话结构分析
3. 训练数据格式转换
"""

import os
import csv
import json
import re
import base64
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  模块级辅助函数
# ════════════════════════════════════════════════════════════════

def _safe_int(val: str) -> Optional[int]:
    """安全地将字符串转换为整数，失败返回 None"""
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def _safe_bool(val: str) -> Optional[bool]:
    """安全地将字符串转换为布尔值，失败返回 None"""
    if not val:
        return None
    return val.strip().lower() in ("true", "1", "yes")


def extract_app_from_description(desc: str) -> str:
    """
    从任务描述中提取 app 名称（启发式）。

    与 data_analyzer/loader.py 中的 extract_app_name 保持一致。
    """
    if not desc:
        return "Unknown"
    patterns = [
        r"^(?:In|Using|Open|Launch)\s+(?:the\s+)?(.+?)(?:\s+app)?\s*,",
        r"^(.+?):\s+",
    ]
    for p in patterns:
        m = re.match(p, desc, re.IGNORECASE)
        if m:
            app = m.group(1).strip()
            if 2 <= len(app) <= 40:
                return app
    return "Unknown"


class MobileForgeDataProcessor:
    """
    MobileForge数据处理器核心类

    负责处理单个任务或少量任务的数据转换
    支持自动检测和处理所有类型的 agent 目录
    """

    # 支持的 agent 名称列表（可自动扩展）
    KNOWN_AGENTS = [
        "UITARS",
        "UITARS_1_5",
        "M3A",
        "M3A_MultiTurn",
        "Qwen3VL",
        "GUIOwl15",
    ]

    def __init__(
        self, rollout_dir: Union[str, List[str]], output_dir: str = "processed_data"
    ):
        """
        初始化处理器

        Args:
            rollout_dir: 包含rollout结果的目录或目录列表
            output_dir: 输出目录
        """
        # 支持多个输入目录
        if isinstance(rollout_dir, str):
            self.rollout_dirs = [Path(rollout_dir)]
        else:
            self.rollout_dirs = [Path(d) for d in rollout_dir]

        self.output_dir = Path(output_dir)

        # results.csv fallback 缓存（懒加载）
        self._csv_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._csv_loaded = False

        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "processed_tasks": 0,
            "successful_trajectories": 0,
            "failed_trajectories": 0,
            "error_trajectories": 0,
            "image_mapping_fixes": 0,
            "placeholder_replacements": 0,
            "input_directories": len(self.rollout_dirs),
            "easy_tasks": 0,
            "hard_tasks": 0,
            "pass1": 0,
            "pass2": 0,
            "pass3": 0,
            "easy_task_ids": [],  # 记录简单任务ID列表
            "hard_task_ids": [],  # 记录困难任务ID列表
            "error_task_ids": [],  # 记录包含error_trajectories的任务ID列表
            "error_attempt_ids": [],  # 记录error_attempt的标识符列表 (task_id|attempt_id)
        }

        logger.info(f"初始化MobileForge数据处理器")
        logger.info(f"输入目录数: {len(self.rollout_dirs)}")
        for i, rollout_dir in enumerate(self.rollout_dirs, 1):
            logger.info(f"  {i}. {rollout_dir}")
        logger.info(f"输出目录: {self.output_dir}")

    def parse_mobile_action(self, action_str: str) -> Dict[str, Any]:
        """解析移动端动作字符串为结构化格式"""
        try:
            if action_str.startswith("open_app"):
                app_match = re.search(r"open_app\(app_name='([^']+)'\)", action_str)
                if app_match:
                    return {"action_type": "open_app", "app_name": app_match.group(1)}

            elif action_str.startswith("click"):
                coord_match = re.search(r"click\(start_box='(\([^)]+\))'\)", action_str)
                if coord_match:
                    coord_str = coord_match.group(1)
                    try:
                        coord_str_clean = re.sub(r"[^\d,()-]", "", coord_str)
                        coords = eval(coord_str_clean) if coord_str_clean else coord_str
                    except:
                        coords = coord_str
                    return {"action_type": "click", "coordinates": coords}

            elif action_str.startswith("type"):
                text_match = re.search(r"type\(content='([^']+)'\)", action_str)
                if text_match:
                    return {"action_type": "type", "text": text_match.group(1)}

            elif action_str.startswith("swipe"):
                swipe_match = re.search(
                    r"swipe\(start=(\([^)]+\)), end=(\([^)]+\))\)", action_str
                )
                if swipe_match:
                    start_coords = eval(swipe_match.group(1))
                    end_coords = eval(swipe_match.group(2))
                    return {
                        "action_type": "swipe",
                        "start_coordinates": start_coords,
                        "end_coordinates": end_coords,
                    }

            elif action_str.startswith("finished"):
                content_match = re.search(r"finished\(content='([^']+)'\)", action_str)
                content = content_match.group(1) if content_match else "Task completed"
                return {"action_type": "finished", "content": content}

            else:
                logger.warning(f"未知动作格式: {action_str}")
                return {"action_type": "unknown", "raw_action": action_str}

        except Exception as e:
            logger.error(f"解析动作 '{action_str}' 时出错: {e}")
            return {
                "action_type": "parse_error",
                "raw_action": action_str,
                "error": str(e),
            }

        return {"action_type": "unknown", "raw_action": action_str}

    def load_image_from_png(self, png_path: Path) -> Optional[str]:
        """从PNG文件加载图像并转换为base64格式"""
        try:
            if not png_path.exists():
                return None

            with Image.open(png_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                buffer = BytesIO()
                img.save(buffer, format="PNG")
                base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{base64_str}"

        except Exception as e:
            logger.error(f"加载图像 {png_path} 时出错: {e}")
            return None

    def load_image_as_pil(self, png_path: Path) -> Optional[Image.Image]:
        """从PNG文件直接加载为PIL Image对象"""
        try:
            if not png_path.exists():
                return None

            img = Image.open(png_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 返回PIL Image对象的副本，避免文件句柄问题
            return img.copy()

        except Exception as e:
            logger.error(f"加载PIL图像 {png_path} 时出错: {e}")
            return None

    def _check_has_base64_images(self, detailed_logs: List[Dict]) -> bool:
        """
        检查detailed_model_logs是否包含base64图像数据

        新版本：直接保存完整的base64图像数据
        旧版本：包含占位符 "[BASE64_IMAGE_DATA_REMOVED_FOR_LOGGING]"
        """
        if not detailed_logs or not isinstance(detailed_logs, list):
            return False

        # 检查前几个步骤的input_messages
        for step_data in detailed_logs[:3]:  # 只检查前3个步骤就足够判断
            if not isinstance(step_data, dict):
                continue

            input_messages = step_data.get("input_messages", [])
            if not isinstance(input_messages, list):
                continue

            for msg in input_messages:
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue

                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue

                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "image_url":
                        continue

                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = str(image_url)

                    # 检查是否包含实际的base64数据
                    if "data:image" in url and "base64," in url:
                        return True
                    elif "[BASE64_IMAGE_DATA_REMOVED_FOR_LOGGING]" in url:
                        return False

        return False

    def analyze_conversation_structure(
        self, input_messages: List[Dict]
    ) -> Dict[str, Any]:
        """分析对话结构，找出图像占位符的位置和数量"""
        image_positions = []
        total_messages = len(input_messages)

        for msg_idx, msg in enumerate(input_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item_idx, item in enumerate(content):
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            # 处理两种图像URL格式：新的直接字符串格式和旧的嵌套格式
                            image_url = item.get("image_url", "")
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url
                            if "[BASE64_IMAGE_DATA_REMOVED_FOR_LOGGING]" in str(url):
                                image_positions.append(
                                    {
                                        "msg_idx": msg_idx,
                                        "item_idx": item_idx,
                                        "position_from_end": total_messages
                                        - msg_idx
                                        - 1,
                                    }
                                )

        return {
            "total_messages": total_messages,
            "image_count": len(image_positions),
            "image_positions": image_positions,
        }

    def create_correct_png_mapping(
        self,
        conversation_analysis: Dict[str, Any],
        available_pngs: List[int],
        current_step: int = None,
    ) -> Dict[Tuple[int, int], int]:
        """
        创建正确的PNG映射（修复版本）

        核心修复逻辑：
        1. UI-TARS保存screenshot为{step-1}.png，即step N保存(N-1).png
        2. 在多轮对话中，UITARS只保留最近5张图像
        3. 图像在对话中的顺序是按时间正序的（最旧的图像在前）
        """
        mapping = {}
        image_positions = conversation_analysis["image_positions"]

        if not image_positions or not available_pngs:
            return mapping

        if current_step is not None and len(available_pngs) > 5:
            # 多轮对话场景：计算该步骤应该看到的PNG列表（时间正序：从最旧到最新）
            step_pngs = []
            for i in range(5):  # 最多5张图像
                target_png_idx = (
                    current_step - 5 + i
                )  # current_step是1-based，PNG是0-based
                if target_png_idx >= 0 and target_png_idx in available_pngs:
                    step_pngs.append(target_png_idx)

            # 如果不足5张，从最早的开始补充
            if len(step_pngs) < 5:
                start_idx = max(0, current_step - len(step_pngs))
                step_pngs = [
                    i for i in range(start_idx, current_step) if i in available_pngs
                ]

            # 按对话位置（从最旧到最新）映射PNG
            for i, pos_info in enumerate(image_positions):
                if i < len(step_pngs):
                    png_idx = step_pngs[i]
                    mapping[(pos_info["msg_idx"], pos_info["item_idx"])] = png_idx
                else:
                    png_idx = step_pngs[-1] if step_pngs else available_pngs[-1]
                    mapping[(pos_info["msg_idx"], pos_info["item_idx"])] = png_idx

            logger.info(
                f"步骤 {current_step}: 创建了 {len(mapping)} 个图像映射，使用PNG {step_pngs}"
            )

        else:
            # 简单场景：按时间顺序映射（最旧的图像位置对应最旧的PNG）
            for i, pos_info in enumerate(image_positions):
                if i < len(available_pngs):
                    png_idx = available_pngs[i]  # 正序取PNG
                    mapping[(pos_info["msg_idx"], pos_info["item_idx"])] = png_idx
                else:
                    png_idx = available_pngs[-1]
                    mapping[(pos_info["msg_idx"], pos_info["item_idx"])] = png_idx

            logger.info(f"简单场景: 创建了 {len(mapping)} 个图像映射")

        return mapping

    def replace_image_placeholders_with_png(
        self, input_messages: List[Dict], attempt_dir: Path, current_step: int = None
    ) -> List[Dict]:
        """修复版本：正确地将input_messages中的图像占位符替换为MobileForge兼容的格式"""
        try:
            # 1. 收集所有可用的PNG文件，转换为base64格式
            png_files = {}
            for png_path in attempt_dir.glob("[0-9]*.png"):
                try:
                    filename = png_path.name
                    if filename.endswith(".png"):
                        index_str = filename[:-4]  # 移除.png后缀
                        if index_str.isdigit():
                            index = int(index_str)
                            # 转换为MobileForge兼容的base64格式
                            base64_str = self.load_image_from_png(png_path)
                            if base64_str:
                                png_files[index] = base64_str
                except ValueError:
                    continue

            if not png_files:
                logger.warning(f"未找到PNG文件在 {attempt_dir}")
                return input_messages

            available_pngs = sorted(png_files.keys())

            # 2. 分析对话结构
            conversation_analysis = self.analyze_conversation_structure(input_messages)

            if conversation_analysis["image_count"] == 0:
                return input_messages

            # 3. 创建正确的PNG映射
            png_mapping = self.create_correct_png_mapping(
                conversation_analysis, available_pngs, current_step
            )

            if not png_mapping:
                logger.warning(f"无法创建PNG映射")
                return input_messages

            # 4. 执行替换，使用MobileForge兼容格式
            processed_messages = []
            replacements_made = 0

            for msg_idx, msg in enumerate(input_messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        processed_content = []
                        for item_idx, item in enumerate(content):
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "image_url"
                            ):
                                image_url = item.get("image_url", {})
                                url = image_url.get("url", "")

                                # 检查是否是需要替换的占位符
                                pos_key = (msg_idx, item_idx)
                                if (
                                    "[BASE64_IMAGE_DATA_REMOVED_FOR_LOGGING]" in url
                                    and pos_key in png_mapping
                                ):
                                    # 使用标准的嵌套image_url格式（OpenAI标准）
                                    png_idx = png_mapping[pos_key]
                                    processed_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": png_files[png_idx]
                                            },  # 恢复嵌套格式
                                        }
                                    )
                                    replacements_made += 1
                                else:
                                    processed_content.append(item)
                            else:
                                processed_content.append(item)

                        processed_messages.append(
                            {"role": msg["role"], "content": processed_content}
                        )
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)

            self.stats["placeholder_replacements"] += replacements_made
            self.stats["image_mapping_fixes"] += 1

            logger.info(f"成功替换了 {replacements_made} 个图像占位符")
            return processed_messages

        except Exception as e:
            logger.error(f"替换图像占位符时出错: {e}")
            return input_messages

    def process_detailed_model_logs(
        self, detailed_log_path: Path, attempt_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        处理详细的模型日志文件（新版本 - 直接使用保存的数据）

        最新的run.py已经在detailed_model_logs.json中保存完整的base64图像数据，
        不再需要图像占位符替换处理。直接使用保存的数据即可。
        """
        try:
            with open(detailed_log_path, "r", encoding="utf-8") as f:
                detailed_logs = json.load(f)

            if not detailed_logs:
                return None

            trajectory = {
                "steps": [],
                "total_steps": 0,
                "success": False,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_api_duration": 0,
                "model_info": {},
                "retry_statistics": {"total_retries": 0, "failed_steps": []},
            }

            # 检查数据格式版本
            has_base64_images = self._check_has_base64_images(detailed_logs)
            logger.info(
                f"检测到数据格式: {'新版本(包含base64图像)' if has_base64_images else '旧版本(需要图像替换)'}"
            )

            # 处理每个详细步骤
            for step_data in detailed_logs:
                if isinstance(step_data, dict) and "step" in step_data:
                    current_step = step_data.get("step", 1)

                    # 获取对话上下文
                    input_messages = step_data.get("input_messages", [])

                    # 根据数据格式版本选择处理方式
                    if has_base64_images:
                        # 新版本：直接使用保存的数据（已包含完整的base64图像数据）
                        # input_messages 已经包含完整的base64图像数据，无需进一步处理
                        logger.debug(
                            f"步骤 {current_step}: 使用新版本格式，直接使用保存的图像数据"
                        )
                    else:
                        # 旧版本：需要图像占位符替换
                        logger.debug(
                            f"步骤 {current_step}: 使用旧版本格式，进行图像替换"
                        )
                        input_messages = self.replace_image_placeholders_with_png(
                            input_messages, attempt_dir, current_step
                        )

                    # 解析动作
                    parsed_action = step_data.get("parsed_action", {})
                    if not parsed_action:
                        raw_response = step_data.get("raw_response", "")
                        parsed_action = self.parse_mobile_action_from_response(
                            raw_response
                        )

                    step_info = {
                        "step_number": current_step,
                        "retry_count": step_data.get("retry_count", 1),
                        "timestamp": step_data.get("timestamp", 0),
                        "input_messages": input_messages,  # 修复后的对话上下文
                        "raw_response": step_data.get("raw_response", ""),
                        "parsed_action": parsed_action,
                        "success": step_data.get("success", False),
                        "error": step_data.get("error", None),
                        "prompt_tokens": step_data.get("prompt_tokens", 0),
                        "completion_tokens": step_data.get("completion_tokens", 0),
                        "total_tokens": step_data.get("total_tokens", 0),
                        "api_call_duration": step_data.get("api_call_duration", 0),
                        "model": step_data.get("model", ""),
                        # DART-GUI Distribution Alignment: rollout log probs
                        "logprobs": step_data.get("logprobs", None),
                        "token_logprobs_sum": step_data.get("token_logprobs_sum", None),
                        "num_completion_tokens_with_logprobs": step_data.get("num_completion_tokens_with_logprobs", None),
                    }

                    trajectory["steps"].append(step_info)

                    # 累加统计信息
                    trajectory["total_prompt_tokens"] += step_info["prompt_tokens"]
                    trajectory["total_completion_tokens"] += step_info[
                        "completion_tokens"
                    ]
                    trajectory["total_api_duration"] += step_info["api_call_duration"]

                    # 记录重试信息
                    if step_info["retry_count"] > 1:
                        trajectory["retry_statistics"]["total_retries"] += (
                            step_info["retry_count"] - 1
                        )
                        trajectory["retry_statistics"]["failed_steps"].append(
                            step_info["step_number"]
                        )

                    if step_info["model"] and not trajectory["model_info"]:
                        trajectory["model_info"] = {"model_name": step_info["model"]}

            trajectory["total_steps"] = len(trajectory["steps"])

            # 判断成功状态（支持status和answer两种完成动作）
            if trajectory["steps"]:
                last_step = trajectory["steps"][-1]
                action_type = last_step.get("parsed_action", {}).get("action_type")
                trajectory["success"] = (
                    action_type
                    in ["status", "answer"]  # 支持status和answer两种完成动作
                    and last_step.get("success", False)
                    and not last_step.get("error")
                )

            return trajectory

        except Exception as e:
            logger.error(f"处理详细模型日志 {detailed_log_path} 时出错: {e}")
            return None

    def parse_mobile_action_from_response(self, raw_response: str) -> Dict[str, Any]:
        """从原始响应中解析移动端动作"""
        try:
            action_match = re.search(
                r"Action:\s*(.+?)(?:\n|$)", raw_response, re.DOTALL
            )
            if not action_match:
                return {"action_type": "unknown", "raw_response": raw_response}

            action_str = action_match.group(1).strip()
            return self.parse_mobile_action(action_str)

        except Exception as e:
            logger.error(f"从响应中解析动作时出错: {e}")
            return {
                "action_type": "parse_error",
                "raw_response": raw_response,
                "error": str(e),
            }

    def _normalize_message_content(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """标准化消息内容格式，确保content字段始终为list格式"""
        normalized_msg = message.copy()
        content = message.get("content", [])

        # 如果content是字符串，转换为标准list格式
        if isinstance(content, str):
            normalized_msg["content"] = [{"type": "text", "text": content}]
        # 如果content已经是list，保持不变
        elif isinstance(content, list):
            normalized_msg["content"] = content
        # 其他情况，创建默认的text内容
        else:
            normalized_msg["content"] = [{"type": "text", "text": str(content)}]

        return normalized_msg

    def process_evaluation_summary(self, eval_path: Path) -> Optional[Dict[str, Any]]:
        """处理轨迹的评估汇总"""
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            # 提取步骤级的合理性分析
            step_reasonableness = eval_data.get("step_reasonableness_analysis", {})
            reasonable_steps = step_reasonableness.get("reasonable_steps", [])
            unreasonable_steps = step_reasonableness.get("unreasonable_steps", [])
            step_analysis_details = step_reasonableness.get("step_analysis", {})

            # 构建步骤级正负样本标注
            step_labels = {}
            for step_num in reasonable_steps:
                step_labels[step_num] = {
                    "label": "positive",
                    "reasonableness": "reasonable",
                    "explanation": step_analysis_details.get(str(step_num), {}).get(
                        "explanation", ""
                    ),
                    "impact": step_analysis_details.get(str(step_num), {}).get(
                        "impact", "positive"
                    ),
                }

            for step_num in unreasonable_steps:
                step_labels[step_num] = {
                    "label": "negative",
                    "reasonableness": "unreasonable",
                    "explanation": step_analysis_details.get(str(step_num), {}).get(
                        "explanation", ""
                    ),
                    "impact": step_analysis_details.get(str(step_num), {}).get(
                        "impact", "negative"
                    ),
                }

            # 从 final_decision.json 中提取任务可行性评估信息
            task_feasibility = self._load_task_feasibility(eval_path.parent)

            return {
                "task_identifier": eval_data.get("task_identifier", ""),
                "task_description": eval_data.get("task_description", ""),
                "final_result": eval_data.get("final_result", 0),
                "final_reason": eval_data.get("final_reason", ""),
                "step_analysis": eval_data.get("step_by_step_analysis", {}),
                "reasonableness_analysis": eval_data.get(
                    "step_reasonableness_analysis", {}
                ),
                "step_labels": step_labels,  # 新增：步骤级标注
                "reasonable_steps": reasonable_steps,  # 新增：合理步骤列表
                "unreasonable_steps": unreasonable_steps,  # 新增：不合理步骤列表
                "task_feasible": task_feasibility.get("task_feasible", None),
                "task_feasible_reason": task_feasibility.get("task_feasible_reason", ""),
                "task_barriers": task_feasibility.get("task_barriers", []),
            }

        except Exception as e:
            logger.error(f"处理评估 {eval_path} 时出错: {e}")
            return None

    # ────────────────────────────────────────────────────────
    # results.csv fallback + app_name 推断
    # ────────────────────────────────────────────────────────

    def _ensure_csv_fallback(self) -> None:
        """
        懒加载 results.csv，缓存每个 task_identifier 的元数据。
        仅在首次调用时读取，之后直接使用缓存。
        """
        if self._csv_loaded:
            return
        for rollout_dir in self.rollout_dirs:
            for candidate in [rollout_dir / "results.csv", rollout_dir.parent / "results.csv"]:
                if not candidate.exists():
                    continue
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            tid = row.get("task_identifier", "")
                            if not tid:
                                continue
                            self._csv_metadata_cache[tid] = {
                                "app_name": row.get("app_name", ""),
                                "app_package": row.get("app_package", ""),
                                "golden_steps": _safe_int(row.get("golden_steps", "")),
                                "trajectory_id": row.get("trajectory_id", ""),
                                "original_goal": row.get("original_goal", ""),
                                "task_reasonable": _safe_bool(row.get("task_reasonable", "")),
                                "task_completed": _safe_bool(row.get("task_completed", "")),
                                "task_id": row.get("task_id", ""),
                                "difficulty_level": row.get("difficulty_level", ""),
                                "core_functionality": row.get("core_functionality", ""),
                                "variation_type": row.get("variation_type", ""),
                                "prerequisites": row.get("prerequisites", ""),
                            }
                    logger.info(
                        f"Fallback: 从 {candidate} 加载了 "
                        f"{len(self._csv_metadata_cache)} 条任务元数据"
                    )
                except Exception as e:
                    logger.warning(f"读取 {candidate} 失败: {e}")
        self._csv_loaded = True

    def _load_task_metadata(self, attempt_dir: Path, task_id: str = "") -> Dict[str, Any]:
        """
        加载任务级元数据。

        优先级：
          1. attempt 目录下的 task_metadata.json
          2. results.csv (fallback)
          3. 空字典

        Args:
            attempt_dir: attempt 目录路径
            task_id: 任务 ID（用于 CSV fallback 查找）

        Returns:
            任务级元数据字典 (app_name, golden_steps 等)
        """
        # 1) 优先从 task_metadata.json 读取
        meta_path = attempt_dir / "task_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("app_name"):
                    return meta
            except Exception as e:
                logger.debug(f"读取 task_metadata.json 失败 ({meta_path}): {e}")

        # 2) fallback: results.csv
        if task_id:
            self._ensure_csv_fallback()
            csv_meta = self._csv_metadata_cache.get(task_id)
            if csv_meta and csv_meta.get("app_name"):
                logger.debug(f"任务 {task_id}: 使用 results.csv fallback 获取元数据")
                return csv_meta

        # 3) 空
        return {}

    def _load_log_actions(self, attempt_dir: Path) -> Dict[str, Any]:
        """
        从 log.json 中加载 action 序列并进行死循环检测

        Args:
            attempt_dir: attempt 目录路径

        Returns:
            包含 log_actions, action_by_step, max_consecutive_same_actions,
            loop_detail 的字典
        """
        result = {
            "log_actions": [],
            "action_by_step": {},
            "max_consecutive_same_actions": 0,
            "loop_detail": "",
        }
        log_path = attempt_dir / "log.json"
        if not log_path.exists():
            return result
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            actions = []
            for entry in log_data:
                if isinstance(entry, dict):
                    actions.append({
                        "step": entry.get("step", 0),
                        "action": entry.get("action"),
                    })
            result["log_actions"] = actions
            result["action_by_step"] = {a["step"]: a["action"] for a in actions}

            # 死循环检测
            if len(actions) >= 2:
                import json as json_mod
                strs = [json_mod.dumps(a.get("action"), sort_keys=True) for a in actions]
                max_c = cur_c = 1
                max_action = ""
                max_start = 0
                for i in range(1, len(strs)):
                    if strs[i] == strs[i - 1]:
                        cur_c += 1
                        if cur_c > max_c:
                            max_c = cur_c
                            max_action = strs[i][:120]
                            max_start = actions[i - cur_c + 1]["step"]
                    else:
                        cur_c = 1
                result["max_consecutive_same_actions"] = max_c
                if max_c >= 2:
                    result["loop_detail"] = f"步骤 {max_start} 起连续 {max_c} 次: {max_action}"
        except Exception as e:
            logger.debug(f"读取 log.json 失败 ({log_path}): {e}")
        return result

    def _check_hint_files(self, attempt_dir: Path) -> Dict[str, bool]:
        """
        检查 eval_hint.json 和 hints_input.json 是否存在

        Args:
            attempt_dir: attempt 目录路径

        Returns:
            {"has_eval_hint": bool, "has_hints_input": bool}
        """
        return {
            "has_eval_hint": (attempt_dir / "eval_hint.json").exists(),
            "has_hints_input": (attempt_dir / "hints_input.json").exists(),
        }

    def _load_task_feasibility(self, attempt_dir: Path) -> Dict[str, Any]:
        """
        从 final_decision.json 中加载任务可行性评估信息

        Args:
            attempt_dir: attempt 目录路径（evaluation_summary.json 所在目录）

        Returns:
            包含 task_feasible, task_feasible_reason, task_barriers 的字典
        """
        default = {
            "task_feasible": None,
            "task_feasible_reason": "",
            "task_barriers": [],
        }
        final_decision_path = attempt_dir / "final_decision.json"
        if not final_decision_path.exists():
            logger.debug(f"未找到 final_decision.json: {final_decision_path}")
            return default

        try:
            with open(final_decision_path, "r", encoding="utf-8") as f:
                decision_data = json.load(f)

            return {
                "task_feasible": decision_data.get("task_feasible", None),
                "task_feasible_reason": decision_data.get("task_feasible_reason", ""),
                "task_barriers": decision_data.get("task_barriers", []),
            }
        except Exception as e:
            logger.warning(f"读取 final_decision.json 出错: {e}")
            return default

    def convert_to_training_format(
        self, task_id: str, trajectory: Dict[str, Any], evaluation: Dict[str, Any],
        attempt_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """从详细日志转换为训练格式（步骤级标注版本，修复重试处理）"""

        # 从详细日志构建完整的对话格式（现在图像映射是正确的）
        conversation = []

        # 处理每个步骤的完整对话上下文
        for i, step in enumerate(trajectory["steps"]):
            input_messages = step.get("input_messages", [])

            # 添加input_messages中的所有消息（现在包含正确的图像），确保格式一致
            for msg in input_messages:
                normalized_msg = self._normalize_message_content(msg)
                conversation.append(normalized_msg)

            # 添加助手的响应，使用标准化格式
            if step.get("raw_response") and step.get("success", False):
                assistant_msg = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": step["raw_response"]}],
                }
                # 使用标准化方法确保格式一致
                normalized_assistant_msg = self._normalize_message_content(
                    assistant_msg
                )
                conversation.append(normalized_assistant_msg)

        # 获取步骤级标注信息
        step_labels = evaluation.get("step_labels", {})
        reasonable_steps = evaluation.get("reasonable_steps", [])
        unreasonable_steps = evaluation.get("unreasonable_steps", [])

        # 从 attempt_extra 中获取额外信息
        extra = attempt_extra or {}
        task_meta = extra.get("task_metadata", {})
        log_info = extra.get("log_info", {})
        hint_info = extra.get("hint_info", {})

        # 确定 app 名称（三级回退: task_metadata.json → results.csv → task_description 推断）
        app_name = task_meta.get("app_name", "")
        if not app_name:
            task_desc = evaluation.get("task_description", "")
            app_name = extract_app_from_description(task_desc) if task_desc else "Unknown"

        # 构建增强的元数据
        enhanced_metadata = {
            "total_steps": trajectory["total_steps"],
            "total_tokens": trajectory["total_prompt_tokens"]
            + trajectory["total_completion_tokens"],
            "total_api_duration": trajectory.get("total_api_duration", 0),
            "model_info": trajectory.get("model_info", {}),
            "retry_statistics": trajectory.get("retry_statistics", {}),
            "app": app_name,
            "difficulty_level": task_meta.get("difficulty_level", ""),
            "golden_steps": task_meta.get("golden_steps"),
            "step_details": [],
            "image_mapping_fixed": True,  # 标记图像映射已修复
            "step_level_labels": step_labels,  # 新增：步骤级标注
            "reasonable_steps": reasonable_steps,  # 新增：合理步骤列表
            "unreasonable_steps": unreasonable_steps,  # 新增：不合理步骤列表
            # 任务可行性评估（来自 final_decision.json）
            "task_feasible": evaluation.get("task_feasible", None),
            "task_feasible_reason": evaluation.get("task_feasible_reason", ""),
            "task_barriers": evaluation.get("task_barriers", []),
            # 死循环检测（来自 log.json）
            "max_consecutive_same_actions": log_info.get("max_consecutive_same_actions", 0),
            "loop_detail": log_info.get("loop_detail", ""),
            # Eval Hint 相关
            "has_eval_hint": hint_info.get("has_eval_hint", False),
            "has_hints_input": hint_info.get("has_hints_input", False),
            # 任务级元数据
            "task_metadata": task_meta,
        }

        # 修复：对重试步骤进行去重处理，每个step_number只保留一个代表性的执行
        unique_steps = {}
        for step in trajectory["steps"]:
            step_number = step["step_number"]
            if step_number not in unique_steps:
                # 第一次遇到这个步骤号，直接添加
                unique_steps[step_number] = step
            else:
                # 已有该步骤，选择更有代表性的执行（优先选择成功的，或最后一次执行）
                existing_step = unique_steps[step_number]
                # 如果当前步骤成功而已有步骤失败，则替换
                if step.get("success", False) and not existing_step.get(
                    "success", False
                ):
                    unique_steps[step_number] = step
                # 如果都成功或都失败，选择重试次数更高的（最新的执行）
                elif step.get("retry_count", 1) > existing_step.get("retry_count", 1):
                    unique_steps[step_number] = step

        logger.info(
            f"任务 {task_id}: 去重前 {len(trajectory['steps'])} 个步骤条目，去重后 {len(unique_steps)} 个唯一步骤"
        )

        # 获取 log.json 中的 action 数据（完整 action，包含坐标等信息）
        action_by_step = log_info.get("action_by_step", {})

        # 添加每步的详细信息，使用去重后的步骤
        for step_number in sorted(unique_steps.keys()):
            step = unique_steps[step_number]
            step_label_info = step_labels.get(step_number, {})

            step_detail = {
                "step_number": step_number,
                "retry_count": step["retry_count"],
                "success": step["success"],
                "error": step.get("error"),
                "action_type": step.get("parsed_action", {}).get("action_type"),
                "action": action_by_step.get(step_number),  # 完整 action (来自 log.json)
                "input_messages": step.get("input_messages", []),  # 修复后的多轮对话
                "raw_response": step.get("raw_response", ""),  # 助手回复
                "tokens": {
                    "prompt": step["prompt_tokens"],
                    "completion": step["completion_tokens"],
                    "total": step["total_tokens"],
                },
                "api_duration": step["api_call_duration"],
                # 新增：步骤级标注信息
                "step_label": step_label_info.get(
                    "label", "unknown"
                ),  # positive/negative/unknown
                "reasonableness": step_label_info.get("reasonableness", "unknown"),
                "reasonableness_explanation": step_label_info.get("explanation", ""),
                "impact": step_label_info.get("impact", "unknown"),
                # 新增：重试信息
                "is_deduplicated": step["retry_count"] > 1
                or len(
                    [s for s in trajectory["steps"] if s["step_number"] == step_number]
                )
                > 1,
                "total_attempts": len(
                    [s for s in trajectory["steps"] if s["step_number"] == step_number]
                ),
                # DART-GUI Distribution Alignment: rollout log probs
                "logprobs": step.get("logprobs", None),
                "token_logprobs_sum": step.get("token_logprobs_sum", None),
                "num_completion_tokens_with_logprobs": step.get("num_completion_tokens_with_logprobs", None),
            }
            enhanced_metadata["step_details"].append(step_detail)

        return {
            "task_id": task_id,
            "task_description": evaluation.get("task_description", ""),
            "golden_steps": trajectory["total_steps"],
            "conversation": conversation,
            "success": trajectory["success"],
            "evaluation_result": evaluation.get("final_result", 0),
            "evaluation_reason": evaluation.get("final_reason", ""),
            "metadata": enhanced_metadata,
            "data_source": "detailed_model_logs_step_labeled",  # 标记数据来源为步骤级标注版本
        }

    def _detect_agent_dirs(self, task_dir: Path) -> List[Tuple[str, Path]]:
        """
        自动检测任务目录中存在的 agent 目录

        Returns:
            List of (agent_name, agent_dir_path) tuples
        """
        detected_agents = []

        # 首先检查已知的 agent 名称
        for agent_name in self.KNOWN_AGENTS:
            agent_dir = task_dir / agent_name
            if agent_dir.exists() and agent_dir.is_dir():
                detected_agents.append((agent_name, agent_dir))

        # 同时检测可能的新 agent（任务目录下的所有子目录，包含 attempt_* 的）
        for subdir in task_dir.iterdir():
            if subdir.is_dir() and subdir.name not in [a[0] for a in detected_agents]:
                # 检查是否包含 attempt_* 子目录（agent 目录的特征）
                attempt_dirs = list(subdir.glob("attempt_*"))
                if attempt_dirs:
                    logger.info(f"检测到新的 agent 目录: {subdir.name}")
                    detected_agents.append((subdir.name, subdir))

        return detected_agents

    def process_single_task(self, task_dir: Path) -> List[Dict[str, Any]]:
        """处理单个任务的所有attempts（修复版本 - 支持所有 agent）"""
        task_id = task_dir.name
        logger.info(f"处理任务: {task_id}")

        training_samples = []
        attempt_stats = {"total_attempts": 0, "pass@attempts": 0}

        # 自动检测任务目录中的 agent 目录
        detected_agents = self._detect_agent_dirs(task_dir)

        if not detected_agents:
            logger.warning(f"任务 {task_id} 未找到任何 agent 目录")
            return training_samples

        logger.info(
            f"任务 {task_id} 检测到 {len(detected_agents)} 个 agent: {[a[0] for a in detected_agents]}"
        )

        # 处理每个检测到的 agent 目录
        for agent_name, agent_dir in detected_agents:
            logger.debug(f"处理 agent: {agent_name}")

            # 处理该 agent 的每个 attempt
            for attempt_dir in sorted(agent_dir.glob("attempt_*")):
                attempt_num = attempt_dir.name
                logger.debug(f"处理 {task_id}/{agent_name}/{attempt_num}")

                # 处理评估
                eval_path = attempt_dir / "evaluation_summary.json"
                if not eval_path.exists():
                    logger.warning(
                        f"{task_id}/{agent_name}/{attempt_num} 未找到evaluation_summary.json"
                    )
                    continue

                evaluation = self.process_evaluation_summary(eval_path)
                if not evaluation:
                    continue

                # 加载额外信息 (task_metadata / log.json / hint 文件)
                task_meta = self._load_task_metadata(attempt_dir, task_id=task_id)
                log_info = self._load_log_actions(attempt_dir)
                hint_info = self._check_hint_files(attempt_dir)
                attempt_extra = {
                    "task_metadata": task_meta,
                    "log_info": log_info,
                    "hint_info": hint_info,
                }

                # 使用detailed_model_logs.json（使用修复版本处理）
                detailed_log_path = attempt_dir / "detailed_model_logs.json"
                trajectory = None
                training_sample = None

                if detailed_log_path.exists():
                    logger.debug(
                        f"使用修复版详细模型日志: {task_id}/{agent_name}/{attempt_num}"
                    )
                    # 传递attempt_dir以便进行图像修复
                    trajectory = self.process_detailed_model_logs(
                        detailed_log_path, attempt_dir
                    )

                    if trajectory:
                        # 使用修复版本的格式转换
                        training_sample = self.convert_to_training_format(
                            task_id, trajectory, evaluation,
                            attempt_extra=attempt_extra,
                        )
                        training_sample["attempt"] = attempt_num
                        training_sample["agent_name"] = agent_name  # 记录 agent 名称

                        # 统计
                        attempt_stats["total_attempts"] += 1
                        if evaluation.get("final_result", 0) == 1:
                            attempt_stats["pass@attempts"] += 1

                        # 更新统计：优先使用 evaluation.final_result 判断
                        # - final_result = 1: 评估成功（无论trajectory success如何）
                        # - final_result = 0: 评估失败
                        # - final_result = -1 或其他: 评估错误
                        final_result = evaluation.get("final_result", -1)

                        if final_result == 1:
                            # 评估成功就算成功，即使trajectory没有使用answer/status结束
                            self.stats["successful_trajectories"] += 1
                        elif final_result == 0:
                            # 评估失败
                            self.stats["failed_trajectories"] += 1
                        else:
                            # 评估错误（-1或其他值）
                            self.stats["error_trajectories"] += 1
                            # 记录包含error的任务ID（去重）
                            if task_id not in self.stats["error_task_ids"]:
                                self.stats["error_task_ids"].append(task_id)
                            # 记录error_attempt的标识符
                            error_attempt_key = f"{task_id}|{attempt_num}"
                            if error_attempt_key not in self.stats["error_attempt_ids"]:
                                self.stats["error_attempt_ids"].append(
                                    error_attempt_key
                                )

                if not training_sample:
                    logger.warning(
                        f"{task_id}/{agent_name}/{attempt_num} 无法处理详细日志"
                    )

                if training_sample:
                    training_samples.append(training_sample)

        # 计算任务级 avg_sr 并注入到每个 sample 中
        avg_sr = (
            attempt_stats["pass@attempts"] / attempt_stats["total_attempts"]
            if attempt_stats["total_attempts"] > 0 else 0.0
        )
        for training_sample in training_samples:
            training_sample["attempt_stats"] = attempt_stats
            training_sample["avg_sr"] = avg_sr
            # 也写入 metadata，方便训练数据筛选
            if "metadata" in training_sample:
                training_sample["metadata"]["avg_sr"] = avg_sr

        if attempt_stats["pass@attempts"] == attempt_stats["total_attempts"]:
            self.stats["easy_tasks"] += 1
            self.stats["easy_task_ids"].append(task_id)
        elif attempt_stats["pass@attempts"] == 0:
            self.stats["hard_tasks"] += 1
            self.stats["hard_task_ids"].append(task_id)
        elif attempt_stats["pass@attempts"] == 1:
            self.stats["pass1"] += 1
        elif attempt_stats["pass@attempts"] == 2:
            self.stats["pass2"] += 1
        elif attempt_stats["pass@attempts"] == 3:
            self.stats["pass3"] += 1

        return training_samples

    def filter_training_data(
        self, training_samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将训练数据过滤为正负样本"""
        positive_samples = []
        negative_samples = []

        for sample in training_samples:
            if sample["success"] and sample["evaluation_result"] == 1:
                positive_samples.append(sample)
            elif sample["evaluation_result"] == 0:
                negative_samples.append(sample)

        return positive_samples, negative_samples

    # ────────────────────────────────────────────────────────
    # 智能任务选择：balanced max_tasks
    # ────────────────────────────────────────────────────────

    def _get_task_app_name(self, task_dir: Path) -> str:
        """
        获取任务对应的 app 名称。

        优先级：
          1. attempt 目录下的 task_metadata.json
          2. results.csv (fallback)
          3. 从 evaluation_summary.json 的 task_description 推断
        """
        task_id = task_dir.name

        # 1) 尝试从 task_metadata.json 读取
        for agent_dir in task_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for attempt_dir in sorted(agent_dir.glob("attempt_*")):
                meta_path = attempt_dir / "task_metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        app = meta.get("app_name", "")
                        if app:
                            return app
                    except Exception:
                        pass

        # 2) fallback: results.csv
        self._ensure_csv_fallback()
        csv_meta = self._csv_metadata_cache.get(task_id)
        if csv_meta and csv_meta.get("app_name"):
            return csv_meta["app_name"]

        # 3) 从 evaluation_summary.json 的 task_description 推断
        for agent_dir in task_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for attempt_dir in sorted(agent_dir.glob("attempt_*")):
                eval_path = attempt_dir / "evaluation_summary.json"
                if eval_path.exists():
                    try:
                        with open(eval_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        desc = data.get("task_description", "")
                        if desc:
                            app = extract_app_from_description(desc)
                            if app != "Unknown":
                                return app
                    except Exception:
                        pass

        return "Unknown"

    @staticmethod
    def _check_task_eval_complete(task_dir: Path) -> bool:
        """
        检查一个任务的所有 attempt 是否都完成了评测
        （每个 attempt 目录都存在 final_decision.json）
        """
        has_any_attempt = False
        for agent_dir in task_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for attempt_dir in sorted(agent_dir.glob("attempt_*")):
                has_any_attempt = True
                if not (attempt_dir / "final_decision.json").exists():
                    return False
        return has_any_attempt  # 如果没有 attempt 也认为不完整

    def _balanced_task_selection(
        self, all_task_dirs: List[Path], max_tasks: int, seed: int = 42
    ) -> List[Path]:
        """
        在 app 之间均衡地选取 max_tasks 个任务。

        选择策略（按优先级）：
        1. 优先保证每个 app 下的任务数量均衡
        2. 在同一 app 内，优先选择完成评测的任务（所有 attempt 都有 final_decision.json）
        3. 若完成评测的任务不够，用未完成的补齐

        Args:
            all_task_dirs: 所有候选任务目录
            max_tasks: 目标选取数量
            seed: 随机种子

        Returns:
            选中的任务目录列表
        """
        rng = random.Random(seed)

        # 1. 收集每个任务的 app 和评测完成状态
        task_info = []
        for td in all_task_dirs:
            app = self._get_task_app_name(td)
            complete = self._check_task_eval_complete(td)
            task_info.append({"dir": td, "app": app, "complete": complete})

        # 2. 按 app 分组
        app_groups: Dict[str, List[dict]] = {}
        for t in task_info:
            app_groups.setdefault(t["app"], []).append(t)

        num_apps = len(app_groups)
        if num_apps == 0:
            return []

        logger.info(f"均衡任务选择: 共 {len(all_task_dirs)} 个任务, {num_apps} 个 app, 目标 {max_tasks} 个")
        for app, tasks in sorted(app_groups.items()):
            complete_cnt = sum(1 for t in tasks if t["complete"])
            logger.info(f"  {app}: {len(tasks)} 个任务 (已完成评测: {complete_cnt})")

        # 3. 计算每个 app 的基础配额
        base_quota = max_tasks // num_apps
        remainder = max_tasks % num_apps

        # 按 app 中的任务数升序排列（小 app 优先分配 remainder）
        sorted_apps = sorted(app_groups.keys(), key=lambda a: len(app_groups[a]))

        selected = []
        deficit = 0  # 某些 app 无法填满配额时的赤字

        for i, app in enumerate(sorted_apps):
            quota = base_quota + (1 if i < remainder else 0) + deficit
            deficit = 0
            tasks = app_groups[app]

            # 在 app 内按 complete 排序（完成的在前），再随机打乱同类
            complete_tasks = [t for t in tasks if t["complete"]]
            incomplete_tasks = [t for t in tasks if not t["complete"]]
            rng.shuffle(complete_tasks)
            rng.shuffle(incomplete_tasks)
            candidates = complete_tasks + incomplete_tasks

            if len(candidates) >= quota:
                selected.extend(candidates[:quota])
            else:
                selected.extend(candidates)
                deficit = quota - len(candidates)  # 传递给后续 app

        # 4. 如果还有赤字（所有 app 都无法补齐），从未选中的任务里补充
        if len(selected) < max_tasks:
            selected_dirs = {t["dir"] for t in selected}
            unselected = [t for t in task_info if t["dir"] not in selected_dirs]
            complete_unsel = [t for t in unselected if t["complete"]]
            incomplete_unsel = [t for t in unselected if not t["complete"]]
            rng.shuffle(complete_unsel)
            rng.shuffle(incomplete_unsel)
            additional = (complete_unsel + incomplete_unsel)[: max_tasks - len(selected)]
            selected.extend(additional)

        result_dirs = [t["dir"] for t in selected]
        logger.info(f"均衡选择完成: 选中 {len(result_dirs)} 个任务")

        # 输出选择后每个 app 的分布
        result_app_counts: Dict[str, int] = {}
        for t in selected:
            result_app_counts[t["app"]] = result_app_counts.get(t["app"], 0) + 1
        for app, cnt in sorted(result_app_counts.items()):
            logger.info(f"  选中 {app}: {cnt} 个任务")

        return result_dirs

    def process_all_tasks(
        self, max_tasks: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """处理rollout目录中的所有任务（支持多目录合并处理）"""
        logger.info(
            f"开始处理来自 {len(self.rollout_dirs)} 个目录的rollout数据（图像映射修复版本）"
        )

        all_training_samples = []
        all_task_dirs = []

        # 收集所有输入目录中的任务
        for rollout_dir in self.rollout_dirs:
            logger.info(f"扫描目录: {rollout_dir}")
            task_dirs_in_dir = list(rollout_dir.glob("*"))
            for task_dir in task_dirs_in_dir:
                if task_dir.is_dir():
                    all_task_dirs.append(task_dir)
            logger.info(f"  发现 {len(task_dirs_in_dir)} 个任务目录")

        logger.info(f"总共发现 {len(all_task_dirs)} 个任务目录")

        # 应用最大任务数限制（均衡选择）
        if max_tasks and max_tasks < len(all_task_dirs):
            all_task_dirs = self._balanced_task_selection(all_task_dirs, max_tasks)
            logger.info(f"应用均衡任务选择，选取 {len(all_task_dirs)} 个任务")

        self.stats["total_tasks"] = len(all_task_dirs)

        # 处理所有任务
        for i, task_dir in enumerate(all_task_dirs, 1):
            try:
                logger.info(f"处理进度: {i}/{len(all_task_dirs)} - {task_dir.name}")
                samples = self.process_single_task(task_dir)
                all_training_samples.extend(samples)
                self.stats["processed_tasks"] += 1
            except Exception as e:
                logger.error(f"处理任务 {task_dir.name} 时出错: {e}")

        # 过滤为正负样本
        positive_samples, negative_samples = self.filter_training_data(
            all_training_samples
        )

        logger.info(f"多目录合并处理完成。统计信息: {self.stats}")
        logger.info(f"正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")

        return {
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "all_samples": all_training_samples,
            "statistics": self.stats,
        }
