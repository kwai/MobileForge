"""
Adaptive Reward Function for MobileForge Training

根据训练数据中的 system prompt 自动识别模型类型，并应用对应的格式检查和 reward 计算：

支持的模型类型：
1. qwen3vl: 使用 <thinking>, <tool_call>, <conclusion> 格式
2. guiowl: 使用 Action:, <tool_call> 格式

自动检测逻辑：
- system prompt 以 "# Tools" 开头 → guiowl
- system prompt 以 "You are a helpful assistant" 开头 → qwen3vl

评分规则：
- format_reward: 检查输出格式是否正确
- action_reward: 验证动作类型和参数的准确性
- overall = action_type_weight * action_type_reward + action_params_weight * action_params_reward
"""

import json
import re
from typing import Tuple


# Metadata - EasyR1框架要求
REWARD_NAME = "adaptive_gui"
REWARD_TYPE = "batch"  # 批量处理模式

# 模型类型检测特征
_QWEN3VL_SYSTEM_PROMPT_START = "You are a helpful assistant"
_GUIOWL_SYSTEM_PROMPT_START = "# Tools"

# 动作名称标准化映射
# 不同的 agent 可能使用不同的动作名称，这里统一标准化
# 格式: {"alias1": "canonical", "alias2": "canonical", ...}
_ACTION_NAME_MAPPING = {
    # Qwen3VL -> 标准名称
    "drag": "swipe",  # Qwen3VL 的 drag 在其他系统中是 swipe
    "input_text": "type",  # Qwen3VL 的 input_text 是 type

    # GUIOwl 标注格式 -> 标准名称
    "status": "terminate",  # GUIOwl 标注的 status 对应 terminate

    # 其他别名
    "navigate_back": "system_button",
    "navigate_home": "system_button",
    "navigate_menu": "system_button",
    "keyboard_enter": "system_button",
}

# 标准动作列表（用于验证）
_STANDARD_ACTIONS = {
    "click", "long_press", "swipe", "type", "system_button",
    "open", "wait", "answer", "terminate", "key"
}


def normalize_action_name(action: str) -> str:
    """
    将动作名称标准化为统一格式

    Args:
        action: 原始动作名称 (如 "drag", "input_text", "type")

    Returns:
        标准化后的动作名称 (如 "swipe", "type")
    """
    if not action:
        return action
    return _ACTION_NAME_MAPPING.get(action.lower(), action.lower())


def detect_model_type(system_prompt: str) -> str:
    """
    根据 system prompt 检测模型类型

    Args:
        system_prompt: system message 的 content

    Returns:
        "qwen3vl", "guiowl", 或 "unknown"
    """
    if not system_prompt:
        return "unknown"

    system_prompt_stripped = system_prompt.strip()

    # 检查 Qwen3VL 特征
    if system_prompt_stripped.startswith(_QWEN3VL_SYSTEM_PROMPT_START):
        return "qwen3vl"

    # 检查 GUIOwl 特征
    if system_prompt_stripped.startswith(_GUIOWL_SYSTEM_PROMPT_START):
        return "guiowl"

    return "unknown"


def extract_system_prompt_from_conversation(conversations: list) -> str:
    """从对话中提取 system prompt"""
    for msg in conversations:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
    return ""


def qwen3vl_format_reward(predict_str: str) -> float:
    """
    检查 Qwen3VL 格式：
    1. 包含 <thinking></thinking>
    2. 包含 <tool_call></tool_call>，内部是有效的 JSON
    3. 包含 <conclusion></conclusion>（必需）

    正确的输出顺序: <thinking> -> <tool_call> -> <conclusion>
    """
    # 检查 thinking 标签
    thinking_pattern = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
    if not thinking_pattern.search(predict_str):
        return 0.0

    # 检查 tool_call 标签
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
    tool_call_match = tool_call_pattern.search(predict_str)
    if not tool_call_match:
        return 0.0

    # 检查 conclusion 标签（必需）
    conclusion_pattern = re.compile(r"<conclusion>.*?</conclusion>", re.DOTALL | re.IGNORECASE)
    if not conclusion_pattern.search(predict_str):
        return 0.0

    # 验证 tool_call 内的 JSON 格式
    try:
        tool_call_json = json.loads(tool_call_match.group(1).strip())

        # 验证必需字段
        if "arguments" in tool_call_json:
            action_args = tool_call_json["arguments"]
        else:
            action_args = {k: v for k, v in tool_call_json.items() if k != "name"}

        if "action" not in action_args:
            return 0.0

        action = action_args["action"]

        # 验证各类动作的必需参数
        if action in ["click", "long_press"]:
            if "coordinate" not in action_args:
                return 0.0
            coord = action_args["coordinate"]
            if not (isinstance(coord, list) and len(coord) >= 2):
                return 0.0

        if action == "swipe":
            coord = action_args.get("coordinate", [])
            coord2 = action_args.get("coordinate2", [])
            direction = action_args.get("direction", "")
            if not direction and not (len(coord) >= 2 and len(coord2) >= 2):
                return 0.0

        if action == "type":
            if "text" not in action_args or not action_args["text"]:
                return 0.0

        if action == "system_button":
            if "button" not in action_args:
                return 0.0

        if action == "terminate":
            if "status" not in action_args:
                return 0.0

        return 1.0
    except json.JSONDecodeError:
        return 0.0


def guiowl_format_reward(predict_str: str) -> float:
    """
    检查 GUIOwl 格式：
    1. 包含 "Action:" 行
    2. 包含 <tool_call></tool_call>，内部是有效的 JSON

    不需要 <thinking> 和 <conclusion> 标签
    """
    # 检查 Action: 标签
    action_pattern = re.compile(r"Action:\s*.+", re.IGNORECASE)
    if not action_pattern.search(predict_str):
        return 0.0

    # 检查 tool_call 标签
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
    tool_call_match = tool_call_pattern.search(predict_str)
    if not tool_call_match:
        return 0.0

    # 验证 tool_call 内的 JSON 格式
    try:
        tool_call_json = json.loads(tool_call_match.group(1).strip())

        # 验证必需字段
        if "arguments" in tool_call_json:
            action_args = tool_call_json["arguments"]
        else:
            action_args = {k: v for k, v in tool_call_json.items() if k != "name"}

        if "action" not in action_args:
            return 0.0

        action = action_args["action"]

        # 验证各类动作的必需参数
        if action in ["click", "long_press"]:
            if "coordinate" not in action_args:
                return 0.0
            coord = action_args["coordinate"]
            if not (isinstance(coord, list) and len(coord) >= 2):
                return 0.0

        if action == "swipe":
            coord = action_args.get("coordinate", [])
            coord2 = action_args.get("coordinate2", [])
            direction = action_args.get("direction", "")
            if not direction and not (len(coord) >= 2 and len(coord2) >= 2):
                return 0.0

        if action == "type":
            if "text" not in action_args or not action_args["text"]:
                return 0.0

        if action == "system_button":
            if "button" not in action_args:
                return 0.0

        if action == "terminate":
            if "status" not in action_args:
                return 0.0

        return 1.0
    except json.JSONDecodeError:
        return 0.0


def format_reward(predict_str: str, model_type: str) -> float:
    """
    根据模型类型计算格式分数

    Args:
        predict_str: 模型输出
        model_type: "qwen3vl" 或 "guiowl"

    Returns:
        1.0 格式正确，0.0 格式错误
    """
    if model_type == "qwen3vl":
        return qwen3vl_format_reward(predict_str)
    elif model_type == "guiowl":
        return guiowl_format_reward(predict_str)
    else:
        # 未知类型，尝试两种格式
        qwen_score = qwen3vl_format_reward(predict_str)
        guiowl_score = guiowl_format_reward(predict_str)
        return max(qwen_score, guiowl_score)


def compute_score(
    reward_inputs: list[dict],
    debug: bool = False,
    action_type_weight: float = 0.5,
    action_params_weight: float = 0.5,
    system_button_mode: str = "default",
) -> list[dict]:
    """
    计算一批样本的得分 - EasyR1 框架要求的接口函数

    支持两种格式自动识别：
    - qwen3vl: <thinking>, <tool_call>, <conclusion>
    - guiowl: Action:, <tool_call>

    Args:
        reward_inputs: 包含多个样本的列表，每个样本包含:
            - response: 模型的响应
            - ground_truth: 正确答案 (JSON 字符串)
            - metadata: 可选，包含 conversations 用于模型类型检测
        debug: 是否输出调试信息
        action_type_weight: 动作类型分数权重 (默认 0.5)
        action_params_weight: 动作参数分数权重 (默认 0.5)
        system_button_mode: system_button 奖励计算模式（保留以兼容 r1gui_qwen3vl）

    Returns:
        每个样本的得分字典列表，包含:
            - overall: 总分
            - action_type: 动作类型匹配分数
            - action_params: 动作参数匹配分数
            - format: 格式分数（仅监控用）
            - model_type: 检测到的模型类型（用于调试）
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        metadata = reward_input.get("metadata", {})

        # 自动检测模型类型
        conversations = metadata.get("conversations", [])
        system_prompt = extract_system_prompt_from_conversation(conversations)
        model_type = detect_model_type(system_prompt)

        # 计算格式分数
        fmt_reward = format_reward(response, model_type)

        # 计算动作准确性分数
        action_type_reward, action_params_reward = compute_action_reward(
            response, ground_truth, model_type
        )

        # 组合总分
        # overall = action_type_weight * action_type_reward + action_params_weight * action_params_reward
        overall = action_type_weight * action_type_reward + action_params_weight * action_params_reward

        score = {
            "overall": overall,
            "format": fmt_reward,
            "action_type": action_type_reward,
            "action_params": action_params_reward,
            "model_type": model_type,  # 用于调试
            "action_type_weight": action_type_weight,
            "action_params_weight": action_params_weight,
        }

        if debug:
            print(f"Model type: {model_type}, Format: {fmt_reward}, "
                  f"Action type: {action_type_reward}, Params: {action_params_reward}")

        scores.append(score)

    return scores


def compute_action_reward(
    predict_str: str, ground_truth_str: str, model_type: str
) -> Tuple[float, float]:
    """
    计算动作准确性分数

    Args:
        predict_str: 模型输出
        ground_truth_str: 正确答案
        model_type: 模型类型

    Returns:
        (action_type_reward, action_params_reward)
    """
    # 提取预测的动作（两种格式通用）
    pred_action = extract_action(predict_str)
    gt_action = extract_ground_truth_action(ground_truth_str)

    # 动作类型匹配
    action_type_reward = 1.0 if pred_action == gt_action else 0.0

    # 动作参数匹配（根据动作类型计算）
    # 只有动作类型正确时才计算参数分。否则会出现 GT=swipe 但 pred=click 时，
    # swipe 参数函数仍给高分，导致整体 reward 被错误抬高的问题。
    if action_type_reward == 0.0:
        action_params_reward = 0.0
    else:
        action_params_reward = compute_params_reward(predict_str, ground_truth_str, gt_action)

    return action_type_reward, action_params_reward


def extract_action(content: str) -> str:
    """从模型输出中提取动作类型（已标准化）"""
    # 尝试 Qwen3VL 格式
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    match = tool_call_pattern.search(content)
    if match:
        try:
            tool_call = json.loads(match.group(1).strip())
            if "arguments" in tool_call:
                raw_action = tool_call["arguments"].get("action", "no action")
            else:
                raw_action = tool_call.get("action", "no action")
            return normalize_action_name(raw_action)
        except json.JSONDecodeError:
            pass
    return "no action"


def extract_ground_truth_action(ground_truth_str: str) -> str:
    """从 ground truth 提取动作类型（已标准化）"""
    try:
        gt_dict = json.loads(ground_truth_str)
        raw_action = gt_dict.get("action", "no action")
        return normalize_action_name(raw_action)
    except json.JSONDecodeError:
        return "no action"


def compute_params_reward(
    predict_str: str, ground_truth_str: str, action: str
) -> float:
    """
    计算动作参数匹配度

    Args:
        predict_str: 模型输出
        ground_truth_str: 正确答案
        action: 动作类型

    Returns:
        0.0-1.0 的匹配度分数
    """
    try:
        gt_dict = json.loads(ground_truth_str)
    except json.JSONDecodeError:
        return 0.0

    # 提取预测的参数
    pred_params = extract_action_params(predict_str)

    if action == "click":
        return compute_click_reward(pred_params, gt_dict)
    elif action == "long_press":
        return compute_long_press_reward(pred_params, gt_dict)
    elif action == "swipe":
        return compute_swipe_reward(pred_params, gt_dict)
    elif action == "type":
        return compute_type_reward(pred_params, gt_dict)
    elif action == "system_button":
        return compute_button_reward(pred_params, gt_dict)
    elif action == "terminate":
        return compute_terminate_reward(pred_params, gt_dict)
    elif action == "answer":
        return compute_answer_reward(pred_params, gt_dict)
    elif action == "wait":
        return 1.0  # wait 动作不需要参数验证
    elif action == "open":
        return compute_open_reward(pred_params, gt_dict)
    elif action == "key":
        return compute_key_reward(pred_params, gt_dict)
    else:
        return 0.0


def extract_action_params(content: str) -> dict:
    """从模型输出中提取动作参数"""
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    match = tool_call_pattern.search(content)
    if match:
        try:
            tool_call = json.loads(match.group(1).strip())
            if "arguments" in tool_call:
                return tool_call["arguments"]
            else:
                return {k: v for k, v in tool_call.items() if k != "name"}
        except json.JSONDecodeError:
            pass
    return {}


def compute_click_reward(pred: dict, gt: dict) -> float:
    """计算 click 动作的参数匹配度"""
    pred_coord = pred.get("coordinate", [])
    gt_bbox = gt.get("gt_bbox", [])

    if not pred_coord:
        return 0.0

    if isinstance(pred_coord[0], (list, tuple)):
        pred_coord = pred_coord[0]

    if len(pred_coord) < 2:
        return 0.0
    if len(gt_bbox) < 2:
        return 0.0

    try:
        x, y = float(pred_coord[0]), float(pred_coord[1])
    except (TypeError, ValueError):
        return 0.0

    # gt_bbox 有 4 个元素 [x1, y1, x2, y2]：使用 bbox 范围判断
    if len(gt_bbox) >= 4:
        x1, y1, x2, y2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]

        if x1 <= x <= x2 and y1 <= y <= y2:
            return 1.0

        gt_x, gt_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5

        max_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if max_dist == 0:
            max_dist = 100

        return max(0, 1 - dist / max_dist)

    # gt_bbox 只有 2 个元素 [x, y]（常见于 MobileForge 数据）：使用点到点距离
    gt_x, gt_y = gt_bbox[0], gt_bbox[1]
    dist = ((x - gt_x) ** 2 + (y - gt_y) ** 2) ** 0.5

    # MobileForge 数据中坐标范围是 0-1000，典型阈值取 50
    # 距离 < 10: reward=1.0, 距离 > 50: reward=0.0
    threshold = 50.0
    reward = max(0.0, 1.0 - dist / threshold)
    return reward


def compute_long_press_reward(pred: dict, gt: dict) -> float:
    """计算 long_press 动作的参数匹配度"""
    return compute_click_reward(pred, gt)  # long_press 和 click 一样验证坐标


def compute_swipe_reward(pred: dict, gt: dict) -> float:
    """
    计算 swipe 动作的参数匹配度

    GUIOwl/Qwen3VL 的 swipe 验证逻辑：
    - 如果 GT 有 direction：验证方向匹配
    - 如果 GT 没有 direction（常见于数据标注不完整）：只要模型执行了 swipe 就给 1.0
    """
    pred_coord = pred.get("coordinate", [])
    pred_coord2 = pred.get("coordinate2", [])
    pred_direction = pred.get("direction", "")

    gt_direction = gt.get("direction", "")

    # 如果 GT 有方向，验证方向匹配
    if gt_direction and pred_direction:
        if pred_direction.lower() == gt_direction.lower():
            return 1.0
        return 0.0

    # 如果 GT 没有方向信息（常见于数据标注不完整）：
    # 只要模型输出了有效的 swipe action（有坐标或方向）就给 1.0
    # 这符合 r1gui_qwen3vl.py 的处理逻辑
    has_valid_swipe = (
        (pred_coord and len(pred_coord) >= 2) or
        (pred_coord2 and len(pred_coord2) >= 2) or
        pred_direction
    )

    if has_valid_swipe:
        return 1.0

    return 0.5  # 默认部分匹配


def compute_type_reward(pred: dict, gt: dict) -> float:
    """计算 type 动作的参数匹配度"""
    pred_text = pred.get("text", "")
    gt_text = gt.get("input_text", "")

    if not pred_text or not gt_text:
        return 0.0

    # 完全匹配
    if pred_text.lower() == gt_text.lower():
        return 1.0

    # 部分匹配（F1 score）
    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())
    common = pred_tokens.intersection(gt_tokens)

    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = len(common) / len(pred_tokens)

    if len(gt_tokens) == 0:
        recall = 0.0
    else:
        recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_button_reward(pred: dict, gt: dict) -> float:
    """计算 system_button 动作的参数匹配度"""
    pred_button = pred.get("button", "")
    gt_button = gt.get("button", "")

    if not pred_button or not gt_button:
        return 0.0

    return 1.0 if pred_button.lower() == gt_button.lower() else 0.0


def compute_terminate_reward(pred: dict, gt: dict) -> float:
    """计算 terminate 动作的参数匹配度（status）"""
    pred_status = pred.get("status", "")
    gt_status = gt.get("status", "")

    if not gt_status:
        return 1.0  # 如果 GT 没有 status，默认通过

    return 1.0 if pred_status.lower() == gt_status.lower() else 0.0


def compute_answer_reward(pred: dict, gt: dict) -> float:
    """
    计算 answer 动作的参数匹配度（答案文本）

    answer 用于输出任务答案，需要验证文本内容。
    """
    pred_text = pred.get("text", "")
    gt_text = gt.get("input_text", "")

    if not gt_text:
        return 1.0  # 如果 GT 没有答案文本，默认通过

    if not pred_text:
        return 0.0

    # 完全匹配
    if pred_text.lower() == gt_text.lower():
        return 1.0

    # 部分匹配（F1 score）
    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())
    common = pred_tokens.intersection(gt_tokens)

    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = len(common) / len(pred_tokens)

    if len(gt_tokens) == 0:
        recall = 0.0
    else:
        recall = len(common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_open_reward(pred: dict, gt: dict) -> float:
    """
    计算 open 动作的参数匹配度（应用名称）

    open 用于打开指定应用，需要验证应用名称。
    """
    pred_app = pred.get("text", "").lower()
    gt_app = gt.get("input_text", "").lower()  # 训练数据中可能用 input_text 存储应用名

    if not gt_app:
        return 1.0  # 如果 GT 没有应用名，默认通过

    if not pred_app:
        return 0.0

    # 完全匹配
    if pred_app == gt_app:
        return 1.0

    # 部分匹配（应用名包含关系）
    if gt_app in pred_app or pred_app in gt_app:
        return 0.8

    return 0.0


def compute_key_reward(pred: dict, gt: dict) -> float:
    """
    计算 key 动作的参数匹配度（按键事件）

    key 用于执行 adb keyevent，如 volume_up, volume_down, power 等。
    """
    pred_key = pred.get("text", "").lower()
    gt_key = gt.get("input_text", "").lower()  # 训练数据中可能用 input_text 存储按键

    if not gt_key:
        return 1.0  # 如果 GT 没有按键信息，默认通过

    if not pred_key:
        return 0.0

    return 1.0 if pred_key == gt_key else 0.0


# 测试用例
if __name__ == "__main__":
    # 测试模型类型检测
    print("=== Model Type Detection ===")
    test_cases = [
        ("# Tools\n...", "guiowl"),
        ("You are a helpful assistant...", "qwen3vl"),
        ("Something else...", "unknown"),
    ]
    for prompt, expected in test_cases:
        result = detect_model_type(prompt)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{prompt[:30]}...' -> {result} (expected: {expected})")

    # 测试动作名称标准化
    print("\n=== Action Name Normalization ===")
    action_tests = [
        ("drag", "swipe"),
        ("input_text", "type"),
        ("click", "click"),
        ("type", "type"),
        ("swipe", "swipe"),
        ("terminate", "terminate"),
        ("open", "open"),
        ("key", "key"),
    ]
    for raw, expected in action_tests:
        result = normalize_action_name(raw)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{raw}' -> '{result}' (expected: '{expected}')")

    # 测试 GUIOwl 格式
    print("\n=== GUIOwl Format Test ===")
    guiowl_response = """Action: Click on the "Audio Recorder" app icon.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [384, 367]}}
</tool_call>"""

    guiowl_reward = guiowl_format_reward(guiowl_response)
    print(f"  GUIOwl format reward: {guiowl_reward}")

    # 测试 Qwen3VL 格式
    print("\n=== Qwen3VL Format Test ===")
    qwen3vl_response = """<thinking>I need to click on the button.</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [384, 367]}}
</tool_call>
<conclusion>Clicked on the button.</conclusion>"""

    qwen3vl_reward = qwen3vl_format_reward(qwen3vl_response)
    print(f"  Qwen3VL format reward: {qwen3vl_reward}")

    # 测试自适应格式奖励
    print("\n=== Adaptive Format Test ===")
    print(f"  GUIOwl with qwen3vl check: {format_reward(guiowl_response, 'qwen3vl')}")
    print(f"  GUIOwl with guiowl check: {format_reward(guiowl_response, 'guiowl')}")
    print(f"  Qwen3VL with qwen3vl check: {format_reward(qwen3vl_response, 'qwen3vl')}")
    print(f"  Qwen3VL with guiowl check: {format_reward(qwen3vl_response, 'guiowl')}")

    # 测试所有动作类型的 reward 计算
    print("\n=== All Action Types Reward Test ===")
    action_tests = [
        # (action, response, gt_dict, expected_overall)
        ("click", """Action: Click the button.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [384, 367]}}
</tool_call>""",
         {"action": "click", "gt_bbox": [350, 330, 420, 400]}, 1.0),

        ("swipe", """Action: Swipe up.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [322, 514], "coordinate2": [281, 233]}}
</tool_call>""",
         {"action": "drag"}, 1.0),  # drag -> swipe

        ("type", """Action: Type the text.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "type", "text": "hello world"}}
</tool_call>""",
         {"action": "input_text", "input_text": "hello world"}, 1.0),  # input_text -> type

        ("system_button", """Action: Press back.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}
</tool_call>""",
         {"action": "system_button", "button": "Back"}, 1.0),

        ("terminate", """Action: Terminate.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>""",
         {"action": "terminate", "status": "success"}, 1.0),

        ("wait", """Action: Wait.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "wait", "time": 2}}
</tool_call>""",
         {"action": "wait"}, 1.0),

        ("open", """Action: Open app.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "open", "text": "camera"}}
</tool_call>""",
         {"action": "open", "input_text": "camera"}, 1.0),

        ("key", """Action: Press volume up.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "key", "text": "volume_up"}}
</tool_call>""",
         {"action": "key", "input_text": "volume_up"}, 1.0),
    ]

    for action, response, gt_dict, expected in action_tests:
        test_input = {
            "response": response,
            "ground_truth": json.dumps(gt_dict),
            "metadata": {
                "conversations": [
                    {"role": "system", "content": "# Tools\nYou may call..."},
                    {"role": "user", "content": [{"type": "text", "text": "Task..."}]},
                ]
            }
        }
        scores = compute_score([test_input])
        score = scores[0]
        status = "✓" if abs(score['overall'] - expected) < 0.01 else "✗"
        print(f"  {status} {action}: overall={score['overall']} (expected: {expected})")

    # 测试格式错误的情况
    print("\n=== Format Error Test ===")
    error_response = "I think I should click the button."
    test_input = {
        "response": error_response,
        "ground_truth": json.dumps({"action": "click", "gt_bbox": [350, 330, 420, 400]}),
        "metadata": {
            "conversations": [
                {"role": "system", "content": "# Tools\nYou may call..."},
            ]
        }
    }
    scores = compute_score([test_input])
    print(f"  Format error response: format={scores[0]['format']} (expected: 0.0)")
