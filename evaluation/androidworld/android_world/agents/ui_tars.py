import ast
import base64
import io
import re
import time
import os
import math
from typing import Dict, List
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface, json_action


MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space
click(start_box='(x1,y1)')
long_press(start_box='(x1,y1)')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='(x1,y1)', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
wait()
press_home()
press_back()
press_enter()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""


# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode="eval")

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def parse_action_qwen2vl(text, factor, image_height, image_width):
    text = text.strip()
    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        all_action.append(action_str)

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip()) for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            continue
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                float_numbers = [float(num) / factor for num in numbers]
                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0],
                        float_numbers[1],
                        float_numbers[0],
                        float_numbers[1],
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append(
            {
                "reflection": reflection,
                "thought": thought,
                "action_type": action_type,
                "action_inputs": action_inputs,
                "text": text,
            }
        )
    return actions


def convert_action(action_str, param_str, image_bounds, min_pixels, max_pixels) -> dict:
    """Convert UI-TARS action format to AndroidWorld json_action format"""

    # Get image dimensions
    image_width = image_bounds[2] - image_bounds[0]
    image_height = image_bounds[3] - image_bounds[1]

    # Parse using the existing UI-TARS parsing logic
    full_action_str = f"Action: {action_str}({param_str})"

    try:
        # Use the existing parse_action_qwen2vl function
        parsed_actions = parse_action_qwen2vl(
            text=full_action_str,
            factor=1000,  # Same factor as original UI-TARS
            image_height=image_height,
            image_width=image_width,
        )

        if not parsed_actions:
            raise ValueError("No actions parsed")

        # Get the first parsed action
        parsed_action = parsed_actions[0]
        action_type = parsed_action.get("action_type")
        action_inputs = parsed_action.get("action_inputs", {})

    except Exception as e:
        print(f"Failed to parse action with UI-TARS parser: {e}")
        # Fallback to simpler parsing
        action_type = action_str
        action_inputs = {}

        # Try to parse basic parameters
        if "start_box=" in param_str:
            start_match = re.search(r"start_box='([^']*)'", param_str)
            if start_match:
                action_inputs["start_box"] = start_match.group(1)

        if "end_box=" in param_str:
            end_match = re.search(r"end_box='([^']*)'", param_str)
            if end_match:
                action_inputs["end_box"] = end_match.group(1)

        if "content=" in param_str:
            content_match = re.search(r"content='([^']*)'", param_str)
            if content_match:
                action_inputs["content"] = content_match.group(1)

        if "direction=" in param_str:
            direction_match = re.search(r"direction='([^']*)'", param_str)
            if direction_match:
                action_inputs["direction"] = direction_match.group(1)

        if "key=" in param_str:
            key_match = re.search(r"key='([^']*)'", param_str)
            if key_match:
                action_inputs["key"] = key_match.group(1)

    # Helper function to parse coordinate string to list
    def parse_coordinate_string(coord_str):
        if isinstance(coord_str, str):
            try:
                # Handle both formats: "[x1, y1, x2, y2]" and "(x1,y1,x2,y2)"
                coord_str = coord_str.strip()
                if coord_str.startswith("[") and coord_str.endswith("]"):
                    return ast.literal_eval(coord_str)
                elif coord_str.startswith("(") and coord_str.endswith(")"):
                    # Convert (x1,y1,x2,y2) to [x1,y1,x2,y2]
                    numbers = coord_str[1:-1].split(",")
                    return [float(x.strip()) for x in numbers]
                else:
                    return ast.literal_eval(coord_str)
            except:
                return None
        return coord_str

    # Convert to the expected return format based on action type
    if action_type in ["click", "long_press"]:
        start_box_str = action_inputs.get("start_box", "")
        start_box = parse_coordinate_string(start_box_str)

        if start_box and len(start_box) >= 2:
            # start_box contains normalized coordinates, convert to pixel coordinates
            if len(start_box) >= 4:
                # Use center of box
                x = int((start_box[0] + start_box[2]) / 2 * image_width)
                y = int((start_box[1] + start_box[3]) / 2 * image_height)
            else:
                # Direct coordinates
                x = int(start_box[0] * image_width)
                y = int(start_box[1] * image_height)
        else:
            # Fallback to original parsing if start_box parsing fails
            pattern = r"(\d+)\s+(\d+)"
            match = re.search(pattern, param_str)
            if not match:
                raise ValueError(f"Cannot parse coordinates from {param_str}")
            x, y = map(int, match.groups())

        ret = {"action_type": action_type, "x": x, "y": y}

    elif action_type == "scroll":
            # If end_box is specified, treat as a drag/swipe from start to end.
            if "end_box" in action_inputs:
                start_box_str = action_inputs.get("start_box", "")
                end_box_str = action_inputs.get("end_box", "")

                start_box = parse_coordinate_string(start_box_str)
                end_box = parse_coordinate_string(end_box_str)

                if start_box and end_box and len(start_box) >= 2 and len(end_box) >= 2:
                    # Coordinates are normalized, convert to pixel coordinates
                    if len(start_box) >= 4:
                        x1 = int((start_box[0] + start_box[2]) / 2 * image_width)
                        y1 = int((start_box[1] + start_box[3]) / 2 * image_height)
                    else:
                        x1 = int(start_box[0] * image_width)
                        y1 = int(start_box[1] * image_height)

                    if len(end_box) >= 4:
                        x2 = int((end_box[0] + end_box[2]) / 2 * image_width)
                        y2 = int((end_box[1] + end_box[3]) / 2 * image_height)
                    else:
                        x2 = int(end_box[0] * image_width)
                        y2 = int(end_box[1] * image_height)

                    # AndroidWorld expects a 'drag' for coordinate-based scrolls.
                    ret = {
                        "action_type": "drag",
                        "coordinate1": [x1, y1],
                        "coordinate2": [x2, y2],
                    }
                else:
                    # Fallback to original parsing if coordinate parsing fails
                    pattern = r"start_box='[^']*(\d+)[,\s]+(\d+)[^']*', end_box='[^']*(\d+)[,\s]+(\d+)[^']*'"
                    match = re.search(pattern, param_str)
                    if not match:
                        raise ValueError(f"Cannot parse scroll coordinates from: {param_str}")
                    x1, y1, x2, y2 = map(int, match.groups())
                    ret = {
                        "action_type": "drag",
                        "coordinate1": [x1, y1],
                        "coordinate2": [x2, y2],
                    }
            else:
                # Original directional scroll logic for stable behavior
                direction = action_inputs.get("direction", "")
                point_str = action_inputs.get("start_box", "")

                if not direction:
                    pattern = r"direction='(.*)'"
                    match = re.search(pattern, param_str, re.DOTALL)
                    if match:
                        direction = match.group(1).strip()

                if point_str:
                    point_coords = parse_coordinate_string(point_str)
                    if point_coords and len(point_coords) >= 2:
                        if len(point_coords) >= 4:
                            # Use center of box
                            x = int(
                                (point_coords[0] + point_coords[2]) / 2 * image_width
                            )
                            y = int(
                                (point_coords[1] + point_coords[3]) / 2 * image_height
                            )
                        else:
                            x = int(point_coords[0] * image_width)
                            y = int(point_coords[1] * image_height)
                        ret = {
                            "action_type": "scroll",
                            "x": x,
                            "y": y,
                            "direction": direction,
                        }
                    else:
                        ret = {"action_type": "scroll", "direction": direction}
                else:
                    ret = {"action_type": "scroll", "direction": direction}

    elif action_type in ["drag", "swipe"]:
        start_box_str = action_inputs.get("start_box", "")
        end_box_str = action_inputs.get("end_box", "")

        start_box = parse_coordinate_string(start_box_str)
        end_box = parse_coordinate_string(end_box_str)

        if start_box and end_box and len(start_box) >= 2 and len(end_box) >= 2:
            # Coordinates are normalized, convert to pixel coordinates
            if len(start_box) >= 4:
                x1 = int((start_box[0] + start_box[2]) / 2 * image_width)
                y1 = int((start_box[1] + start_box[3]) / 2 * image_height)
            else:
                x1 = int(start_box[0] * image_width)
                y1 = int(start_box[1] * image_height)

            if len(end_box) >= 4:
                x2 = int((end_box[0] + end_box[2]) / 2 * image_width)
                y2 = int((end_box[1] + end_box[3]) / 2 * image_height)
            else:
                x2 = int(end_box[0] * image_width)
                y2 = int(end_box[1] * image_height)
        else:
            # Fallback to original parsing if coordinate parsing fails
            pattern = r"start_box='[^']*(\d+)[,\s]+(\d+)[^']*', end_box='[^']*(\d+)[,\s]+(\d+)[^']*'"
            match = re.search(pattern, param_str)
            if not match:
                raise ValueError(f"Cannot parse drag coordinates from: {param_str}")
            x1, y1, x2, y2 = map(int, match.groups())

        ret = {"action_type": "drag", "coordinate1": [x1, y1], "coordinate2": [x2, y2]}

    elif action_type == "type":
        text = action_inputs.get("content", "")
        if not text:
            pattern = r"content='(.*)'"
            match = re.search(pattern, param_str, re.DOTALL)
            if match:
                text = match.group(1).strip()
        ret = {"action_type": "input_text", "text": text}

    elif action_type == "press_home":
        ret = {"action_type": "navigate_home"}
    elif action_type == "press_back":
        ret = {"action_type": "navigate_back"}
    elif action_type == "open_app":
        app_name = action_inputs.get("app_name", "")
        if not app_name:
            pattern = r"app_name=\'(.*)\'"
            match = re.search(pattern, param_str, re.DOTALL)
            if match:
                app_name = match.group(1).strip()
        ret = {"action_type": "open_app", "app_name": app_name}

    elif action_type == "finished":
        status = action_inputs.get("content", "")
        if not status:
            pattern = r"content='(.*)'"
            match = re.search(pattern, param_str, re.DOTALL)
            if match:
                status = match.group(1).strip()
        if not status:  # status为空
          ret = {"action_type": "status", "goal_status": ""}
        else:  # status不为空
            ret = {"action_type": "answer", "text": status}

    else:
        ret = {"action_type": "unknown"}

    return ret


class UITARS(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "UI-TARS",
        wait_after_action_seconds: float = 2.0,
        config: dict = None,
    ):
        super().__init__(env, name)

        # Read configuration from config parameter - config is required
        if not config:
            raise ValueError(
                "Config parameter is required for UITARS agent. Please provide UITARS_BASE_URL, UITARS_API_KEY, and UITARS_MODEL in config."
            )

        # Get required configuration parameters - no defaults provided
        if "UITARS_BASE_URL" not in config:
            raise ValueError("UITARS_BASE_URL must be specified in config")
        if "UITARS_API_KEY" not in config:
            raise ValueError("UITARS_API_KEY must be specified in config")
        if "UITARS_MODEL" not in config:
            raise ValueError("UITARS_MODEL must be specified in config")

        base_url = config["UITARS_BASE_URL"]
        api_key = config["UITARS_API_KEY"]
        self.model_name = config["UITARS_MODEL"]

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Agent state
        self.history = []
        self.step_idx = 1
        self.message_history = []
        self.wait_after_action_seconds = wait_after_action_seconds

        # UI-TARS specific attributes
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

        # Enhanced logging for model interactions
        self.detailed_model_logs = []  # Store complete input/output for each model call
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Image processing parameters
        self.max_pixels = 1280 * 28 * 28
        self.min_pixels = 100 * 28 * 28
        # Read history_n from config, default to 1 if not specified
        self.history_n = config["UITARS_HISTORY_N"]

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []
        self.step_idx = 1
        self.message_history = []

        # Reset UI-TARS specific state
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []

        # Reset enhanced logging
        self.detailed_model_logs = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def pil_to_base64(self, image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _sanitize_messages_for_logging(self, messages):
        """
        Keep original messages for logging without any sanitization.
        This preserves the complete base64 image data for debugging purposes.
        """
        # Return the original messages without any modification
        return messages

    def get_enhanced_log_data(self):
        """
        Get enhanced logging data including detailed model interactions.
        This method should be called when saving execution results.
        """
        return {
            "detailed_model_logs": self.detailed_model_logs,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_model_calls": len(self.detailed_model_logs),
        }

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "step_idx": self.step_idx,
            "raw_screenshot": None,
            "raw_response": None,
            "raw_action": None,
            "judgement": None,
            "action_output": None,
            "action_raw_response": None,
            "summary": None,
            "summary_raw_response": None,
            "actual_action_coordinates": None,
        }

        print("----------step " + str(len(self.history) + 1))

        state = self.get_post_transition_state()
        step_data["raw_screenshot"] = state.pixels.copy()
        before_screenshot = state.pixels.copy()

        # Convert screenshot to PIL Image
        before_screenshot = Image.fromarray(before_screenshot, "RGB")

        image_bounds = [0, 0, before_screenshot.width, before_screenshot.height]

        # Resize image if needed
        if before_screenshot.width * before_screenshot.height > self.max_pixels:
            resize_factor = math.sqrt(
                self.max_pixels / (before_screenshot.width * before_screenshot.height)
            )
            width, height = (
                int(before_screenshot.width * resize_factor),
                int(before_screenshot.height * resize_factor),
            )
            before_screenshot = before_screenshot.resize((width, height))
        if before_screenshot.width * before_screenshot.height < self.min_pixels:
            resize_factor = math.sqrt(
                self.min_pixels / (before_screenshot.width * before_screenshot.height)
            )
            width, height = (
                math.ceil(before_screenshot.width * resize_factor),
                math.ceil(before_screenshot.height * resize_factor),
            )
            before_screenshot = before_screenshot.resize((width, height))

        if before_screenshot.mode != "RGB":
            before_screenshot = before_screenshot.convert("RGB")

        # Store image in history
        self.history_images.append(before_screenshot)
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n :]

        # Convert to base64
        screenshot_base64 = self.pil_to_base64(before_screenshot)
        new_image_bounds = [0, 0, before_screenshot.width, before_screenshot.height]

        # Prepare user prompt
        sys_prompt = MOBILE_USE_DOUBAO

        # Build current user messages for this step
        if self.step_idx == 1:
            current_user_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": sys_prompt + goal,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            },
                        },
                    ],
                }
            ]
        else:
            current_user_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            },
                        }
                    ],
                }
            ]

        # Build messages for API call (similar to ui_tars_1_5.py logic)
        messages_for_api = list(self.message_history) + list(current_user_messages)

        # 1) Find initial user message with text in full message list
        initial_user_with_text = None
        for msg in messages_for_api:
            if msg.get("role") != "user":
                continue
            c = msg.get("content")
            if isinstance(c, str) and c:
                initial_user_with_text = msg
                break
            if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "text" for p in c
            ):
                initial_user_with_text = msg
                break

        # Limit to last 5 images to avoid context length issues
        def _has_image(m):
            c = m.get("content")
            return isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c
            )

        # 2) Collect indices of all user messages with images
        image_indices = [
            i
            for i, m in enumerate(messages_for_api)
            if m.get("role") == "user" and _has_image(m)
        ]
        # Keep only the last self.history_n
        keep_image_indices = set(image_indices[-(self.history_n) :])

        # 3) Reconstruct filtered_messages with rules:
        #    - keep all assistant messages
        #    - keep initial user-with-text always, but strip image if its index not in keep set
        #    - keep other user messages only if their index in keep set
        filtered_messages = []
        for idx, message in enumerate(messages_for_api):
            role = message.get("role")
            if role == "assistant":
                filtered_messages.append(message)
                continue
            if role != "user":
                continue

            content = message.get("content")
            if message is initial_user_with_text:
                # include initial; strip image if needed
                if idx in keep_image_indices:
                    filtered_messages.append(message)
                else:
                    # Make a shallow copy with image parts removed
                    if isinstance(content, list):
                        filtered = [
                            part
                            for part in content
                            if not (
                                isinstance(part, dict)
                                and part.get("type") == "image_url"
                            )
                        ]
                        filtered_messages.append(
                            {
                                "role": "user",
                                "content": filtered if filtered else content,
                            }
                        )
                    else:
                        filtered_messages.append(message)
            else:
                # other user messages: include only if their images are selected
                if idx in keep_image_indices:
                    filtered_messages.append(message)

        # Retry logic for action parsing failures
        max_retries = 100
        converted_action = None

        for retry_count in range(max_retries):
            # Record the API call details for enhanced logging
            api_call_start_time = time.time()
            model_log_entry = {
                "step": self.step_idx,
                "retry_count": retry_count + 1,
                "timestamp": api_call_start_time,
                "input_messages": self._sanitize_messages_for_logging(
                    filtered_messages
                ),
                "model": self.model_name,
                "raw_response": None,
                "parsed_action": None,
                "success": False,
                "error": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_call_duration": 0.0,
            }

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=filtered_messages,
                )

                api_call_end_time = time.time()
                model_log_entry["api_call_duration"] = (
                    api_call_end_time - api_call_start_time
                )

                raw_response = response.choices[0].message.content
                model_log_entry["raw_response"] = raw_response

                # Extract token usage information
                if hasattr(response, "usage") and response.usage:
                    model_log_entry["prompt_tokens"] = getattr(
                        response.usage, "prompt_tokens", 0
                    )
                    model_log_entry["completion_tokens"] = getattr(
                        response.usage, "completion_tokens", 0
                    )
                    model_log_entry["total_tokens"] = getattr(
                        response.usage, "total_tokens", 0
                    )

                    # Update total counters
                    self.total_prompt_tokens += model_log_entry["prompt_tokens"]
                    self.total_completion_tokens += model_log_entry["completion_tokens"]

                # Parse action from response
                action_pattern = r"Action: (.*)"
                action_match = re.search(action_pattern, raw_response, re.DOTALL)
                if not action_match:
                    if retry_count < max_retries - 1:
                        print(
                            f"No action found in response, retrying... ({retry_count + 1}/{max_retries})"
                        )
                        continue
                    else:
                        print("No action found in the response after all retries.")
                        step_data["summary"] = (
                            "No action found in the model's response."
                        )
                        self.history.append(step_data)
                        return base_agent.AgentInteractionResult(False, step_data)

                action_raw_str = action_match.group(1).strip()

                # Parse action string and parameters
                first_bracket_idx = None
                for i, c in enumerate(action_raw_str):
                    if c == "(":
                        first_bracket_idx = i
                        break

                if first_bracket_idx is None:
                    if retry_count < max_retries - 1:
                        print(
                            f"Failed to find opening bracket in action, retrying... ({retry_count + 1}/{max_retries})"
                        )
                        continue
                    else:
                        print(
                            "Failed to find opening bracket in action after all retries."
                        )
                        step_data["summary"] = (
                            "Invalid action format - no opening bracket found."
                        )
                        self.history.append(step_data)
                        return base_agent.AgentInteractionResult(False, step_data)

                action_str = action_raw_str[:first_bracket_idx]
                param_str = action_raw_str[first_bracket_idx + 1 : -1]

                try:
                    converted_action = json_action.JSONAction(
                        **convert_action(
                            action_str,
                            param_str,
                            image_bounds,
                            self.min_pixels,
                            self.max_pixels,
                        )
                    )
                    print(f"converted_action: {converted_action}")
                    step_data["action_output_json"] = converted_action

                    # Success - update message history and break retry loop
                    step_data["raw_response"] = raw_response
                    step_data["raw_action"] = action_raw_str
                    step_data["action_raw_response"] = raw_response
                    step_data["action_output"] = action_raw_str
                    print(raw_response)
                    print("raw_action: " + action_raw_str)

                    current_assistant_message = {
                        "role": "assistant",
                        "content": raw_response,
                    }
                    self.message_history.extend(current_user_messages)
                    self.message_history.append(current_assistant_message)
                    break

                except Exception as e:
                    model_log_entry["error"] = str(e)
                    model_log_entry["parsed_action"] = None
                    if retry_count < max_retries - 1:
                        print(
                            f"Failed to convert action, retrying... ({retry_count + 1}/{max_retries}): {e}"
                        )
                        # Still save this failed attempt to model logs
                        self.detailed_model_logs.append(model_log_entry)
                        continue
                    else:
                        print(
                            "Failed to convert the output to a valid action after all retries."
                        )
                        print(str(e))
                        # Save the final failed attempt
                        self.detailed_model_logs.append(model_log_entry)
                        step_data["summary"] = (
                            "Can not parse the output to a valid action. Please make sure to pick"
                            " the action from the list with required parameters (if any) in the"
                            " correct JSON format!"
                        )
                        self.history.append(step_data)
                        return base_agent.AgentInteractionResult(False, step_data)

            except Exception as e:
                model_log_entry["error"] = str(e)
                if retry_count < max_retries - 1:
                    print(
                        f"Error when fetching response from client, retrying... ({retry_count + 1}/{max_retries}): {e}"
                    )
                    # Still save this failed attempt to model logs
                    self.detailed_model_logs.append(model_log_entry)
                    continue
                else:
                    print(f"Client error after all retries: {e}")
                    # Save the final failed attempt
                    self.detailed_model_logs.append(model_log_entry)
                    step_data["summary"] = f"Client error: {e}"
                    self.history.append(step_data)
                    return base_agent.AgentInteractionResult(False, step_data)

            finally:
                # Always record this model call attempt
                if converted_action is not None:
                    model_log_entry["success"] = True
                    model_log_entry["parsed_action"] = {
                        "action_type": converted_action.action_type,
                        "action_data": converted_action.__dict__,
                    }

                # Save the model log entry for this attempt
                self.detailed_model_logs.append(model_log_entry)

        # If we reach here without converted_action, something went wrong
        if converted_action is None:
            step_data["summary"] = "Failed to generate valid action after all retries."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # Handle completion status
        if converted_action.action_type == "status":
            if "infeasible" in converted_action.goal_status:
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)
        
        if converted_action.action_type == 'answer':
            print('Agent answered with:', converted_action.text)

        # Execute action
        try:
            actual_action_coordinates = self.env.execute_action(converted_action)

            if converted_action.action_type == 'answer':
                converted_action.action_type = 'status'
                step_data["summary"] = "Agent thinks the request has been completed."
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(True, step_data)
            
            step_data["actual_action_coordinates"] = actual_action_coordinates
        except Exception as e:
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            return base_agent.AgentInteractionResult(False, step_data)

        time.sleep(self.wait_after_action_seconds)

        # Get after-action state
        state = self.env.get_state(wait_to_stabilize=False)
        after_screenshot = state.pixels.copy()
        step_data["after_screenshot"] = after_screenshot.copy()

        self.history.append(step_data)
        self.step_idx += 1

        return base_agent.AgentInteractionResult(False, step_data)