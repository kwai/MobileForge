# Copyright 2025 The android_world Authors.
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

"""GUI-Owl-1.5 for Android - Adapted for AndroidWorld framework."""

import base64
import copy
import io
import json
import os
import re
import time
from typing import Any

from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import mobile_agent_utils_new as mobile_agent_utils
from android_world.agents.coordinate_resize import update_image_size_
from android_world.env import interface
from android_world.env import json_action

# GUI-Owl 使用的虚拟分辨率（系统提示中声明）
GUI_OWL_VIRTUAL_WIDTH = 1000
GUI_OWL_VIRTUAL_HEIGHT = 1000


# ANSI颜色代码
class Colors:
    """终端颜色工具类"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def step(text: str) -> str:
        return f"{Colors.CYAN}{Colors.BOLD}🔹 {text}{Colors.RESET}"

    @staticmethod
    def important(text: str) -> str:
        return f"{Colors.MAGENTA}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}\n{text}\n{'=' * 60}{Colors.RESET}"


def pil_to_base64(image):
    """将 PIL 图像转换为 base64 字符串."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_base64_url(image, format="PNG"):
    """将 PIL 图像转换为 data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{image_base64}"


# GUI-Owl-1.5 System Prompt
GUI_OWL_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''


class GUIOwl15(base_agent.EnvironmentInteractingAgent):
    """GUI-Owl-1.5 Agent for Android using vision-language model."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        config: dict[str, Any],
        name: str = "GUIOwl15",
        last_image: int = 5,
    ):
        """Initializes a GUI-Owl-1.5 agent.

        Args:
          env: The environment.
          config: Same as Qwen3-VL: 'QWEN_BASE_URL', 'QWEN_API_KEY', 'QWEN_MODEL'
                  (from config.yaml).
          name: The agent name.
          last_image: Number of recent images to keep in context.
        """
        super().__init__(env, name)

        if not config.get("QWEN_BASE_URL"):
            raise ValueError("QWEN_BASE_URL is required in config")
        if not config.get("QWEN_API_KEY"):
            raise ValueError("QWEN_API_KEY is required in config")
        if not config.get("QWEN_MODEL"):
            raise ValueError("QWEN_MODEL is required in config")

        self.client = OpenAI(
            base_url=config["QWEN_BASE_URL"],
            api_key=config["QWEN_API_KEY"]
        )
        self.model_name = config["QWEN_MODEL"]
        self.last_image = last_image

        self._actions = []
        self._screenshots = []
        self.cur_user_messages = []
        self.action_history = []

    def reset(self, go_home_on_reset: bool = False):
        """Resets the agent."""
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._screenshots.clear()
        self.cur_user_messages.clear()
        self.action_history.clear()

    def _cut_current_messages(self, messages, last_image=None):
        """裁剪消息历史，只保留最近 N 个带图像的消息."""
        if last_image is None:
            last_image = self.last_image

        non_empty_user_indices = []
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and msg.get('content') and len(msg['content']) > 0:
                non_empty_user_indices.append(i)

        if len(non_empty_user_indices) > last_image:
            indices_to_clear = non_empty_user_indices[:-last_image]
        else:
            indices_to_clear = []

        for index in indices_to_clear:
            if index == 1:
                messages[index]['content'] = [messages[index]['content'][0]]
            else:
                messages[index]['content'] = []

        return messages

    def _convert_format(self, goal, messages):
        """转换消息格式用于模型输入."""
        new_messages = copy.deepcopy(messages[:1])
        history = []

        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and (msg["content"] == [] or (len(msg["content"]) == 1 and msg["content"][0]["type"] == "text")):
                history.append(messages[i+1]["content"][0]["text"].split("Action:")[-1].split("<tool_call>")[0].strip())
            if i != 1 and msg.get('role') == 'user' and msg["content"] != []:
                if len(history) == 0:
                    new_messages = copy.deepcopy(messages)
                    new_messages[1]["content"][0]["text"] = f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\nNo previous action."
                    return new_messages
                history_string = ""
                for j, h in enumerate(history):
                    history_string += f"Step{j+1}: {h}\n"
                history_string = history_string[:-1]
                new_messages.append(
                    {
                        "role": "user", "content": [
                            {"type": "text", "text": f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\n{history_string}"},
                            {"type": "image_url", "image_url": {"url": msg["content"][0]["image_url"]["url"]}}
                        ]
                    }
                )
                new_messages += copy.deepcopy(messages[i+1:])
                return new_messages

        return copy.deepcopy(messages)

    def _parse_action(self, action_response: str):
        """解析模型输出，提取动作."""
        try:
            dummy_action = action_response.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
            dummy_action = json.loads(dummy_action)
            dummy_action['arguments']['action'] = dummy_action['arguments']['action'].replace('tap', 'click')
            return dummy_action
        except (json.JSONDecodeError, IndexError) as e:
            print(Colors.error(f"Failed to parse action from response: {e}"))
            return None

    def _convert_coordinates(self, coords, screen_width, screen_height):
        """将 GUI-Owl 的 1000x1000 坐标转换为实际设备坐标."""
        if coords and len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
            x = int(coords[0] * screen_width / GUI_OWL_VIRTUAL_WIDTH)
            y = int(coords[1] * screen_height / GUI_OWL_VIRTUAL_HEIGHT)
            return x, y
        return None, None

    def _parse_and_convert_action(self, dummy_action, screen_width, screen_height):
        """解析并转换动作，将 1000x1000 坐标转换为实际设备坐标."""
        action_args = dummy_action.get('arguments', {})
        action_type = action_args.get('action')

        # click
        if action_type == 'click':
            coords = action_args.get('coordinate', [None, None])
            x, y = self._convert_coordinates(coords, screen_width, screen_height)
            if x is not None and y is not None:
                return json_action.JSONAction(action_type='click', x=x, y=y)
            return None

        # long_press
        elif action_type == 'long_press':
            coords = action_args.get('coordinate', [None, None])
            x, y = self._convert_coordinates(coords, screen_width, screen_height)
            if x is not None and y is not None:
                return json_action.JSONAction(action_type='long_press', x=x, y=y)
            return None

        # swipe with coordinates -> drag
        elif action_type == 'swipe':
            coords = action_args.get('coordinate', [None, None])
            coords2 = action_args.get('coordinate2', [None, None])
            if (len(coords) >= 2 and coords[0] is not None and coords2 is not None and
                len(coords2) >= 2 and coords2[0] is not None):
                x1, y1 = self._convert_coordinates(coords, screen_width, screen_height)
                x2, y2 = self._convert_coordinates(coords2, screen_width, screen_height)
                if x1 is not None and x2 is not None:
                    return json_action.JSONAction(
                        action_type='drag',
                        coordinate1=(x1, y1),
                        coordinate2=(x2, y2)
                    )

        # type
        elif action_type == 'type':
            text = action_args.get('text', '')
            if text:
                return json_action.JSONAction(action_type='input_text', text=text)

        # answer
        elif action_type == 'answer':
            text = action_args.get('text', '')
            return json_action.JSONAction(action_type='answer', text=text)

        # system_button
        elif action_type == 'system_button':
            button = action_args.get('button', '')
            if button == 'Back':
                return json_action.JSONAction(action_type='navigate_back')
            elif button == 'Home':
                return json_action.JSONAction(action_type='navigate_home')
            elif button == 'Menu':
                return json_action.JSONAction(action_type='navigate_menu')
            elif button == 'Enter':
                return json_action.JSONAction(action_type='keyboard_enter')

        # open
        elif action_type == 'open':
            app_name = action_args.get('text', '').lower()
            if app_name:
                return json_action.JSONAction(action_type='open_app', app_name=app_name)

        # wait
        elif action_type == 'wait':
            return json_action.JSONAction(action_type='wait')

        # terminate
        elif action_type == 'terminate':
            status = action_args.get('status', '')
            if status == 'success':
                return json_action.JSONAction(action_type='status', goal_status='complete')
            else:
                return json_action.JSONAction(action_type='status', goal_status='infeasible')

        print(Colors.error(f"Unknown or malformed action: {action_type}"))
        return None

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Performs a step of the agent on the environment.

        Args:
          goal: The goal/task description.

        Returns:
          AgentInteractionResult containing done status and step data.
        """
        step_num = len(self._screenshots)
        print(Colors.header(f"Step {step_num + 1}"))

        step_data = {
            "screenshot": None,
            "action_response": None,
            "action": None,
        }

        state = self.get_post_transition_state()
        step_data["screenshot"] = state.pixels.copy()
        screenshot = Image.fromarray(state.pixels)

        screenshot_url = pil_to_base64_url(screenshot)
        self._screenshots.append(screenshot)

        # 构建消息
        if step_num == 0:
            self.cur_user_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": GUI_OWL_SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {goal}\n\nPrevious actions:\nNo previous action."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_url}
                        }
                    ]
                }
            ]
        else:
            self.cur_user_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_url}
                        }
                    ]
                }
            )

        self.cur_user_messages = self._cut_current_messages(self.cur_user_messages, self.last_image)
        input_messages = self._convert_format(goal, self.cur_user_messages)

        # 调用模型
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=input_messages,
            )
            action_response = response.choices[0].message.content
        except Exception as e:
            print(Colors.error(f"Error calling model: {e}"))
            step_data["action_response"] = str(e)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.important("GUI-Owl output:"))
        print(Colors.info(f"{action_response}\n"))

        step_data["action_response"] = action_response

        self.cur_user_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_response}]
            }
        )

        # 解析动作
        dummy_action = self._parse_action(action_response)
        if dummy_action is None:
            print(Colors.error("Failed to parse action"))
            return base_agent.AgentInteractionResult(False, step_data)

        # 处理 answer action
        if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
            dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
            self.env.interaction_cache = self._actions[-1]['arguments']['text']

        self._actions.append(dummy_action)
        self.action_history.append(dummy_action['arguments'].get('action', 'unknown'))

        # 获取屏幕尺寸用于坐标转换
        screen_size = self.env.device_screen_size

        # 解析并转换动作（从 1000x1000 到实际设备坐标）
        try:
            action = self._parse_and_convert_action(dummy_action, screen_size[0], screen_size[1])
        except Exception as e:
            print(Colors.error(f"Failed to convert action: {e}"))
            return base_agent.AgentInteractionResult(False, step_data)

        if action is None:
            print(Colors.error("Failed to parse action"))
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.success(f"Parsed action: {action.action_type}"))
        if hasattr(action, 'x') and action.x is not None:
            print(Colors.info(f"  Coordinates: ({action.x}, {action.y})"))
        if hasattr(action, 'coordinate1') and action.coordinate1 is not None:
            print(Colors.info(f"  Drag from {action.coordinate1} to {action.coordinate2}"))

        step_data["action"] = action

        # 检查是否完成
        if action.action_type == json_action.STATUS:
            print(Colors.success("Task completed with status"))
            return base_agent.AgentInteractionResult(True, step_data)

        if action.action_type == json_action.ANSWER:
            print(Colors.success(f"Answer: {action.text}"))
            try:
                self.env.execute_action(action)
            except Exception as e:
                print(Colors.error(f"Error executing answer action: {e}"))
            return base_agent.AgentInteractionResult(True, step_data)

        # 执行动作
        try:
            self.env.execute_action(action)
            print(Colors.success(f"Executed action: {action.action_type}"))
            time.sleep(2.0)
        except Exception as e:
            print(Colors.error(f"Error executing action: {e}"))
            return base_agent.AgentInteractionResult(False, step_data)

        return base_agent.AgentInteractionResult(False, step_data)
