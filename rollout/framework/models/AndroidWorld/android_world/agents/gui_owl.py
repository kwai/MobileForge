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

"""GUI-Owl-1.5 Agent for Android - Adapted for MobileForge framework."""

import base64
import copy
import io
import json
import re
import time
from typing import Any

from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface
from android_world.env import json_action


# GUI-Owl uses 1000x1000 virtual resolution (declared in system prompt)
GUI_OWL_VIRTUAL_WIDTH = 1000
GUI_OWL_VIRTUAL_HEIGHT = 1000


# ANSI color codes for terminal output
class Colors:
    """Terminal color utility class."""

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


def pil_to_base64_url(image, format="PNG"):
    """Convert PIL Image to data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{image_base64}"


# GUI-Owl-1.5 System Prompt
GUI_OWL_SYSTEM_PROMPT = """# Tools

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
    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button\"", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
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
- If finishing, use action=terminate in the tool call."""


class GUIOwl15(base_agent.EnvironmentInteractingAgent):
    """GUI-Owl-1.5 Agent for Android using vision-language model.

    GUI-Owl-1.5 uses a different system prompt and message format compared to Qwen3VL.
    It keeps only the last N images in context and formats messages differently.
    """

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
          config: Configuration dictionary containing 'QWEN_BASE_URL', 'QWEN_API_KEY',
                  and 'QWEN_MODEL' (same keys as Qwen3VL).
          name: The agent name.
          last_image: Number of recent images to keep in context (default: 5).
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

        # Enhanced logging (consistent with Qwen3VL)
        self.detailed_model_logs = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset(self, go_home_on_reset: bool = False):
        """Resets the agent."""
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._screenshots.clear()
        self.cur_user_messages.clear()
        self.action_history.clear()

        # Reset enhanced logging
        self.detailed_model_logs = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def get_enhanced_log_data(self):
        """Get enhanced logging data including detailed model interactions."""
        return {
            "detailed_model_logs": self.detailed_model_logs,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_model_calls": len(self.detailed_model_logs),
        }

    def _cut_current_messages(self, messages, last_image=None):
        """Trim message history, keeping only the last N images.

        Args:
          messages: List of message dictionaries.
          last_image: Number of recent images to keep.

        Returns:
          Trimmed messages list.
        """
        if last_image is None:
            last_image = self.last_image

        # Find indices of user messages with actual content (images)
        non_empty_user_indices = []
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and msg.get('content') and len(msg['content']) > 0:
                non_empty_user_indices.append(i)

        # Keep only the last `last_image` messages with images
        if len(non_empty_user_indices) > last_image:
            indices_to_clear = non_empty_user_indices[:-last_image]
        else:
            indices_to_clear = []

        for index in indices_to_clear:
            if index == 1:
                # Keep system prompt, clear content
                messages[index]['content'] = [messages[index]['content'][0]]
            else:
                messages[index]['content'] = []

        return messages

    def _convert_format(self, goal, messages):
        """Convert message format for model input.

        GUI-Owl uses a different format from Qwen3VL:
        - Previous actions are summarized in the user prompt
        - Only keeps recent images in context

        Args:
          goal: The task goal.
          messages: Current message history.

        Returns:
          Converted messages for API call.
        """
        new_messages = copy.deepcopy(messages[:1])  # Keep system message
        history = []

        # Extract action history from assistant responses
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user' and (
                msg["content"] == [] or
                (len(msg["content"]) == 1 and msg["content"][0]["type"] == "text")
            ):
                # This is a text-only user message (after first step)
                if i + 1 < len(messages):
                    assistant_text = messages[i + 1]["content"][0]["text"]
                    action = assistant_text.split("Action:")[-1].split("<tool_call>")[0].strip()
                    history.append(action)

        # Check if this is the first step (index == 1)
        first_user_msg = messages[1] if len(messages) > 1 else None
        if first_user_msg and first_user_msg.get('role') == 'user':
            if first_user_msg["content"] == [] or (len(first_user_msg["content"]) == 1 and first_user_msg["content"][0]["type"] == "text"):
                if len(history) == 0:
                    # First step with no history
                    new_messages = copy.deepcopy(messages)
                    new_messages[1]["content"][0]["text"] = (
                        f"Please generate the next move according to the UI screenshot, "
                        f"instruction and previous actions.\n\n"
                        f"Instruction: {goal}\n\n"
                        f"Previous actions:\nNo previous action."
                    )
                    return new_messages

        # Build history string
        if history:
            history_string = ""
            for j, h in enumerate(history):
                history_string += f"Step{j+1}: {h}\n"
            history_string = history_string[:-1]  # Remove trailing newline

            # Find the last user message with image
            for i, msg in enumerate(messages):
                if i != 1 and msg.get('role') == 'user' and msg["content"] != []:
                    new_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Please generate the next move according to the UI screenshot, "
                                    f"instruction and previous actions.\n\n"
                                    f"Instruction: {goal}\n\n"
                                    f"Previous actions:\n{history_string}"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": msg["content"][0]["image_url"]["url"]}
                            }
                        ]
                    })
                    new_messages += copy.deepcopy(messages[i+1:])
                    return new_messages

        return copy.deepcopy(messages)

    def _parse_action(self, action_response: str):
        """Parse model output to extract action.

        Args:
          action_response: Raw response from the model.

        Returns:
          Parsed action dictionary or None if parsing fails.
        """
        try:
            dummy_action = action_response.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
            dummy_action = json.loads(dummy_action)
            # Normalize action names
            action_name = dummy_action['arguments']['action']
            if action_name == 'tap':
                action_name = 'click'
            dummy_action['arguments']['action'] = action_name
            return dummy_action
        except (json.JSONDecodeError, IndexError) as e:
            print(Colors.error(f"Failed to parse action from response: {e}"))
            return None

    def _convert_coordinates(self, coords, screen_width, screen_height):
        """Convert GUI-Owl's 1000x1000 coordinates to actual device coordinates.

        Args:
          coords: Coordinates in 1000x1000 virtual space.
          screen_width: Actual screen width.
          screen_height: Actual screen height.

        Returns:
          Tuple of (x, y) in actual device coordinates, or (None, None) if invalid.
        """
        if coords and len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
            x = int(coords[0] * screen_width / GUI_OWL_VIRTUAL_WIDTH)
            y = int(coords[1] * screen_height / GUI_OWL_VIRTUAL_HEIGHT)
            return x, y
        return None, None

    def _parse_and_convert_action(self, dummy_action, screen_width, screen_height):
        """Parse and convert action from GUI-Owl format to JSONAction.

        Args:
          dummy_action: Action dictionary from model.
          screen_width: Actual screen width.
          screen_height: Actual screen height.

        Returns:
          JSONAction object or None if parsing fails.
        """
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

        # Build messages
        if step_num == 0:
            # First step: include system prompt and initial user message
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
                            "text": (
                                f"Please generate the next move according to the UI screenshot, "
                                f"instruction and previous actions.\n\n"
                                f"Instruction: {goal}\n\n"
                                f"Previous actions:\nNo previous action."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_url}
                        }
                    ]
                }
            ]
        else:
            # Subsequent steps: append screenshot
            self.cur_user_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_url}
                    }
                ]
            })

        # Trim message history
        self.cur_user_messages = self._cut_current_messages(self.cur_user_messages, self.last_image)
        input_messages = self._convert_format(goal, self.cur_user_messages)

        # Prepare log entry
        api_call_start_time = time.time()
        model_log_entry = {
            "step": step_num + 1,
            "timestamp": api_call_start_time,
            "input_messages": input_messages,
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

        # Call model
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=input_messages,
            )
            action_response = response.choices[0].message.content

            api_call_end_time = time.time()
            model_log_entry["api_call_duration"] = api_call_end_time - api_call_start_time
            model_log_entry["raw_response"] = action_response

            # Extract token usage
            if hasattr(response, "usage") and response.usage:
                model_log_entry["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0)
                model_log_entry["completion_tokens"] = getattr(response.usage, "completion_tokens", 0)
                model_log_entry["total_tokens"] = getattr(response.usage, "total_tokens", 0)
                self.total_prompt_tokens += model_log_entry["prompt_tokens"]
                self.total_completion_tokens += model_log_entry["completion_tokens"]

        except Exception as e:
            print(Colors.error(f"Error calling model: {e}"))
            model_log_entry["error"] = str(e)
            model_log_entry["api_call_duration"] = time.time() - api_call_start_time
            self.detailed_model_logs.append(model_log_entry)
            step_data["action_response"] = str(e)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.important("GUI-Owl output:"))
        print(Colors.info(f"{action_response}\n"))

        step_data["action_response"] = action_response

        # Add assistant response to history
        self.cur_user_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": action_response}]
        })

        # Parse action
        dummy_action = self._parse_action(action_response)
        if dummy_action is None:
            print(Colors.error("Failed to parse action"))
            model_log_entry["error"] = "Failed to parse action from output"
            self.detailed_model_logs.append(model_log_entry)
            return base_agent.AgentInteractionResult(False, step_data)

        # Handle answer action specially (cache the answer)
        if len(self._actions) > 0 and self._actions[-1]['arguments']['action'] == 'answer':
            dummy_action = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
            self.env.interaction_cache = self._actions[-1]['arguments']['text']

        self._actions.append(dummy_action)
        self.action_history.append(dummy_action['arguments'].get('action', 'unknown'))

        # Get screen size for coordinate conversion
        screen_size = self.env.device_screen_size

        # Parse and convert action
        try:
            action = self._parse_and_convert_action(dummy_action, screen_size[0], screen_size[1])
        except Exception as e:
            print(Colors.error(f"Failed to convert action: {e}"))
            model_log_entry["error"] = f"Failed to convert action: {e}"
            self.detailed_model_logs.append(model_log_entry)
            return base_agent.AgentInteractionResult(False, step_data)

        if action is None:
            print(Colors.error("Failed to parse action"))
            model_log_entry["error"] = "Action parsing returned None"
            self.detailed_model_logs.append(model_log_entry)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.success(f"Parsed action: {action.action_type}"))
        if hasattr(action, 'x') and action.x is not None:
            print(Colors.info(f"  Coordinates: ({action.x}, {action.y})"))
        if hasattr(action, 'coordinate1') and action.coordinate1 is not None:
            print(Colors.info(f"  Drag from {action.coordinate1} to {action.coordinate2}"))

        # Convert action to dict for storage
        parsed_action_dict = {"action_type": action.action_type}
        if hasattr(action, "x") and action.x is not None:
            parsed_action_dict["x"] = action.x
        if hasattr(action, "y") and action.y is not None:
            parsed_action_dict["y"] = action.y
        if hasattr(action, "text") and action.text is not None:
            parsed_action_dict["text"] = action.text
        if hasattr(action, "coordinate1") and action.coordinate1 is not None:
            parsed_action_dict["coordinate1"] = action.coordinate1
        if hasattr(action, "coordinate2") and action.coordinate2 is not None:
            parsed_action_dict["coordinate2"] = action.coordinate2
        if hasattr(action, "goal_status") and action.goal_status is not None:
            parsed_action_dict["goal_status"] = action.goal_status
        if hasattr(action, "app_name") and action.app_name is not None:
            parsed_action_dict["app_name"] = action.app_name

        # Add actual_action_coordinates for benchmark_run.py compatibility
        if action.action_type in ("click", "long_press"):
            step_data["actual_action_coordinates"] = [action.x, action.y]
        elif action.action_type in ("scroll", "swipe", "drag"):
            step_data["actual_action_coordinates"] = [
                action.coordinate1[0], action.coordinate1[1],
                action.coordinate2[0], action.coordinate2[1],
            ]

        step_data["action"] = parsed_action_dict

        # Log successful parsing
        model_log_entry["parsed_action"] = parsed_action_dict
        model_log_entry["success"] = True
        self.detailed_model_logs.append(model_log_entry)

        # Check if task is complete
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

        # Execute action
        try:
            self.env.execute_action(action)
            print(Colors.success(f"Executed action: {action.action_type}"))
            time.sleep(2.0)
        except Exception as e:
            print(Colors.error(f"Error executing action: {e}"))
            return base_agent.AgentInteractionResult(False, step_data)

        return base_agent.AgentInteractionResult(False, step_data)
