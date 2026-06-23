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

"""Qwen3-VL: Vision-Language Model Agent for Android."""

import base64
import cv2
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


# ANSI颜色代码
class Colors:
    """终端颜色工具类"""

    RED = "\033[91m"  # 错误
    GREEN = "\033[92m"  # 成功
    YELLOW = "\033[93m"  # 警告
    BLUE = "\033[94m"  # 信息
    MAGENTA = "\033[95m"  # 重要信息
    CYAN = "\033[96m"  # 步骤/调试
    BOLD = "\033[1m"  # 粗体
    RESET = "\033[0m"  # 重置

    @staticmethod
    def error(text: str) -> str:
        """红色错误信息"""
        return f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        """绿色成功信息"""
        return f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """黄色警告信息"""
        return f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """蓝色信息"""
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def step(text: str) -> str:
        """青色步骤信息"""
        return f"{Colors.CYAN}{Colors.BOLD}🔹 {text}{Colors.RESET}"

    @staticmethod
    def important(text: str) -> str:
        """紫色重要信息"""
        return f"{Colors.MAGENTA}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        """标题"""
        return f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}\n{text}\n{'=' * 60}{Colors.RESET}"


# Qwen3-VL system prompt defining the tool call format and available actions
SYSTEM_PROMPT = """
You are a helpful assistant that can help with tasks on a mobile device.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000 (coordinates range from 0 to 1000).
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `answer`: Output the answer.
* `system_button`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task. Use status='success' if task completed successfully, status='failure' if task is infeasible or cannot be completed.",
                    "enum": [
                        "click",
                        "long_press",
                        "swipe",
                        "type",
                        "answer",
                        "system_button",
                        "wait",
                        "terminate"
                    ]
                },
                "coordinate": {
                    "type": "array",
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`."
                },
                "coordinate2": {
                    "type": "array",
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`."
                },
                "text": {
                    "type": "string",
                    "description": "Required only by `action=type` and `action=answer`."
                },
                "time": {
                    "type": "number",
                    "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`."
                },
                "button": {
                    "type": "string",
                    "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                    "enum": [
                        "Back",
                        "Home",
                        "Menu",
                        "Enter"
                    ]
                },
                "status": {
                    "type": "string",
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "enum": [
                        "success",
                        "failure"
                    ]
                }
            },
            "required": [
                "action"
            ]
        }
    }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Thinking: a <thinking>...</thinking> block explaining the next move (no multi-step reasoning).
2) Tool call: a <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.
3) Conclusion: a <conclusion>...</conclusion> block with two parts:
   - UI observation: key elements/information visible on current screen (e.g., prices, options, text content) that may be useful for later steps.
   - Intended action: what action you are about to perform.


Rules:
- Output exactly in the order: <thinking>,<tool_call>,<conclusion>.
- Keep <thinking> brief (one sentence). <conclusion> can be longer to capture important UI details.
- Do not output anything else outside those three parts.
- If task completed successfully, use action=terminate with status='success'.
- If task is infeasible (e.g., required app/feature not available, impossible request), use action=terminate with status='failure'.
- IMPORTANT: Action history may not reflect actual execution results. Always verify current UI state before deciding next action - trust what you SEE, not what was intended.
- If you see "Previous Attempt Hints" in the task description, these are lessons learned from previous failed attempts at this same task. Pay attention to these hints and try to avoid repeating the same mistakes."""


# "The user query: Record an audio clip using Audio Recorder app and save it. Task progress (You have done the following operation on the current device): Step 1: Open the audio recorder app.; Step 2: click on the "Get started" button located at the middle and lower part of the screen.; Step 3: "click on the 'Apply' button located at the bottom right."; . Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags. After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags.

# """


class Qwen3VL(base_agent.EnvironmentInteractingAgent):
    """Qwen3-VL agent for Android using vision-language model."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        config: dict[str, Any],
        name: str = "Qwen3VL",
    ):
        """Initializes a Qwen3VL agent.

        Args:
          env: The environment.
          config: Configuration dictionary containing 'QWEN_BASE_URL', 'QWEN_API_KEY',
                  'QWEN_MODEL', and optionally 'QWEN_MAX_PIXELS'.
          name: The agent name.
        """
        super().__init__(env, name)

        if not config.get("QWEN_BASE_URL"):
            raise ValueError("QWEN_BASE_URL is required in config")
        if not config.get("QWEN_API_KEY"):
            raise ValueError("QWEN_API_KEY is required in config")
        if not config.get("QWEN_MODEL"):
            raise ValueError("QWEN_MODEL is required in config")

        base_url = config["QWEN_BASE_URL"]
        api_key = config["QWEN_API_KEY"]
        model_name = config["QWEN_MODEL"]
        max_pixels = config.get("QWEN_MAX_PIXELS")

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_pixels = max_pixels
        self.history = []
        self.qwen3_vl_operations = []
        self.qwen3_vl_step_data = []

        # Enhanced logging for model interactions (similar to UITARS)
        self.detailed_model_logs = []  # Store complete input/output for each model call
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset(self, go_home_on_reset: bool = False):
        """Resets the agent."""
        super().reset(go_home_on_reset)
        self.history = []
        self.qwen3_vl_operations = []
        self.qwen3_vl_step_data = []

        # Reset enhanced logging
        self.detailed_model_logs = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _extract_conclusion(self, output: str) -> str:
        """Extract conclusion text from model output.

        If <conclusion> tags are missing, falls back to extracting text from
        <thinking> tags.

        Args:
          output: Raw output from the model.

        Returns:
          Text within <conclusion> tags, or <thinking> tags if conclusion is missing,
          or empty string if neither is found.
        """
        conclusion_match = re.search(
            r"<conclusion>\s*(.*?)\s*</conclusion>", output, re.DOTALL | re.IGNORECASE
        )
        if conclusion_match:
            return conclusion_match.group(1).strip()

        # Fallback to thinking if conclusion is missing
        thinking_match = re.search(
            r"<thinking>\s*(.*?)\s*</thinking>", output, re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            return thinking_match.group(1).strip()

        return ""

    def _resize_screenshot(self, screenshot_base64: str) -> str:
        """Return the original screenshot as it is.

        Since no resizing is needed, this function simply returns the input.

        Args:
          screenshot_base64: Base64 encoded screenshot

        Returns:
          The original screenshot as base64 string
        """
        return screenshot_base64

    def _sanitize_messages_for_logging(self, messages):
        """
        Keep original messages for logging without any sanitization.
        This preserves the complete base64 image data for debugging purposes.

        Args:
          messages: List of message dictionaries

        Returns:
          The original messages without any modification
        """
        # Return the original messages without any modification
        return messages

    def get_enhanced_log_data(self):
        """
        Get enhanced logging data including detailed model interactions.
        This method should be called when saving execution results.

        Returns:
          Dictionary containing detailed model logs and statistics
        """
        return {
            "detailed_model_logs": self.detailed_model_logs,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_model_calls": len(self.detailed_model_logs),
        }

    def _parse_qwen3_vl_output(
        self, output: str, screen_width: int, screen_height: int
    ) -> json_action.JSONAction | None:
        """Parse Qwen3-VL model output and convert to JSONAction.

        Args:
          output: Raw output from the model
          screen_width: Screen width in pixels
          screen_height: Screen height in pixels

        Returns:
          JSONAction object or None if parsing fails
        """
        # Extract tool call from output
        action_match = re.search(
            r"<tool_call>\s*(.*)\s*</tool_call>", output, re.DOTALL
        )
        if not action_match:
            print(Colors.error(f"No tool call found in output"))
            print(Colors.warning(f"Model output: {output[:200]}..."))
            return None

        try:
            action_dict = json.loads(action_match.group(1).strip())
            # Handle both formats:
            # 1. Standard: {"name": "mobile_use", "arguments": {"action": "...", ...}}
            # 2. Simplified: {"name": "mobile_use", "action": "...", ...}
            if "arguments" in action_dict:
                action_args = action_dict.get("arguments", {})
            else:
                # If no "arguments" key, treat the whole dict (except "name") as arguments
                action_args = {k: v for k, v in action_dict.items() if k != "name"}
        except json.JSONDecodeError as e:
            print(Colors.error(f"Failed to parse JSON: {e}"))
            print(Colors.warning(f"Raw content: {action_match.group(1).strip()[:200]}"))
            return None

        action_type = action_args.get("action")

        # Convert GUI-Owl actions to JSONAction format
        if action_type == "click":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x, y = int(coords[0]), int(coords[1])
                x = int(x * screen_width / 1000)
                y = int(y * screen_height / 1000)
                return json_action.JSONAction(action_type="click", x=x, y=y)

        elif action_type == "long_press":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x, y = int(coords[0]), int(coords[1])
                x = int(x * screen_width / 1000)
                y = int(y * screen_height / 1000)
                # Note: long_press duration is hardcoded in adb_utils, so we don't pass it
                return json_action.JSONAction(action_type="long_press", x=x, y=y)

        elif action_type == "swipe":
            coords = action_args.get("coordinate", [None, None])
            coords2 = action_args.get("coordinate2", [None, None])

            # Check if it's a swipe with start and end coordinates (which maps to drag in AndroidWorld)
            if (
                len(coords) >= 2
                and len(coords2) >= 2
                and coords[0] is not None
                and coords2[0] is not None
            ):
                x1 = int(coords[0] * screen_width / 1000)
                y1 = int(coords[1] * screen_height / 1000)
                x2 = int(coords2[0] * screen_width / 1000)
                y2 = int(coords2[1] * screen_height / 1000)

                return json_action.JSONAction(
                    action_type="drag", coordinate1=(x1, y1), coordinate2=(x2, y2)
                )

            direction = action_args.get("direction")

            # Map Qwen3-VL directions to AndroidWorld conventions
            direction_map = {
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
            }

            if direction in direction_map:
                mapped_direction = direction_map[direction]

                if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                    x, y = int(coords[0]), int(coords[1])
                    x = int(x * screen_width / 1000)
                    y = int(y * screen_height / 1000)
                    return json_action.JSONAction(
                        action_type="swipe",
                        x=x,
                        y=y,
                        direction=mapped_direction,
                    )
                else:
                    return json_action.JSONAction(
                        action_type="swipe",
                        direction=mapped_direction,
                    )

        elif action_type == "drag":
            coords1 = action_args.get("coordinate", [None, None])
            coords2 = action_args.get("coordinate2", [None, None])

            if (
                len(coords1) >= 2
                and len(coords2) >= 2
                and coords1[0] is not None
                and coords2[0] is not None
            ):
                x1 = int(coords1[0] * screen_width / 1000)
                y1 = int(coords1[1] * screen_height / 1000)
                x2 = int(coords2[0] * screen_width / 1000)
                y2 = int(coords2[1] * screen_height / 1000)

                return json_action.JSONAction(
                    action_type="drag", coordinate1=(x1, y1), coordinate2=(x2, y2)
                )

        elif action_type == "type":
            text = action_args.get("text", "")
            if text:
                return json_action.JSONAction(action_type="input_text", text=text)

        elif action_type == "answer":
            text = action_args.get("text", "")
            return json_action.JSONAction(action_type="answer", text=text)

        elif action_type == "system_button":
            button = action_args.get("button", "")
            if button == "Back":
                return json_action.JSONAction(action_type="navigate_back")
            elif button == "Home":
                return json_action.JSONAction(action_type="navigate_home")
            elif button == "Menu":
                return json_action.JSONAction(action_type="navigate_menu")
            elif button == "Enter":
                return json_action.JSONAction(action_type="keyboard_enter")

        elif action_type == "open":
            app_name = action_args.get("text", "").lower()
            if app_name:
                return json_action.JSONAction(action_type="open_app", app_name=app_name)

        elif action_type == "wait":
            # Note: wait action in actuation.py sleeps for 1 second, ignoring time parameter
            return json_action.JSONAction(action_type="wait")

        elif action_type == "terminate":
            status = action_args.get("status", "")
            if status == "success":
                return json_action.JSONAction(
                    action_type="status", goal_status="complete"
                )
            else:
                return json_action.JSONAction(
                    action_type="status", goal_status="infeasible"
                )

        print(Colors.error(f"Unknown or malformed action: {action_type}"))
        return None

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Performs a step of the agent on the environment.

        Args:
          goal: The goal/task description.

        Returns:
          AgentInteractionResult containing done status and step data.
        """
        step_data = {
            "before_screenshot": None,
            "action_output": None,
            "raw_response": None,
        }

        step_num = len(self.history)
        print(Colors.header(f"Step {step_num + 1}"))

        # Get current state
        state = self.get_post_transition_state()
        step_data["before_screenshot"] = state.pixels.copy()

        # Convert screenshot to base64
        _, buffer = cv2.imencode(".png", state.pixels)
        screenshot_base64 = base64.b64encode(buffer).decode("utf-8")

        # Resize screenshot
        screenshot_resized = self._resize_screenshot(screenshot_base64)

        # Build action history string (keep for compatibility)
        action_history_str = []
        for i, hist_item in enumerate(self.history):
            action_summary = hist_item.get("action_summary", "unknown")
            action_history_str.append(f"Step{i + 1}: {action_summary}")
        action_history_str = (
            ", ".join(action_history_str) if action_history_str else "None"
        )

        # Build action_history_desc from conclusions
        action_history_desc = []
        for hist_item in self.history:
            conclusion = hist_item.get("conclusion", "")
            if conclusion:
                action_history_desc.append(conclusion)

        # Format action_history_desc for prompt
        if action_history_desc:
            action_history_desc_str = "\n".join(
                [f"Step {i + 1}: {desc}" for i, desc in enumerate(action_history_desc)]
            )
        else:
            action_history_desc_str = ""

        # Create user prompt using action_history_desc
        user_prompt = (
            f"The user query: {goal}\n"
            f"Task progress (You have done the following operation on the current device): "
            f"{action_history_desc_str}<image>"
            ". Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags. After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."
        )

        step_data["user_prompt"] = user_prompt
        step_data["system_prompt"] = SYSTEM_PROMPT
        print(Colors.step("User prompt:"))
        print(Colors.info(f"{user_prompt}\n"))

        # Prepare messages for API call
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image;base64,{screenshot_resized}"},
                    },
                ],
            },
        ]

        # Record the API call details for enhanced logging
        api_call_start_time = time.time()
        model_log_entry = {
            "step": step_num + 1,
            "retry_count": 1,  # Qwen3VL doesn't retry by default, but keep for consistency
            "timestamp": api_call_start_time,
            "input_messages": self._sanitize_messages_for_logging(api_messages),
            "model": self.model_name,
            "raw_response": None,
            "parsed_action": None,
            "success": False,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_call_duration": 0.0,
            # Add logprobs support for DART-GUI distribution alignment
            "logprobs": None,  # Will store token-level log probabilities
            "token_logprobs_sum": None,  # Sum of all token log probs (for rollout_log_probs)
        }

        # Call Qwen3-VL model with logprobs enabled for DART-GUI distribution alignment
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                logprobs=True,  # Enable logprobs for distribution alignment
            )

            api_call_end_time = time.time()
            model_log_entry["api_call_duration"] = (
                api_call_end_time - api_call_start_time
            )

            response_str = response.choices[0].message.content
            model_log_entry["raw_response"] = response_str

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

            # Extract logprobs information for DART-GUI distribution alignment
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                if hasattr(logprobs_data, "content") and logprobs_data.content:
                    # Store detailed token-level logprobs
                    token_logprobs = []
                    logprobs_sum = 0.0
                    
                    for token_info in logprobs_data.content:
                        token_entry = {
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                        }
                        # Optionally include bytes if available
                        if hasattr(token_info, "bytes") and token_info.bytes:
                            token_entry["bytes"] = token_info.bytes
                        
                        token_logprobs.append(token_entry)
                        logprobs_sum += token_info.logprob
                    
                    model_log_entry["logprobs"] = token_logprobs
                    model_log_entry["token_logprobs_sum"] = logprobs_sum
                    model_log_entry["num_completion_tokens_with_logprobs"] = len(token_logprobs)
                    
                    print(Colors.info(f"  Logprobs recorded: {len(token_logprobs)} tokens, sum={logprobs_sum:.4f}"))

        except Exception as e:
            print(Colors.error(f"Error calling Qwen3-VL model: {e}"))
            model_log_entry["error"] = str(e)
            model_log_entry["api_call_duration"] = time.time() - api_call_start_time
            self.detailed_model_logs.append(model_log_entry)
            step_data["raw_response"] = str(e)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.important("Qwen3-VL output:"))
        print(Colors.info(f"{response_str}\n"))
        step_data["action_output"] = response_str
        step_data["raw_response"] = response_str

        # Extract conclusion from response
        conclusion = self._extract_conclusion(response_str)
        step_data["conclusion"] = conclusion

        # Parse output to action
        screen_size = self.env.device_screen_size
        action = self._parse_qwen3_vl_output(
            response_str, screen_size[0], screen_size[1]
        )

        if action is None:
            print(Colors.error("Failed to parse action from Qwen3-VL output"))
            step_data["action_summary"] = "Failed to parse action"
            step_data["parsed_action"] = {}
            action_history_desc.append("Failed to parse action")
            step_data["action_history_desc"] = action_history_desc.copy()

            # Log the failure
            model_log_entry["parsed_action"] = None
            model_log_entry["error"] = "Failed to parse action from output"
            self.detailed_model_logs.append(model_log_entry)

            self.history.append(step_data)
            self._update_qwen3_vl_data(goal, step_num + 1, step_data, {}, conclusion)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.success(f"Parsed action: {action.action_type}"))
        if hasattr(action, "x") and action.x is not None:
            print(Colors.info(f"  Coordinates: ({action.x}, {action.y})"))

        # Convert action to dict for storage
        parsed_action_dict = {
            "action_type": action.action_type,
        }
        if hasattr(action, "x") and action.x is not None:
            parsed_action_dict["x"] = action.x
        if hasattr(action, "y") and action.y is not None:
            parsed_action_dict["y"] = action.y
        if hasattr(action, "text") and action.text is not None:
            parsed_action_dict["text"] = action.text
        if hasattr(action, "direction") and action.direction is not None:
            parsed_action_dict["direction"] = action.direction
        if hasattr(action, "coordinate1") and action.coordinate1 is not None:
            parsed_action_dict["coordinate1"] = action.coordinate1
        if hasattr(action, "coordinate2") and action.coordinate2 is not None:
            parsed_action_dict["coordinate2"] = action.coordinate2
        if hasattr(action, "goal_status") and action.goal_status is not None:
            parsed_action_dict["goal_status"] = action.goal_status

        step_data["parsed_action"] = parsed_action_dict

        # Log successful parsing
        model_log_entry["parsed_action"] = parsed_action_dict
        model_log_entry["success"] = True
        self.detailed_model_logs.append(model_log_entry)

        # Check if task is complete
        if action.action_type == "status":
            step_data["action_summary"] = (
                f"Task completed with status: {action.goal_status}"
            )
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(True, step_data)
        elif action.action_type == "answer":
            step_data["action_summary"] = f"Answer: {action.text}"
            # Execute answer action to save the answer to interaction_cache
            try:
                actual_action_coordinates = self.env.execute_action(action)
                step_data["actual_action_coordinates"] = actual_action_coordinates
                print(Colors.success(f"Executed answer action"))
                print(Colors.info(f"  Answer text: {action.text}"))
            except Exception as e:
                print(Colors.error(f"Error executing answer action: {e}"))
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(True, step_data)

        # Execute action
        try:
            actual_action_coordinates = self.env.execute_action(action)
            step_data["actual_action_coordinates"] = actual_action_coordinates
            step_data["action_summary"] = f"{action.action_type}"
            print(Colors.success(f"Executed action: {action.action_type}"))
            time.sleep(2.0)
        except Exception as e:
            print(Colors.error(f"Error executing action: {e}"))
            step_data["action_summary"] = (
                f"Error executing {action.action_type}: {str(e)}"
            )
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(False, step_data)

        # Add conclusion to action_history_desc
        action_history_desc.append(
            conclusion if conclusion else step_data["action_summary"]
        )
        step_data["action_history_desc"] = action_history_desc.copy()

        self.history.append(step_data)
        self._update_qwen3_vl_data(
            goal, step_num + 1, step_data, parsed_action_dict, conclusion
        )

        # Add Qwen3 VL specific data to step_data for checkpointer
        step_data["qwen3_vl_operations"] = self.qwen3_vl_operations
        step_data["qwen3_vl_step_data"] = self.qwen3_vl_step_data

        return base_agent.AgentInteractionResult(False, step_data)

    def _update_qwen3_vl_data(
        self,
        goal: str,
        step_id: int,
        step_data: dict,
        parsed_action: dict,
        conclusion: str,
    ):
        """Update Qwen3 VL specific data structures.

        Args:
          goal: The goal/task description.
          step_id: Current step number.
          step_data: Step data dictionary.
          parsed_action: Parsed action dictionary.
          conclusion: Extracted conclusion text.
        """
        # Update operations list
        self.qwen3_vl_operations = {
            "instruction": goal,
            "episode_id": f"episode_0",
            "steps": [],
        }

        for i, hist_item in enumerate(self.history):
            step_info = {
                "step_id": i + 1,
                "image_path": f"./screenshots/step_{i:02d}.png",
                "action": hist_item.get("parsed_action", {}),
                "conclusion": hist_item.get("conclusion", ""),
            }
            self.qwen3_vl_operations["steps"].append(step_info)

        # Update step data list
        self.qwen3_vl_step_data = []
        for i, hist_item in enumerate(self.history):
            step_entry = {
                "step_id": i + 1,
                "screenshot_path": f"./screenshots/step_{i:02d}.png",
                "action_prompt": hist_item.get("user_prompt", ""),
                "system_prompt": hist_item.get("system_prompt", ""),
                "action_output": hist_item.get("raw_response", ""),
                "parsed_action": hist_item.get("parsed_action", {}),
                "conclusion": hist_item.get("conclusion", ""),
                "action_history_desc": hist_item.get("action_history_desc", []),
            }
            self.qwen3_vl_step_data.append(step_entry)
