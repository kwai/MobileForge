import ast
import base64
import io
import re
import time
import os

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.env import interface, json_action
from android_world.agents.uitars_utils import (
    smart_resize,
    parse_action_to_structure_output,
)

MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""


def convert_action(action_str, param_str, image_bounds, min_pixels, max_pixels) -> dict:
    # Reconstruct the full action string for parse_action_to_structure_output
    full_action_str = f"Action: {action_str}({param_str})"

    # Get image dimensions for parse_action_to_structure_output
    image_width = image_bounds[2] - image_bounds[0]
    image_height = image_bounds[3] - image_bounds[1]

    # Use parse_action_to_structure_output to parse the action
    parsed_actions = parse_action_to_structure_output(
        text=full_action_str,
        factor=28,  # IMAGE_FACTOR from uitars_utils
        origin_resized_height=image_height,
        origin_resized_width=image_width,
        model_type="qwen25vl",
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    if not parsed_actions:
        raise ValueError("No actions parsed")

    # Get the first parsed action
    parsed_action = parsed_actions[0]
    action_type = parsed_action.get("action_type")
    action_inputs = parsed_action.get("action_inputs", {})

    # Helper function to parse coordinate string to list
    def parse_coordinate_string(coord_str):
        if isinstance(coord_str, str):
            try:
                # Remove any extra whitespace and parse as literal
                return ast.literal_eval(coord_str.strip())
            except:
                return None
        return coord_str

    # Convert to the expected return format
    if action_type in ["click", "long_press"]:
        start_box_str = action_inputs.get("start_box", "")
        start_box = parse_coordinate_string(start_box_str)

        if start_box and len(start_box) >= 2:
            # start_box contains normalized coordinates, convert to pixel coordinates
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
        direction = action_inputs.get("direction", "")
        point_str = action_inputs.get("start_box", "")
        if not direction:
            pattern = r"direction=\'(.*)\',"
            match = re.search(pattern, param_str, re.DOTALL)
            if match:
                direction = match.group(1).strip()
        if point_str:
            point_coords = parse_coordinate_string(point_str)
            if point_coords and len(point_coords) >= 2:
                x = int(point_coords[0] * image_width)
                y = int(point_coords[1] * image_height)
                ret = {"action_type": "scroll", "x": x, "y": y, "direction": direction}
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
            x1 = int(start_box[0] * image_width)
            y1 = int(start_box[1] * image_height)
            x2 = int(end_box[0] * image_width)
            y2 = int(end_box[1] * image_height)
        else:
            # Fallback to original parsing if coordinate parsing fails
            pattern = r"start_point='<point>(\d+)\s+(\d+)</point>', end_point='<point>(\d+)\s+(\d+)</point>'"
            match = re.search(pattern, param_str)
            if not match:
                raise ValueError(f"Cannot parse drag coordinates from: {param_str}")
            x1, y1, x2, y2 = map(int, match.groups())

        ret = {"action_type": "drag", "coordinate1": [x1, y1], "coordinate2": [x2, y2]}
    elif action_type == "type":
        text = action_inputs.get("content", "")
        if not text:
            pattern = r"content=\'(.*)\'"
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
            pattern = r"content=\'(.*)\'"
            match = re.search(pattern, param_str, re.DOTALL)
            if match:
                status = match.group(1).strip()
        ret = {"action_type": "status", "goal_status": status}
    else:
        ret = {"action_type": "unknown"}

    return ret


class UITARS_1_5(base_agent.EnvironmentInteractingAgent):
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
                "Config parameter is required for UITARS_1_5 agent. Please provide UITARS_BASE_URL, UITARS_API_KEY, and UITARS_MODEL in config."
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

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.history = []
        self.step_idx = 1
        self.message_history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.max_pixels = 1280 * 28 * 28
        self.min_pixels = 100 * 28 * 28
        # Read history_n from config, default to 5 if not specified (UITARS_1_5 default)
        self.history_n = config["UITARS_HISTORY_N"]

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []
        self.step_idx = 1
        self.message_history = []

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

        sys_prompt = MOBILE_USE_DOUBAO

        before_screenshot = Image.fromarray(before_screenshot, "RGB")
        buffer = io.BytesIO()
        before_screenshot.save(buffer, format="PNG")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        screenshot_base64 = base64.b64encode(image_bytes).decode("utf-8")

        image_bounds = [0, 0, before_screenshot.width, before_screenshot.height]

        # Build current user messages for this step
        if self.step_idx == 1:
            # First step: split into two user messages (text then image)
            current_user_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": sys_prompt + goal,
                            # "text": "请分析这张图片，并以JSON格式返回你所看到的它的精确宽度和高度。例如: {\"width\": 1920, \"height\": 1080}"
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
            # Subsequent steps: only send the image as user message
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

        # Prepare messages for the API call over (history + current)
        # 1) Find initial user message with text in history
        initial_user_with_text = None
        for msg in self.message_history:
            if msg.get("role") != "user":
                continue
            c = msg.get("content")
            if isinstance(c, str) and c:
                initial_user_with_text = msg
                break
            if isinstance(c, list) and any(
                isinstance(p, dict)
                and p.get("type") == "text"
                and p.get("min_pixels") is not None
                and p.get("max_pixels") is not None
                for p in c
            ):
                initial_user_with_text = msg
                break

        # 2) Build a combined list for this call
        full_for_this_call = list(self.message_history) + list(current_user_messages)

        # Helper to detect image presence
        def _has_image(m):
            c = m.get("content")
            return isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c
            )

        # 3) Collect indices of all user messages with images
        image_indices = [
            i
            for i, m in enumerate(full_for_this_call)
            if m.get("role") == "user" and _has_image(m)
        ]
        # Keep only the last 5
        keep_image_indices = set(image_indices[-5:])

        # 4) Reconstruct messages_for_api with rules:
        #    - keep all assistant
        #    - keep initial user-with-text always, but strip image if its index not in keep set
        #    - keep other user messages only if their index in keep set
        messages_for_api = []
        for idx, message in enumerate(full_for_this_call):
            role = message.get("role")
            if role == "assistant":
                messages_for_api.append(message)
                continue
            if role != "user":
                continue

            content = message.get("content")
            if message is initial_user_with_text:
                # include initial; strip image if needed
                if idx in keep_image_indices:
                    messages_for_api.append(message)
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
                        messages_for_api.append(
                            {
                                "role": "user",
                                "content": filtered if filtered else content,
                            }
                        )
                    else:
                        messages_for_api.append(message)
            else:
                # other user messages: include only if their images are selected
                if idx in keep_image_indices:
                    messages_for_api.append(message)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_for_api,
        )
        # response = self.client.chat.completions.create(
        #     model="app-211nse-1751533064887408034",  # app-211nse-1751533064887408034 为您当前的智能体应用的ID
        #     messages=messages_for_api,
        # )

        raw_response = response.choices[0].message.content
        current_assistant_message = {
            "role": "assistant",
            "content": raw_response,
        }

        # Append the current turn to the full message history for the next step
        self.message_history.extend(current_user_messages)
        self.message_history.append(current_assistant_message)

        # Prune older user image messages from history, keep only the latest 5
        # Keep: all assistant messages and the very first user text message (strip its image if not within last 5)
        # 1) Find initial user-with-text in history
        initial_user_with_text_hist = None
        for msg in self.message_history:
            if msg.get("role") != "user":
                continue
            c = msg.get("content")
            if isinstance(c, str) and c:
                initial_user_with_text_hist = msg
                break
            if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "text" for p in c
            ):
                initial_user_with_text_hist = msg
                break

        # 2) Compute indices of user image messages in history
        image_indices_hist = [
            i
            for i, m in enumerate(self.message_history)
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in m.get("content")
            )
        ]
        keep_image_indices_hist = set(image_indices_hist[-self.history_n :])

        # 3) Rebuild history with stripping image from initial if needed
        new_history = []
        for idx, msg in enumerate(self.message_history):
            role = msg.get("role")
            if role == "assistant":
                new_history.append(msg)
                continue
            if role != "user":
                continue

            if msg is initial_user_with_text_hist:
                if idx in keep_image_indices_hist:
                    new_history.append(msg)
                else:
                    c = msg.get("content")
                    if isinstance(c, list):
                        filtered = [
                            part
                            for part in c
                            if not (
                                isinstance(part, dict)
                                and part.get("type") == "image_url"
                            )
                        ]
                        new_history.append(
                            {"role": "user", "content": filtered if filtered else c}
                        )
                    else:
                        new_history.append(msg)
            else:
                if idx in keep_image_indices_hist:
                    new_history.append(msg)
        self.message_history = new_history

        action_pattern = r"Action: (.*)"
        action_match = re.search(action_pattern, raw_response, re.DOTALL)
        if not action_match:
            # Handle cases where the model does not output an action
            print("No action found in the response.")
            step_data["summary"] = "No action found in the model's response."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)
        action_raw_str = action_match.group(1).strip()

        step_data["raw_response"] = raw_response
        step_data["raw_action"] = action_raw_str
        step_data["action_raw_response"] = raw_response
        step_data["action_output"] = action_raw_str

        print(raw_response)

        for i, c in enumerate(action_raw_str):
            if c == "(":
                first_bracket_idx = i
                break

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
        except Exception as e:
            print("Failed to convert the output to a valid action.")
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        try:
            actual_action_coordinates = self.env.execute_action(converted_action)
            step_data["actual_action_coordinates"] = actual_action_coordinates
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        time.sleep(self.wait_after_action_seconds)

        state = self.env.get_state(wait_to_stabilize=False)
        after_screenshot = state.pixels.copy()
        step_data["after_screenshot"] = after_screenshot.copy()

        self.history.append(step_data)
        self.step_idx += 1
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )
