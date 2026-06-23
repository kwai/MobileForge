import requests
import os
from typing import Dict, List, Tuple, Any, Optional
import nest_asyncio
import time
import aiohttp
import asyncio
import dataclasses
import xml.etree.ElementTree as ET
from dataclasses import asdict
from PIL import Image
from utils.prompt_templates import REASONING, SUMMARY, RANKER


### 用于demo
def get_a_message3():
    demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
    if not demo_on:
        return None
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:8768")
        + "/get_a_massage3"
    )
    try:
        rsp = requests.get(url)
        try:
            return rsp.json()
        except:
            return rsp.text
    except:
        return None


def is_need_stop() -> bool:
    demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
    if not demo_on:
        return False
    msg = str(get_a_message3()).lower()
    return "stop" in msg


def send_message(message: dict = None, text: str = None, images: list[str] = None):
    demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
    if not demo_on:
        return None
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:8768")
        + "/sent_a_massage"
    )
    try:
        _message = message if message else {}
        if text:
            _message["text"] = text
        if images:
            _message["images"] = images
        if len(_message.keys()) == 0:
            return None
        rsp = requests.post(url, json=_message)
    except:
        pass


def send_message2(message: dict = None, text: str = None, images: list[str] = None):
    demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
    if not demo_on:
        return None
    url = (
        os.getenv("MESSAGE_SERVER_ENDPOINT", "http://127.0.0.1:8768")
        + "/sent_a_massage2"
    )
    try:
        _message = message if message else {}
        if text:
            _message["text"] = text
        if images:
            _message["images"] = images
        if len(_message.keys()) == 0:
            return None
        rsp = requests.post(url, json=_message)
    except:
        pass


### 用于检索知识


def retrieval_batch_api(
    queries: list[Image.Image],
    top_k: int = 1,
    threshold: float = 0.9,
    package_name: str = None,
) -> list[list[dict]]:
    """检索出query对应的knowledge

    Returns:
        List[List[dict[str,Any]]]: 返回的结果列表（注意长度可能小于top_k）
    """
    ret, rsp, max_retries = None, None, 3
    data = {
        "package_name": package_name,  # "com.example.app",
        "queries": [pil_to_webp_base64(query) for query in queries],  # "base64 image",
        "top_k": top_k,
        "threshold": threshold,
    }
    url = os.getenv("RAG_SERVER_ENDPOINT", "http://localhost:8769") + "/retrieval_batch"
    for i in range(max_retries):
        try:
            rsp = requests.post(url, json=data, timeout=300)
            ret = rsp.json()
            return ret["results"]
        except Exception as e:
            print(f"retrieval_api error: {e} retrying {i+1}/{max_retries}")
            if i == max_retries - 1:
                raise e
            time.sleep(1)


### 用于排序知识
async def ranking_ask(
    prompt: str,
    session: aiohttp.ClientSession,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    api_url = (
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
    )
    api_key = os.getenv("OPENAI_API_KEY")
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_retries = 3
    for i in range(max_retries):
        try:
            async with session.post(api_url, json=data, headers=headers) as response:
                d = await response.json()
                content = (
                    d.get("choices", [{}])[0].get("message", {}).get("content", "1")
                )
                usage["prompt_tokens"] += d.get("usage", {}).get("prompt_tokens", 0)
                usage["completion_tokens"] += d.get("usage", {}).get(
                    "completion_tokens", 0
                )
                return content
        except Exception as e:
            print(f"Error: {e} at {i}th retry")
            await asyncio.sleep(1)
    return "1"


def generate_ranking_prompt(
    instruction: str, knowledge_a: str, knowledge_b: str
) -> str:
    return RANKER.format(
        task_goal=instruction, knowledge_a=knowledge_a, knowledge_b=knowledge_b
    )


async def compare_knowledge_utility(
    instruction: str,
    knowledge_1: str,
    knowledge_2: str,
    session: aiohttp.ClientSession,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> bool:
    ranking_prompt = generate_ranking_prompt(instruction, knowledge_1, knowledge_2)
    selected_index_str = await ranking_ask(ranking_prompt, session, usage)
    return "2" not in selected_index_str


async def merge_sorted_partitions(
    instruction: str,
    left_partition: List[Dict[str, Any]],
    right_partition: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    usage: Dict[str, int],
) -> List[Dict[str, Any]]:
    sorted_entries = []
    left_idx = right_idx = 0

    while left_idx < len(left_partition) and right_idx < len(right_partition):
        comparison_result = await compare_knowledge_utility(
            instruction,
            str(left_partition[left_idx]["hints"]),
            str(right_partition[right_idx]["hints"]),
            session,
            usage,
        )

        if comparison_result:
            sorted_entries.append(left_partition[left_idx])
            left_idx += 1
        else:
            sorted_entries.append(right_partition[right_idx])
            right_idx += 1

    sorted_entries.extend(left_partition[left_idx:])
    sorted_entries.extend(right_partition[right_idx:])
    return sorted_entries


async def AsyncRanker(
    instruction: str,
    knowledge_entries: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> List[Dict[str, Any]]:
    """Asynchronously rank knowledge entries based on instruction

    Args:
        instruction (str): Task description
        knowledge_entries (List[Dict[str, Any]]): List of knowledge entries to rank
        session (aiohttp.ClientSession): Shared aiohttp session for API calls
        usage (Dict[str, int]): Token usage tracker

    Returns:
        List[Dict[str, Any]]: Sorted knowledge entries
    """
    if len(knowledge_entries) <= 1:
        return knowledge_entries

    mid_index = len(knowledge_entries) // 2

    # Create tasks for both partitions
    left_task = asyncio.create_task(
        AsyncRanker(instruction, knowledge_entries[:mid_index], session, usage)
    )
    right_task = asyncio.create_task(
        AsyncRanker(instruction, knowledge_entries[mid_index:], session, usage)
    )

    # Await both partitions
    left_partition = await left_task
    right_partition = await right_task

    # Merge the sorted partitions
    return await merge_sorted_partitions(
        instruction, left_partition, right_partition, session, usage
    )


async def _async_main(
    instruction: str,
    knowledge_entries: List[Dict[str, Any]],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> List[Dict[str, Any]]:
    """Internal async main function to handle the async session and ranking"""

    async with aiohttp.ClientSession() as session:
        sorted_entries = await AsyncRanker(
            instruction, knowledge_entries, session, usage
        )
        return sorted_entries


def Ranker(
    instruction: str,
    knowledge_entries: List[Dict[str, Any]],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for the asynchronous ranker

    Args:
        instruction (str): Task description
        knowledge_entries (List[Dict[str, Any]]): List of knowledge entries to rank

    Returns:
        List[Dict[str, Any]]: Sorted knowledge entries
    """
    # 应用 nest_asyncio 来允许嵌套事件循环
    nest_asyncio.apply()

    try:
        # 在异步环境中运行
        if asyncio._get_running_loop() is not None:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    _async_main(instruction, knowledge_entries, usage)
                )
            finally:
                loop.close()
        else:
            # 在同步环境中运行
            return asyncio.run(_async_main(instruction, knowledge_entries, usage))
    except RuntimeError as e:
        # 如果出现运行时错误，尝试使用现有的事件循环
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            _async_main(instruction, knowledge_entries, usage)
        )


import base64
import re
from typing import Any, Optional
import cv2
import numpy as np


def parse_reason_action_output(
    raw_reason_action_output: str,
) -> tuple[Optional[str], Optional[str]]:
    r"""Parses llm action reason output.

    Args:
      raw_reason_action_output: Raw string output that supposes to have the format
        'Reasoning: xxx\nAction:xxx'.

    Returns:
      If parsing successfully, returns reason and action.
    """
    # 空值防护，避免正则匹配 None 导致 TypeError
    if not raw_reason_action_output:
        return None, None
    
    reason_result = re.search(
        r"Reasoning:(.*)Action:", raw_reason_action_output, flags=re.DOTALL
    )
    reason = reason_result.group(1).strip() if reason_result else None
    action_result = re.search(r"Action:(.*)", raw_reason_action_output, flags=re.DOTALL)
    action = action_result.group(1).strip() if action_result else None
    return reason, action


from utils.utils import extract_json
import io
from PIL import Image
import base64


from utils.utils import (
    pil_to_webp_base64,
    ndarray_to_webp_base64,
    resize_pil_image,
    resize_ndarray_image,
)


import logging
import uiautomator2 as u2
from PIL import Image
from typing import List, Tuple, Union
import time
import re
from utils.device import (
    _generate_ui_elements_description_list,
    validate_ui_element,
    add_ui_element_mark,
    _ui_element_logical_corner,
    _logical_to_physical,
    add_screenshot_label,
    Device,
)


def _action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
    knowledge_prompt: str = "",
) -> str:
    """Generate the prompt for the action selection.

    Args:
      goal: The current goal.
      history: Summaries for previous steps.
      ui_elements: A list of descriptions for the UI elements.

    Returns:
      The text prompt for action selection that will be sent to MLLM.
    """
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    return REASONING.format(
        task_goal=goal,
        history=history,
        ui_elements=ui_elements if ui_elements else "Not available",
        knowledge=knowledge_prompt if knowledge_prompt else "Not available",
    )


def _summarize_prompt(
    action: str,
    reasoning: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reasoning: The reasoning to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY.format(
        task_goal=goal,
        before_ui_elements=before_elements,
        after_ui_elements=after_elements,
        action=action,
        reasoning=reasoning,
    )


from MLLM_Agent import json_action

import io
import numpy as np


import requests


def ask_mllm(text_prompt: str, images: list[np.ndarray]) -> tuple[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
    }

    payload = {
        "model": os.getenv("OPENAI_API_MODEL"),
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        "max_tokens": 1000,
    }
    low_resolution = os.getenv("LOW_RESOLUTION", "False").lower() == "true"
    for image in images:
        if low_resolution:
            # Resize the image to a lower resolution for faster processing.
            image = resize_ndarray_image(image, 1000)
        payload["messages"][0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{ndarray_to_webp_base64(image)}"
                },
            }
        )

    counter = 5  # max_retry
    wait_seconds = 1
    response = None
    while counter > 0:
        try:
            response = requests.post(
                #'https://api.openai.com/v1/chat/completions',
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if response.ok and "choices" in response.json():
                return response.json()["choices"][0]["message"]["content"], response
            print(
                "Error calling OpenAI API with error message: "
                + response.json()["error"]["message"]
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Want to catch all exceptions happened during LLM calls.
            time.sleep(wait_seconds)
            wait_seconds *= 2
            counter -= 1
            print("Error calling LLM, will retry soon...")
            print(e)
            if response is not None:
                print(response.text)
    return "Error calling LLM", None


import copy


def send_android_intent(
    command: str,
    action: str,
    device_controller: Device,
    data_uri: str | None = None,
    mime_type: str | None = None,
    extras: dict[str, Any] | None = None,
    timeout_sec: int = 10,
):
    """Sends an intent to Android device using adb.

    This is a low-level command for sending an intent with additional parameters.
    When these additional parameters are not necessary, consider instead using
    `adb_utils.start_activity()` or `env.execute_adb_call()` with
    `AdbRequest.StartActivity` or `AdbRequest.SendBroadcast`.

    Args:
      command: Either "start" for start activity intents or "broadcast" for
        broadcast intents.
      action: The broadcast action (e.g. "android.intent.action.VIEW").
      device_controller: The environment to which the broadcast is sent.
      data_uri: Optional intent data URI (e.g. "content://contacts/people/1").
      mime_type: Optional mime type (e.g. "image/png").
      extras: Dictionary containing keys and values to be sent as extras.
      timeout_sec: The maximum time in seconds to wait for the broadcast to
        complete.

    Returns:
      AdbResponse object.
    """
    if command not in ["start", "broadcast"]:
        raise ValueError('Intent command must be either "start" or "broadcast"')

    # adb_command = ["shell", "am", command, "-a", action]
    adb_command = ["am", command, "-a", action]

    if data_uri:
        adb_command.extend(["-d", f'"{data_uri}"'])

    if mime_type:
        adb_command.extend(["-t", f'"{mime_type}"'])

    if extras:
        for key, value in extras.items():
            if value is tuple:
                type_override, value = value
                if type_override == "str":
                    adb_command.extend(["--es", key, f'"{value}"'])
                elif type_override == "bool":
                    adb_command.extend(["--ez", key, f'"{value}"'])
                elif type_override == "int":
                    adb_command.extend(["--ei", key, f'"{value}"'])
                elif type_override == "long":  # long type only available via override.
                    adb_command.extend(["--el", key, f'"{value}"'])
                elif type_override == "float":
                    adb_command.extend(["--ef", key, f'"{value}"'])
                elif type_override == "string array":
                    array_str = ",".join(value)
                    adb_command.extend(["--esa", key, f'"{array_str}"'])
            elif isinstance(value, str):
                adb_command.extend(["--es", key, f'"{value}"'])
            elif isinstance(value, bool):
                adb_command.extend(["--ez", key, f'"{value}"'])
            elif isinstance(value, int):
                adb_command.extend(["--ei", key, f'"{value}"'])
            # long type only available via override above.
            elif isinstance(value, float):
                adb_command.extend(["--ef", key, f'"{value}"'])
            elif isinstance(value, list):
                array_str = ",".join(value)
                adb_command.extend(["--esa", key, f'"{array_str}"'])
            else:
                raise ValueError(f"Unrecognized extra type for {key}")

    return device_controller.run_shell_command(adb_command, timeout=timeout_sec)


def display_message(message: str, device_controller: Device, header: str = "") -> None:
    send_android_intent(
        command="broadcast",
        action="com.example.ACTION_UPDATE_OVERLAY",
        device_controller=device_controller,
        extras={"task_type_string": header, "goal_string": message},
    )


def execute_adb_action(
    action: json_action.JSONAction,
    device_controller: Device,
    screen_elements: list[Any] = None,  # list[UIElement]
    screen_size: tuple[int, int] = None,  # (width, height)
) -> None:
    """Execute an action based on a JSONAction object.

    Args:
        action: JSONAction object containing the action to be executed.
        screen_elements: List of UI elements on the screen.
        screen_size: The (width, height) of the screen.
        env: The environment to execute the action in.
    """
    if action.action_type == json_action.ANSWER:
        if action.text:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": action.text,
                }
            )
            display_message(
                action.text,
                header="Agent answered:",
                device_controller=device_controller,
            )
        return
    if action.action_type in ["click", "double_tap", "long_press"]:
        idx = action.index
        x = action.x
        y = action.y
        if idx is not None and screen_elements is not None:
            if idx < 0 or idx >= len(screen_elements):
                raise ValueError(
                    f"Invalid element index: {idx}, must be between 0 and"
                    f" {len(screen_elements)-1}."
                )
            element = screen_elements[idx]
            if element.bbox_pixels is None:
                raise ValueError("Bbox is not present on element.")
            x, y = element.bbox_pixels.center
            x, y = int(x), int(y)
            if action.action_type == "click":
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Click on element {idx} at ({x}, {y}).",
                    }
                )
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Double click on element {idx} at ({x}, {y}).",
                    }
                )
                device_controller.double_click(x, y)
            else:
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Long press on element {idx} at ({x}, {y}).",
                    }
                )
                device_controller.long_click(x, y)
        elif x is not None and y is not None:
            if action.action_type == "click":
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Click on screen at ({x}, {y}).",
                    }
                )
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Double click on screen at ({x}, {y}).",
                    }
                )
                device_controller.double_click(x, y)
            else:
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": f"Long press on screen at ({x}, {y}).",
                    }
                )
                device_controller.long_click(x, y)
        else:
            raise ValueError(f"Invalid click action: {action}")
        return x, y

    elif action.action_type == "input_text":
        text = action.text
        if text:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": f"Input text: '{text}'.",
                }
            )
            device_controller.input_text(text, smart_enter=False, clear_first=False)
        else:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "MLLM responded with an invalid action. Retrying...",
                }
            )
            logging.warning(
                "Input_text action indicated, but no text provided. No "
                "action will be executed."
            )

    elif action.action_type == "keyboard_enter":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Press enter button.",
            }
        )
        device_controller.enter()

    elif action.action_type == "navigate_home":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Press home button.",
            }
        )
        device_controller.home()

    elif action.action_type == "navigate_back":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Press back button.",
            }
        )
        device_controller.back()

    elif action.action_type == "scroll":
        screen_width, screen_height = screen_size
        if action.index:
            x_min, y_min, x_max, y_max = (
                max(screen_elements[action.index].bbox_pixels.x_min, 0),
                max(screen_elements[action.index].bbox_pixels.y_min, 0),
                min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
                min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
            )
        else:
            x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

        start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        direction = action.direction
        if direction == "down":
            end_x, end_y = (x_min + x_max) // 2, y_min
        elif direction == "up":
            end_x, end_y = (x_min + x_max) // 2, y_max
        elif direction == "right":
            end_x, end_y = x_min, (y_min + y_max) // 2
        elif direction == "left":
            end_x, end_y = x_max, (y_min + y_max) // 2
        else:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "MLLM responded with an invalid action. Retrying...",
                }
            )
            print("Invalid direction")
            return
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": f"Scroll {direction} from ({start_x}, {start_y}) to ({end_x}, {end_y}).",
            }
        )
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "swipe":  # Inverse of scroll.
        screen_width, screen_height = screen_size
        mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
        direction = action.direction
        if direction == "down":
            start_x, start_y = mid_x, 0
            end_x, end_y = mid_x, screen_height
        elif direction == "up":
            start_x, start_y = mid_x, screen_height
            end_x, end_y = mid_x, 0
        elif direction == "left":
            start_x, start_y = 0, mid_y
            end_x, end_y = screen_width, mid_y
        elif direction == "right":
            start_x, start_y = screen_width, mid_y
            end_x, end_y = 0, mid_y
        else:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "MLLM responded with an invalid action. Retrying...",
                }
            )
            print("Invalid direction")
            return
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": f"Swipe {direction} from ({start_x}, {start_y}) to ({end_x}, {end_y}).",
            }
        )
        device_controller.swipe(
            int(start_x), int(start_y), int(end_x), int(end_y), duration=0.5
        )
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "open_app":
        app_name = action.app_name
        if app_name:
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": f"Open app: {app_name}.",
                }
            )
            launch_app(app_name, device_controller)
        else:
            raise ValueError("No app name provided")

    elif action.action_type == "wait":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "Wait for 2 seconds.",
            }
        )
        time.sleep(2.0)

    elif action.action_type == "launch_adb_activity":
        if action.activity_nickname == "app_drawer":
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "Open app drawer.",
                }
            )
            device_controller.home()
            time.sleep(1.0)
            start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            device_controller.swipe(start_x, start_y, end_x, end_y)
        elif action.activity_nickname == "quick_settings":
            start_x, start_y = int(screen_size[0] / 2), 30
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "Open quick settings.",
                }
            )
            device_controller.swipe(start_x, start_y, end_x, end_y, duration=0.1)
    elif action.action_type == "change_orientation":
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": f"Change orientation to {action.orientation}.",
            }
        )
        change_orientation(action.orientation, device_controller)
    elif action.action_type == json_action.UNKNOWN:
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "MLLM responded with an invalid action. Retrying...",
            }
        )
        print("Unknown action type; no action will be executed. Try again...")
    else:
        send_message(
            {
                "message_type": "action",
                "display_type": "text",
                "message": "MLLM responded with an invalid action. Retrying...",
            }
        )
        print("Invalid action type")


def launch_app(app_name: str, device_controller: Device) -> Optional[str]:
    """Uses regex and ADB activity to try to launch an app.

    Args:
      app_name: The name of the app, as represented as a key in
        _PATTERN_TO_ACTIVITY.
      device_controller: The device controller to execute the command.

    Returns:
      The name of the app that is launched.
    """
    # Maps app names to the activity that should be launched to open the app.
    _PATTERN_TO_ACTIVITY = {
        "google chrome|chrome": (
            "com.android.chrome/com.google.android.apps.chrome.Main"
        ),
        "google chat": "com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity",
        "settings|system settings": "com.android.settings/.Settings",
        "youtube|yt": "com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity",
        "google play|play store|gps": (
            "com.android.vending/com.google.android.finsky.activities.MainActivity"
        ),
        "gmail|gemail|google mail|google email|google mail client": (
            "com.google.android.gm/.ConversationListActivityGmail"
        ),
        "google maps|gmaps|maps|google map": (
            "com.google.android.apps.maps/com.google.android.maps.MapsActivity"
        ),
        "google photos|gphotos|photos|google photo|google pics|google images": "com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity",
        "google calendar|gcal": (
            "com.google.android.calendar/com.android.calendar.AllInOneActivity"
        ),
        "camera": "com.android.camera2/com.android.camera.CameraLauncher",
        "audio recorder": "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity",
        "google drive|gdrive|drive": (
            "com.google.android.apps.docs/.drive.startup.StartupActivity"
        ),
        "google keep|gkeep|keep": (
            "com.google.android.keep/.activities.BrowseActivity"
        ),
        "grubhub": (
            "com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity"
        ),
        "tripadvisor": "com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity",
        "starbucks": "com.starbucks.mobilecard/.main.activity.LandingPageActivity",
        "google docs|gdocs|docs": "com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google sheets|gsheets|sheets": "com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google slides|gslides|slides": "com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "clock": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "google search|google": "com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity",
        "contacts": "com.google.android.contacts/com.android.contacts.activities.PeopleActivity",
        "facebook|fb": "com.facebook.katana/com.facebook.katana.LoginActivity",
        "whatsapp|wa": "com.whatsapp/com.whatsapp.Main",
        "instagram|ig": (
            "com.instagram.android/com.instagram.mainactivity.MainActivity"
        ),
        "twitter|tweet": "com.twitter.android/com.twitter.app.main.MainActivity",
        "snapchat|sc": "com.snapchat.android/com.snap.mushroom.MainActivity",
        "telegram|tg": "org.telegram.messenger/org.telegram.ui.LaunchActivity",
        "linkedin": (
            "com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity"
        ),
        "spotify|spot": "com.spotify.music/com.spotify.music.MainActivity",
        "netflix": "com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity",
        "amazon shopping|amazon|amzn": (
            "com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity"
        ),
        "tiktok|tt": "com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity",
        "discord": "com.discord/com.discord.app.AppActivity$Main",
        "reddit": "com.reddit.frontpage/com.reddit.frontpage.MainActivity",
        "pinterest": "com.pinterest/com.pinterest.activity.PinterestActivity",
        "android world": "com.example.androidworld/.MainActivity",
        "files": "com.google.android.documentsui/com.android.documentsui.files.FilesActivity",
        "markor": "net.gsantner.markor/net.gsantner.markor.activity.MainActivity",
        "clipper": "ca.zgrs.clipper/ca.zgrs.clipper.Main",
        "messages": "com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity",
        "simple sms messenger|simple sms|sms messenger": "com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity",
        "dialer|phone": "com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity",
        "simple calendar pro|simple calendar": "com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity",
        "simple gallery pro|simple gallery": "com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity",
        "miniwob": "com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity",
        "simple draw pro": "com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity",
        "pro expense|pro expense app": (
            "com.arduia.expense/com.arduia.expense.ui.MainActivity"
        ),
        "broccoli|broccoli app|broccoli recipe app|recipe app": (
            "com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity"
        ),
        "caa|caa test|context aware access": "com.google.ccc.hosted.contextawareaccess.thirdpartyapp/.ChooserActivity",
        "osmand": "net.osmand/net.osmand.plus.activities.MapActivity",
        "tasks|tasks app|tasks.org:": (
            "org.tasks/com.todoroo.astrid.activity.MainActivity"
        ),
        "open tracks sports tracker|activity tracker|open tracks|opentracks": (
            "de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity"
        ),
        "joplin|joplin app": "net.cozic.joplin/.MainActivity",
        "vlc|vlc app|vlc player": "org.videolan.vlc/.gui.MainActivity",
        "retro music|retro|retro player": (
            "code.name.monkey.retromusic/.activities.MainActivity"
        ),
    }

    def get_adb_activity(app_name: str) -> Optional[str]:
        """Get a mapping of regex patterns to ADB activities top Android apps."""
        for pattern, activity in _PATTERN_TO_ACTIVITY.items():
            if re.match(pattern.lower(), app_name.lower()):
                return activity

    # Special app names that will trigger opening the default app.
    _DEFAULT_URIS: dict[str, str] = {
        "calendar": "content://com.android.calendar",
        "browser": "http://",
        "contacts": "content://contacts/people/",
        "email": "mailto:",
        "gallery": "content://media/external/images/media/",
    }
    if app_name in _DEFAULT_URIS:
        data_uri = _DEFAULT_URIS[app_name]
        adb_command = [
            #'shell',
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            data_uri,
        ]
        device_controller.run_shell_command(adb_command)
        time.sleep(1.0)  # Wait for the app to launch.
        return app_name

    activity = get_adb_activity(app_name)
    if activity is None:
        logging.error("Failed to launch app: %r", app_name)
        return None
    # start_activity(activity, extra_args=[], env=env, timeout_sec=5)
    args = [
        "am",
        "start",
        "-a",
        "android.intent.action.MAIN",
        "-c",
        "android.intent.category.LAUNCHER",
        "-n",
        activity,
    ]
    device_controller.run_shell_command(args)
    time.sleep(1.0)  # Wait for the app to launch.
    return app_name


def change_orientation(orientation: str, device_controller: Device) -> None:
    """Changes the screen orientation.

    Args:
      orientation: str, The new orientation. Can be portrait, landscape,
        reverse_portrait, or reverse_landscape.
      device_controller: The device controller to execute the command.

    Raises:
      ValueError if invalid orientation is provided.
    """
    _ORIENTATIONS = {
        "portrait": "0",
        "landscape": "1",
        "portrait_reversed": "2",
        "landscape_reversed": "3",
    }
    if orientation not in _ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation provided: {orientation} not in"
            f" {_ORIENTATIONS.keys()}"
        )
    command = [
        # "shell",
        "settings",
        "put",
        "system",
    ]
    # Turn off accelerometer.
    device_controller.run_shell_command(command + ["accelerometer_rotation", "0"])
    device_controller.run_shell_command(
        command + ["user_rotation", _ORIENTATIONS[orientation]]
    )


from utils.utils import save_object_to_disk, load_object_from_disk, str_to_md5
from datetime import datetime


class GUI_explorer(object):
    def __init__(
        self,
        device_serial: str = None,
        step_interval: float = 2.0,
        rag_top_k: int = None,
        rag_threshold: float = None,
    ):
        self.history = []
        self.device = Device(device_serial=device_serial)
        self.step_interval = step_interval  # Wait a few seconds for the screen to stabilize after executing an action.
        self.early_stop = False  # Early stop flag for demo.
        self.demo_on = os.getenv("TURN_ON_DEMO_MODE", "False").lower() == "true"
        if rag_top_k is None:
            rag_top_k = int(os.getenv("RAG_TOP_K", 1))
        self.rag_top_k = rag_top_k
        if rag_threshold is None:
            rag_threshold = float(os.getenv("RAG_THRESHOLD", 0.9))
        self.rag_threshold = rag_threshold

    def reset(self, go_home_on_reset: bool = False):
        # Hide the coordinates on screen which might affect the vision model.
        if go_home_on_reset:
            self.device.home()
        self.history = []

    def run(
        self,
        task_goal: str,
        max_rounds: int = 30,
        step_interval: float = 2.0,
        step_data_output_dir: str = "./tmp",
    ):
        self.reset()
        self.step_interval = step_interval
        step_datas = []
        os.makedirs(step_data_output_dir, exist_ok=True)
        for i in range(max_rounds):
            if self.early_stop:
                break
            self.early_stop = is_need_stop()
            if self.early_stop:
                break
            print(f"Round {i + 1}/{max_rounds}")
            stop, step_data = self.step(task_goal)
            step_datas.append(step_data)
            if self.early_stop:
                break
            self.early_stop = is_need_stop()
            if self.early_stop:
                break
            if stop:
                break
            # time.sleep(self.step_interval)
            self.device.wait_to_stabilize()
        output_path = os.path.join(
            step_data_output_dir,
            f"{datetime.now().strftime('%y%m%d%H%M%S.%f')}_{str_to_md5(task_goal)[:16]}.pkl.zst",
        )
        save_object_to_disk(step_datas, output_path, compress_level=20)
        self.early_stop = False  # Reset early stop flag for next run.
        while is_need_stop():
            time.sleep(0.1)  # 消耗完多余（由于没能及时停止导致用户多次发送）的stop信号
        send_message(
            {
                "message_type": "done",
                "display_type": "text",
                "message": "Done.",
            }
        )
        send_message2(
            {
                "message_type": "done",
                "display_type": "text",
                "message": "Done.",
            }
        )
        print("Done.")
        return step_datas

    def step(self, goal: str) -> tuple[bool, dict[str, Any]]:
        step_data = {
            "goal": goal,
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
            "converted_action": "error_retry",
            "actual_action_coordinates": None,
            "before_screenshot": None,
            "after_screenshot": None,
            "ui_elements": None,
            "top_app_package_name": None,
            "target_element": None,
            "ranker_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            "logical_screen_size": None,
        }
        print("Step: " + str(len(self.history) + 1))

        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)
        before_ui_elements = self.device.wait_to_stabilize()
        orientation = self.device.get_orientation()
        logical_screen_size = self.device.get_screen_size()
        step_data["logical_screen_size"] = logical_screen_size
        physical_frame_boundary = self.device.get_physical_frame_boundary()

        step_data["ui_elements"] = [
            asdict(ui_element) for ui_element in before_ui_elements
        ]
        before_ui_elements_list = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        pil_before_screenshot = self.device.get_screenshot().convert("RGB")
        before_screenshot = np.array(pil_before_screenshot)
        step_data["raw_screenshot"] = before_screenshot.copy()
        step_data["before_screenshot"] = before_screenshot.copy()
        knowledge_prompt = ""
        all_knowledge = []  # NOTE:按照index排序的知识
        cropped_pils = []  # NOTE:记录切片后的图片的PIL对象，以便批量检索
        cropped_idxs = []  # NOTE:记录切片后的图片的索引
        # top_app_package_name = self.env.get_top_app_package_name()
        top_app_package_name = self.device.get_top_package_name()
        step_data["top_app_package_name"] = top_app_package_name
        for index, ui_element in enumerate(before_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
                if ui_element.bbox_pixels:
                    upper_left_logical, lower_right_logical = (
                        _ui_element_logical_corner(ui_element, orientation)
                    )
                    upper_left_physical = _logical_to_physical(
                        upper_left_logical,
                        logical_screen_size,
                        physical_frame_boundary,
                        orientation,
                    )
                    lower_right_physical = _logical_to_physical(
                        lower_right_logical,
                        logical_screen_size,
                        physical_frame_boundary,
                        orientation,
                    )
                    cropped_pil = pil_before_screenshot.crop(
                        (
                            upper_left_physical[0],
                            upper_left_physical[1],
                            lower_right_physical[0],
                            lower_right_physical[1],
                        )
                    )
                    cropped_pils.append(cropped_pil)
                    cropped_idxs.append(index)

        # NOTE:检索知识库
        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)
        retrieval_result = retrieval_batch_api(
            queries=cropped_pils,
            top_k=self.rag_top_k,
            threshold=self.rag_threshold,
            package_name=top_app_package_name,
        )
        all_knowledge = []
        for idx, res in zip(cropped_idxs, retrieval_result):
            if res:
                hints = {}
                for i, item in enumerate(res):
                    hints[f"hint_{i+1}"] = item["knowledge"]
                all_knowledge.append(
                    {
                        "index": idx,
                        "hints": hints,
                    }
                )
        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)

        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        # NOTE: Generate guidance for the current task
        prioritized_knowledge = all_knowledge
        if not self.demo_on:
            prioritized_knowledge = Ranker(
                goal, all_knowledge, step_data["ranker_usage"]
            )
        knowledge_prompt_demo = ""
        for item in prioritized_knowledge:
            knowledge_prompt += f'\nUI element {item["index"]}: {item["hints"]["hint_1"] if len(item["hints"]) ==1 else item["hints"]}'
            knowledge_prompt_demo += knowledge_prompt

        if len(knowledge_prompt) > 0:
            send_message2(
                {
                    "message_type": "knowledge",
                    "display_type": "text",
                    "message": knowledge_prompt_demo.removeprefix("\n").strip(),
                }
            )
            knowledge_prompt = f"\nHere are some tips for you:{knowledge_prompt}\n"
        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)
        action_prompt = _action_selection_prompt(
            goal,
            [
                "Step " + str(i + 1) + "- " + step_info["summary"]
                for i, step_info in enumerate(self.history)
            ],
            before_ui_elements_list,
            knowledge_prompt=knowledge_prompt,
        )
        step_data["action_prompt"] = action_prompt
        
        # 格式错误重试机制：最多重试5次
        max_format_retries = 5
        reason = None
        action = None
        converted_action = None
        
        for format_retry in range(max_format_retries):
            action_output, raw_response = ask_mllm(
                action_prompt,
                [
                    step_data["raw_screenshot"],
                    before_screenshot,
                ],
            )

            # 检查 raw_response 是否为空
            if not raw_response:
                if format_retry < max_format_retries - 1:
                    print(f"LLM call failed, retrying ({format_retry + 1}/{max_format_retries})...")
                    continue
                else:
                    print(f"LLM call failed after {max_format_retries} retries.")
                    step_data["summary"] = "LLM call failed after max retries."
                    self.history.append(step_data)
                    return (False, step_data)
            
            step_data["action_output"] = action_output
            step_data["action_raw_response"] = raw_response

            # 检查 action_output 是否为空
            if not action_output:
                if format_retry < max_format_retries - 1:
                    print(f"LLM returned empty content, retrying ({format_retry + 1}/{max_format_retries})...")
                    continue
                else:
                    print(f"LLM returned empty content after {max_format_retries} retries.")
                    step_data["summary"] = "LLM returned empty content after max retries."
                    self.history.append(step_data)
                    return (False, step_data)

            reason, action = parse_reason_action_output(action_output)

            # 检查解析结果格式是否正确
            if (not reason) or (not action):
                if format_retry < max_format_retries - 1:
                    print(f"Action prompt output format error, retrying ({format_retry + 1}/{max_format_retries})...")
                    continue
                else:
                    print(f"Action prompt output format error after {max_format_retries} retries.")
                    step_data["summary"] = (
                        "Output for action selection is not in the correct format after max retries."
                    )
                    self.history.append(step_data)
                    return (False, step_data)

            # 尝试解析 JSON action
            try:
                converted_action = json_action.JSONAction(
                    **extract_json(action),
                )
                step_data["converted_action"] = converted_action
            except Exception as e:  # pylint: disable=broad-exception-caught
                if format_retry < max_format_retries - 1:
                    print(f"Failed to parse JSON action: {e}, retrying ({format_retry + 1}/{max_format_retries})...")
                    continue
                else:
                    print(f"Failed to parse JSON action after {max_format_retries} retries: {e}")
                    step_data["summary"] = (
                        "Can not parse the output to a valid action after max retries."
                    )
                    self.history.append(step_data)
                    step_data["converted_action"] = "error_retry"
                    send_message(
                        {
                            "message_type": "action",
                            "display_type": "text",
                            "message": "MLLM responded with an invalid action after max retries.",
                        }
                    )
                    return (False, step_data)

            # 检查索引是否越界
            if (
                converted_action.action_type
                in ["click", "long_press", "input_text", "scroll"]
                and converted_action.index is not None
            ):
                if converted_action.index >= len(before_ui_elements):
                    if format_retry < max_format_retries - 1:
                        print(f"Index out of range, retrying ({format_retry + 1}/{max_format_retries})...")
                        continue
                    else:
                        print(f"Index out of range after {max_format_retries} retries.")
                        step_data["summary"] = (
                            "The parameter index is out of range after max retries."
                        )
                        self.history.append(step_data)
                        step_data["converted_action"] = "error_retry"
                        send_message(
                            {
                                "message_type": "action",
                                "display_type": "text",
                                "message": "MLLM responded with an invalid action after max retries.",
                            }
                        )
                        return (False, step_data)
            
            # 成功解析，跳出重试循环
            break
        
        # 打印解析成功的结果
        print("Reasoning: " + reason)
        print("Action: " + action)
        send_message2(
            {
                "message_type": "reasoning",
                "display_type": "text",
                "message": reason.strip(),
            }
        )

        if (
            converted_action.action_type
            in ["click", "long_press", "input_text", "scroll"]
            and converted_action.index is not None
        ):
            # Add mark to the target element.
            add_ui_element_mark(
                step_data["raw_screenshot"],
                before_ui_elements[converted_action.index],
                converted_action.index,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
            step_data["target_element"] = asdict(
                before_ui_elements[converted_action.index]
            )

        if converted_action.action_type == "status":
            step_data["summary"] = "Agent thinks the request has been completed."
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
                step_data["summary"] = (
                    "Agent thinks the mission is infeasible and stopped."
                )
            self.history.append(step_data)
            if converted_action.goal_status == "infeasible":
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": "Task infeasible.",
                    }
                )
            else:
                send_message(
                    {
                        "message_type": "action",
                        "display_type": "text",
                        "message": "Task completed.",
                    }
                )
            return (True, step_data)  # complete和infeasible都返回True，表示任务结束

        if converted_action.action_type == "answer":
            print("Agent answered with: " + converted_action.text)

        try:
            self.early_stop = is_need_stop()
            if self.early_stop:
                return (True, step_data)
            actual_action_coordinates = execute_adb_action(
                converted_action,
                self.device,
                before_ui_elements,
                logical_screen_size,
            )
            step_data["actual_action_coordinates"] = actual_action_coordinates
            self.early_stop = is_need_stop()
            if self.early_stop:
                return (True, step_data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            step_data["converted_action"] = "error_retry"
            send_message(
                {
                    "message_type": "action",
                    "display_type": "text",
                    "message": "MLLM responded with an invalid action. Retrying...",
                }
            )
            return (False, step_data)

        # time.sleep(self.step_interval)
        self.device.wait_to_stabilize()
        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)

        orientation = self.device.get_orientation()
        logical_screen_size = self.device.get_screen_size()
        physical_frame_boundary = self.device.get_physical_frame_boundary()

        after_ui_elements = self.device._get_ui_elements()
        after_ui_elements_list = _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        after_screenshot = np.array(self.device.get_screenshot())
        step_data["after_screenshot"] = after_screenshot.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        add_screenshot_label(step_data["before_screenshot_with_som"], "before")
        add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            before_ui_elements_list,
            after_ui_elements_list,
        )
        self.early_stop = is_need_stop()
        if self.early_stop:
            return (True, step_data)
        summary, raw_response = ask_mllm(
            summary_prompt,
            [
                before_screenshot,
                after_screenshot,
            ],
        )

        if not raw_response:
            step_data["summary"] = (
                "Some error occurred calling LLM during summarization phase."
            )
            self.history.append(step_data)
            return (False, step_data)

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        print("Summary: " + summary)
        send_message(
            {
                "message_type": "summary",
                "display_type": "text",
                "message": summary,
            }
        )
        step_data["summary_raw_response"] = raw_response

        self.history.append(step_data)
        return (False, step_data)
