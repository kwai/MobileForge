import os
import sys
import json
import time
import re
from utils.utils import (
    get_apk,
    APK,
    str_to_md5,
    openai_request,
    resize_pil_image,
    pil_to_webp_base64,
    load_object_from_disk,
    save_object_to_disk,
)
from utils.device import Device, UIElement
from MLLM_Agent.GUI_explorer import GUI_explorer, execute_adb_action
from PIL import Image
from utils.prompt_templates import TASK_GOAL_GENERATOR
from datetime import datetime
from glob import glob
from utils.memory import load_knowledge_raw_data, load_memory
from utils.knowledge_generation import update_trajectory_to_knowledge
from tqdm import tqdm


def parse_task(text: str) -> list[str]:
    pattern = r"\d+\.+(.*)"
    matches = re.findall(pattern, text)
    for i, match in enumerate(matches):
        matches[i] = match.strip()
    return matches


def task_goal_generator(
    screenshot: Image.Image,
    package_name: str = None,
    app_name: str = None,
    activity_list: str = None,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> list[str]:
    p = TASK_GOAL_GENERATOR.format(
        package_name=package_name if package_name else "Not Available",
        app_name=app_name if app_name else "Not Available",
        activity_list=activity_list if activity_list else "Not Available",
    )
    # print(f"Prompt: \n{p}")
    low_resolution = os.getenv("LOW_RESOLUTION", "False").lower() == "true"
    if low_resolution:
        screenshot = resize_pil_image(screenshot, 1000)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{pil_to_webp_base64(screenshot)}",
                    },
                },
                {"type": "text", "text": p},
            ],
        },
    ]
    rsp_txt = openai_request(
        messages=messages, timeout=300, usage=usage, max_tokens=8192
    )
    return parse_task(rsp_txt)


def is_task_explored(root_dir: str, task: str) -> bool:
    t = str_to_md5(task)[:16]
    res = glob(os.path.join(root_dir, "**", f"*{t}*"), recursive=True)
    return len(res) > 0


def explore_dfs(
    current_task: str,
    current_depth: int,
    exploration_output_root_dir: str,
    max_exploration_tasks: int,
    max_exploration_steps: int,
    apk_object: APK,
    device_controller: Device,
    agent: GUI_explorer,
    previous_actions: list,
    package_name: str,
    is_first_task: bool = False,
    max_exploration_depth: int = 3,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    if (
        not is_first_task
    ):  # 如果不是第一个任务，那么需要根据之前的动作序列来复现到父节点的结束状态
        print(f"{'*' * current_depth}Restore To Parent State")
        device_controller.stop_all_apps()
        device_controller.launch_app(package_name, front=True)
        for action in previous_actions:
            converted_action, before_ui_elements, logical_screen_size = action
            time.sleep(3)
            device_controller.wait_to_stabilize()
            execute_adb_action(
                converted_action,
                device_controller,
                before_ui_elements,
                logical_screen_size,
            )
    # 运行任务
    exploration_output_root_dir = os.path.abspath(exploration_output_root_dir)
    if is_task_explored(exploration_output_root_dir, current_task):
        print(f"{'*' * current_depth}Task {current_task} already explored. Skip.")
        return
    records = agent.run(
        current_task,
        max_rounds=max_exploration_steps,
        step_data_output_dir=exploration_output_root_dir,
    )

    if current_depth >= max_exploration_depth:  # 结束探索
        return
    # 生成用于恢复到current_task结束后的状态的动作序列
    restore_actions = []
    for step_data in records:
        if isinstance(step_data["converted_action"], str):
            continue
        action = (
            step_data["converted_action"],
            [UIElement(**ui_element) for ui_element in step_data["ui_elements"]],
            step_data["logical_screen_size"],
        )
        restore_actions.append(action)
    restore_actions = previous_actions + restore_actions
    # 生成任务列表
    device_controller.wait_to_stabilize()
    screenshot = device_controller.get_screenshot()
    activity = [act for act in apk_object.get_activities() if "sdk" not in act.lower()]
    activity_str = "\n".join(activity)
    app_name = apk_object.get_app_name()
    task_list = task_goal_generator(
        screenshot=screenshot,
        package_name=package_name,
        app_name=app_name,
        activity_list=activity_str,
        usage=usage,
    )
    print(f"{'*' * current_depth}Generated {len(task_list)} tasks: {task_list}")
    if len(task_list) == 0:
        return
    exploration_output_root_dir = os.path.join(
        exploration_output_root_dir,
        f"depth_{current_depth+1}",
    )
    for i in range(max_exploration_tasks):
        if i < len(task_list):
            task = task_list[i]
            print(
                f"{'*' * current_depth}Exploring task {i+1}/{min(len(task_list),max_exploration_tasks)}: {task}"
            )
            if is_task_explored(exploration_output_root_dir, task):
                print(f"{'*' * current_depth}Task {task} already explored. Skip.")
                continue
            explore_dfs(
                current_task=task,
                current_depth=current_depth + 1,
                exploration_output_root_dir=exploration_output_root_dir,
                max_exploration_tasks=max_exploration_tasks,
                max_exploration_steps=max_exploration_steps,
                agent=agent,
                apk_object=apk_object,
                device_controller=device_controller,
                previous_actions=restore_actions,
                package_name=package_name,
                is_first_task=bool(i == 0),
                max_exploration_depth=max_exploration_depth,
                usage=usage,
            )


def auto_exploration(
    package_name: str,
    exploration_output_root_dir: str = "./output",
    device_serial: str = None,
    max_exploration_tasks: int = 10,
    max_exploration_steps: int = 30,
    max_exploration_depth: int = 5,  # 从首页开始的任务扩展深度（最多扩展 max_exploration_depth-1 代）
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    exploration_output_root_dir = os.path.abspath(exploration_output_root_dir)
    exploration_output_root_dir = os.path.join(
        exploration_output_root_dir, package_name
    )
    os.makedirs(exploration_output_root_dir, exist_ok=True)
    apk_path = os.path.join(exploration_output_root_dir, f"{package_name}.apk")
    if os.path.exists(apk_path):
        os.remove(apk_path)
    print(f"Fetching the APK file of {package_name}.")
    res = get_apk(package_name, apk_path, device_serial)
    if res == "ERROR":
        print(f"Failed to get the APK file of {package_name}.")
        sys.exit(1)
    print("Analyzing the APK file.")
    apk_object = APK(apk_path)
    app_info = {
        "app_name": apk_object.get_app_name(),
        "app_version": apk_object.get_androidversion_code(),
        "app_version_name": apk_object.get_androidversion_name(),
        "app_pkg": apk_object.get_package(),
        "app_main_activity": apk_object.get_main_activity(),
    }
    with open(
        os.path.join(exploration_output_root_dir, "app_info.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(app_info, file, indent=2, ensure_ascii=False)
    device = Device(device_serial)
    agent = GUI_explorer(device_serial=device_serial)
    print("Killing all apps.")
    device.stop_all_apps()
    # device.launch_app(package_name, front=True,activity=app_info["app_main_activity"])
    device.launch_app(package_name, front=True)
    time.sleep(5)  # 等待app启动完成
    task_list = []
    print("Generating exploration tasks.")
    device.wait_to_stabilize()
    screenshot = device.get_screenshot()
    activity = [act for act in apk_object.get_activities() if "sdk" not in act.lower()]
    activity_str = "\n".join(activity)
    app_name = apk_object.get_app_name()
    task_list = task_goal_generator(
        screenshot=screenshot,
        package_name=package_name,
        app_name=app_name,
        activity_list=activity_str,
        usage=usage,
    )
    print(f"Generated {len(task_list)} tasks: {task_list}")
    if max_exploration_tasks < len(task_list):
        task_list = task_list[:max_exploration_tasks]
    print(f"Exploring {len(task_list)} tasks: {task_list}")
    for i, task in enumerate(task_list):
        print(f"Exploring task {i+1}/{len(task_list)}: {task}")
        explore_dfs(
            current_task=task,
            current_depth=1,
            exploration_output_root_dir=exploration_output_root_dir,
            max_exploration_tasks=max_exploration_tasks,
            max_exploration_steps=max_exploration_steps,
            agent=agent,
            apk_object=apk_object,
            device_controller=device,
            previous_actions=[],
            package_name=package_name,
            is_first_task=bool(i == 0),
            max_exploration_depth=max_exploration_depth,
            usage=usage,
        )
    print("Auto exploration finished.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-package_name", help="The package name of the APK, e.g. com.android.settings"
    )
    parser.add_argument(
        "-device_serial", help="The serial number of the device, see `adb devices`"
    )
    parser.add_argument(
        "-output_dir",
        help="The directory to save the task file",
        default="./exploration_output_25100701",
    )
    parser.add_argument(
        "-max_branching_factor",
        help="The max number of tasks to explore at each node",
        default=3,
    )
    parser.add_argument(
        "-max_exploration_steps",
        help="The max number of steps to explore for each task",
        default=30,
    )
    parser.add_argument(
        "-max_exploration_depth",
        help="The max depth of exploration",
        default=5,
    )
    args = parser.parse_args()
    
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    
    print(args)
    print("Starting auto exploration.")
    auto_exploration(
        package_name=args.package_name,
        exploration_output_root_dir=args.output_dir,
        device_serial=args.device_serial,
        max_exploration_tasks=int(args.max_branching_factor),
        max_exploration_steps=int(args.max_exploration_steps),
        max_exploration_depth=int(args.max_exploration_depth),
        usage=usage,
    )
    print(f"Task goal generator token usage: {usage}")

    ## 将轨迹中的知识提取出来
    # 如果版本升级，是否清空知识库重新生成
    EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE = (
        os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE", "False").lower()
        == "false"
    )
    # 如果版本降级，是否清空知识库重新生成
    EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE = (
        os.getenv("EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE", "True").lower()
        == "true"
    )
    app_info_json_paths = glob(
        os.path.join(args.output_dir, "**", "app_info.json"), recursive=True
    )
    app_infos = {}
    for app_info_json_path in app_info_json_paths:
        with open(app_info_json_path, "r", encoding="utf-8") as f:
            app_info = json.load(f)
        app_infos[app_info["app_pkg"]] = app_info

    # knowledge_raw_data = {}
    print("Loading knowledge raw data.")
    knowledge_raw_data = load_knowledge_raw_data()
    fusion_knowledge = []
    locations = []
    for k, v in knowledge_raw_data.items():
        if k in app_infos:
            app_info = app_infos[k]
            new_app_version = app_info["app_version"]
            old_app_version = v["app_version"]
            if new_app_version > old_app_version:
                if EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_UPGRADE:
                    print(
                        f"App {k} version upgraded from {old_app_version} to {new_app_version}, clear knowledge base."
                    )
                    v["knowledge"] = []
            if new_app_version < old_app_version:
                if EMPTY_KNOWLEDGE_BASE_WHEN_VERSION_DOWNGRADE:
                    print(
                        f"App {k} version downgraded from {old_app_version} to {new_app_version}, clear knowledge base."
                    )
                    v["knowledge"] = []
        fusion_knowledge.extend(v["knowledge"])
        locations.extend([(k, i) for i in range(len(v["knowledge"]))])

    for k, app_info in app_infos.items():
        if k not in knowledge_raw_data:
            print(f"App {k} not in knowledge raw data, add it.")
            knowledge_raw_data[k] = app_info
            knowledge_raw_data[k]["knowledge"] = []

    print("Loading fusion memory. This may take a while.")
    fusion_memory = load_memory(fusion_knowledge)

    for app_info_json_path in tqdm(
        app_info_json_paths, ncols=80, desc="Extracting knowledge"
    ):
        app_info_json_dir = os.path.dirname(app_info_json_path)
        pkl_paths = glob(
            os.path.join(app_info_json_dir, "**", "*.pkl.zst"), recursive=True
        )
        for pkl_path in tqdm(
            pkl_paths,
            ncols=80,
            desc=f"Processing pkl files in {os.path.basename(app_info_json_dir)}",
            leave=False,
        ):
            trajectory_data = load_object_from_disk(pkl_path)
            update_trajectory_to_knowledge(
                trajectory_data=trajectory_data,
                locations=locations,
                fusion_memory=fusion_memory,
                knowledge_data=knowledge_raw_data,
                usage=usage,
            )
    print(f"Total token usage: {usage}")
    knowledge_base_root_path = os.path.abspath(os.getenv("KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH","./knowledge_base"))
    fp = os.path.join(knowledge_base_root_path, "knowledge_data.pkl")
    save_object_to_disk(knowledge_raw_data, fp, compress_level=20)
