"""
重构后的Curriculum Generator主入口

根据exploration_output_vis_25100701中处理过的轨迹数据生成任务
输入：
1. 原始的app名字
2. 原始的任务目标
3. 原始的任务执行的每一步的截图（动作可视化之后的）
4. 任务示例作为fewshot提示
5. 任务生成的原则

输出：
1. 判断原始任务是否合理，agent是否正确完成了任务
2. 新生成的任务+对应的执行需要的步数估计
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import glob
import sys
from datetime import datetime
import concurrent.futures
import threading
import time

# Add the parent directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from trajectory_parser import TrajectoryParser
from action_visualizer import ActionVisualizer
from unified_task_processor import UnifiedTaskProcessor
from result_saver import RefactoredResultSaver
from android_world_loader import AndroidWorldTaskLoader


class RefactoredCurriculumGenerator:
    """重构后的课程生成器主类"""

    def __init__(self, vis_data_dir: str = "exploration_output_vis_25100701"):
        """
        初始化重构后的课程生成器

        Args:
            vis_data_dir: 可视化数据目录路径
        """
        self.vis_data_dir = Path(vis_data_dir)
        self.trajectory_parser = TrajectoryParser(self.vis_data_dir)
        self.action_visualizer = ActionVisualizer()
        self.unified_processor = UnifiedTaskProcessor()
        self.result_saver = RefactoredResultSaver()

        # 生成当前运行的时间戳，用于创建专属的输出目录
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 加载通用few-shot示例
        self.general_fewshot_examples = self._load_general_fewshot_examples()

        # 初始化AndroidWorld任务加载器
        android_world_excel_path = (
            Path(__file__).parent / "251103-android-world-tasks-to-app.xlsx"
        )
        self.android_world_loader = AndroidWorldTaskLoader(
            str(android_world_excel_path)
        )

        # 任务生成原则
        self.task_generation_principles = self._load_task_principles()

        # 线程安全的状态追踪
        self.lock = threading.Lock()
        self.progress_stats = {
            "processed": 0,
            "failed": 0,
            "total": 0,
            "start_time": None,
        }

        # 包名到准确应用名的映射
        self.app_name_mapping = {
            "com.android.camera2": "Camera",
            "com.android.chrome": "Chrome",
            "com.google.android.deskclock": "Clock",
            "com.google.android.contacts": "Contacts",
            "com.google.android.documentsui": "Files",
            "com.android.settings": "Settings",
            "net.gsantner.markor": "Markor",
            "com.simplemobiletools.calendar.pro": "Simple Calendar Pro",
            "org.tasks": "Tasks",
            "com.simplemobiletools.draw.pro": "Simple Draw Pro",
            "com.simplemobiletools.gallery.pro": "Simple Gallery Pro",
            "com.simplemobiletools.smsmessenger": "Simple SMS Messenger",
            "com.dimowner.audiorecorder": "Audio Recorder",
            "com.arduia.expense": "Pro Expense",
            "com.flauschcode.broccoli": "Broccoli Recipe",
            "net.osmand": "OsmAnd",
            "de.dennisguse.opentracks": "OpenTracks",
            "org.videolan.vlc": "VLC",
            "net.cozic.joplin": "Joplin",
            "code.name.monkey.retromusic": "Retro Music",
        }

    def process_app(
        self,
        app_package: str,
        vis_data_dir: str,
        output_dir: str,
        max_trajectories: int = None,
        fewshot_count: int = None,
    ) -> Dict[str, Any]:
        """
        处理指定应用的所有轨迹并生成任务 (测试用包装器)

        Args:
            app_package: 应用包名
            vis_data_dir: 可视化数据目录路径
            output_dir: 输出目录路径
            max_trajectories: 最大处理轨迹数量
            fewshot_count: few-shot示例数量

        Returns:
            处理结果摘要
        """
        # 更新数据目录
        self.vis_data_dir = Path(vis_data_dir)
        self.trajectory_parser = TrajectoryParser(self.vis_data_dir)

        # 调用实际处理方法
        results = self.process_app_trajectories(
            app_package, output_dir, max_trajectories
        )

        # 返回摘要信息
        summary = {
            "processed_trajectories": max_trajectories or 0,
            "total_generated_tasks": 0,
            "reasonable_tasks": 0,
        }

        return summary

    def process_app_trajectories(
        self, app_package: str, output_dir: str, limit: int = None, max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        顺序处理指定应用的所有轨迹并生成任务（app内部不并行）

        Args:
            app_package: 应用包名，如 "com.android.camera2"
            output_dir: 输出目录路径
            limit: 处理轨迹数量限制
            max_retries: 最大重试次数

        Returns:
            处理结果列表
        """
        print(f"\nProcessing app: {app_package}")
        print(f"Sequential processing within app, Max retries: {max_retries}")

        # Parse app information
        app_info = self.trajectory_parser.parse_app_info(app_package)
        if not app_info:
            print(f"Unable to get app info: {app_package}")
            return []

        # 优先使用映射的准确名称，否则使用 app_info 中的名称
        app_name = self.app_name_mapping.get(app_package, app_info.get("app_name", "Unknown App"))
        print(f"App name: {app_name} (from {'mapping' if app_package in self.app_name_mapping else 'app_info'})")

        # Get all trajectories
        trajectories = self.trajectory_parser.get_all_trajectories(app_package)
        if limit:
            trajectories = trajectories[:limit]

        print(f"Found {len(trajectories)} trajectories")

        # 初始化进度追踪
        with self.lock:
            self.progress_stats = {
                "processed": 0,
                "failed": 0,
                "total": len(trajectories),
                "start_time": time.time(),
            }

        # 初始化已生成任务列表，用于避免重复
        generated_instructions = []

        # 顺序处理轨迹（app内部不并行）
        all_results = []

        print(f"\nStarting sequential processing...")

        # 顺序处理每个轨迹
        for i, trajectory_file in enumerate(trajectories, 1):
            print(
                f"\nProcessing trajectory {i}/{len(trajectories)}: {trajectory_file.name}"
            )

            # 带重试机制的处理
            result = None
            for attempt in range(max_retries + 1):
                try:
                    result = self._process_single_trajectory(
                        app_package,
                        app_name,
                        trajectory_file,
                        generated_instructions.copy(),
                    )

                    if result:
                        # 更新结果列表
                        all_results.append(result)

                        # 更新已生成任务列表
                        new_tasks = result.get("generated_tasks", [])
                        for task in new_tasks:
                            generated_instructions.append(task.get("instruction", ""))

                        # 更新进度
                        self._update_progress(success=True)
                        break
                    else:
                        if attempt < max_retries:
                            print(
                                f"    Retry {attempt + 1}/{max_retries} for trajectory {trajectory_file.name}"
                            )
                            time.sleep(1)  # 短暂延迟后重试
                        continue

                except Exception as e:
                    if attempt < max_retries:
                        print(
                            f"    Retry {attempt + 1}/{max_retries} for trajectory {trajectory_file.name}: {e}"
                        )
                        time.sleep(2)  # 延迟后重试
                        continue
                    else:
                        print(
                            f"    Failed to process trajectory after {max_retries} retries: {trajectory_file.name}: {e}"
                        )
                        self._update_progress(success=False)
                        break

            if not result:
                self._update_progress(success=False)

            # 输出进度信息
            self._print_progress()

        # 输出最终统计
        self._print_final_stats()

        # Save results
        output_path = Path(output_dir)
        self.result_saver.save_app_results(
            app_package, app_name, all_results, output_path
        )

        print(
            f"\nApp {app_package} processing completed, generated {len(all_results)} results"
        )
        print(f"Total unique tasks generated: {len(generated_instructions)}")

        return all_results

    def _process_single_trajectory(
        self,
        app_package: str,
        app_name: str,
        trajectory_file: Path,
        generated_instructions: List[str],
    ) -> Optional[Dict[str, Any]]:
        """处理单条轨迹"""

        # 1. 解析轨迹数据
        trajectory_data = self.trajectory_parser.parse_trajectory(trajectory_file)
        if not trajectory_data:
            print(f"  Failed to parse trajectory data")
            return None

        trajectory_id = trajectory_data.get("trajectory_id", "")
        goal = trajectory_data.get("goal", "")
        depth = trajectory_data.get("depth", 0)
        steps = trajectory_data.get("steps", [])

        print(f"  Trajectory ID: {trajectory_id}")
        print(f"  Goal: {goal}")
        print(f"  Depth: {depth}")
        print(f"  Steps: {len(steps)}")

        # 创建轨迹专用的调试目录，使用session时间戳，放在generated_tasks下
        debug_dir = (
            Path(f"generated_tasks/debug_output_{self.session_timestamp}")
            / app_package
            / trajectory_id
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 2. 获取和可视化截图
        screenshot_info = self.trajectory_parser.get_trajectory_screenshots(
            app_package, trajectory_id
        )

        if not screenshot_info:
            print(f"  Unable to get screenshot info")
            return None

        # 3. 生成动作可视化截图
        visualized_screenshots = self.action_visualizer.create_visualized_screenshots(
            trajectory_data, screenshot_info
        )

        print(f"  Generated {len(visualized_screenshots)} visualized screenshots")

        # 保存可视化截图到调试目录
        self._save_visualized_screenshots(visualized_screenshots, debug_dir)

        # 4. 获取当前应用的AndroidWorld任务作为few-shot示例（不设置上限）
        app_specific_fewshot = self.android_world_loader.get_app_fewshot_examples(
            app_package, app_name, max_examples=None
        )

        # 合并通用few-shot示例和应用特定的AndroidWorld任务
        combined_fewshot_examples = []

        # 优先使用AndroidWorld任务（如果存在）
        if app_specific_fewshot:
            combined_fewshot_examples.extend(app_specific_fewshot)
            print(
                f"  Using {len(app_specific_fewshot)} AndroidWorld tasks as few-shot examples"
            )

        # 补充通用示例（如果AndroidWorld任务不足）
        if len(combined_fewshot_examples) < 3:
            needed_general = 3 - len(combined_fewshot_examples)
            combined_fewshot_examples.extend(
                self.general_fewshot_examples[:needed_general]
            )
            print(f"  Added {needed_general} general few-shot examples")

        print(f"  Total few-shot examples: {len(combined_fewshot_examples)}")

        # 5. 使用统一处理器进行评估和生成
        unified_result = self.unified_processor.process_task(
            app_name=app_name,
            original_goal=goal,
            visualized_screenshots=visualized_screenshots,
            fewshot_examples=combined_fewshot_examples,
            task_principles=self.task_generation_principles,
            existing_tasks=generated_instructions,
        )

        evaluation_result = unified_result.get("evaluation", {})
        generated_tasks = unified_result.get("generated_tasks", [])

        print(
            f"  Task evaluation result: {evaluation_result.get('task_reasonable', 'Unknown')}"
        )
        print(
            f"  Completion status: {evaluation_result.get('task_completed', 'Unknown')}"
        )
        print(f"  Generated {len(generated_tasks)} new tasks")

        # 保存统一处理的调试信息（包括few-shot示例信息）
        self._save_unified_debug_info(
            unified_result, debug_dir, combined_fewshot_examples
        )

        # 保存AndroidWorld调试信息
        self._save_android_world_debug_info(
            app_package, app_name, app_specific_fewshot, debug_dir
        )

        # 6. 构建结果
        result = {
            "trajectory_id": trajectory_id,
            "app_package": app_package,
            "app_name": app_name,
            "original_goal": goal,
            "depth": depth,
            "step_count": len(steps),
            "evaluation": evaluation_result,
            "generated_tasks": generated_tasks,
            "visualized_screenshots_count": len(visualized_screenshots),
            "processing_timestamp": self.result_saver.get_timestamp(),
            "debug_dir": str(debug_dir),
        }

        return result

    def _load_general_fewshot_examples(self) -> List[Dict[str, Any]]:
        """加载通用few-shot示例 - 简化版本，只包含instruction和action_description"""
        examples = [
            {
                "instruction": 'Open Novelship. Search "Addidas". Filter products to sneakers and size to 8. Sort by popularity. Select the first item. Add it to wishlist.',
                "action_descriptions": [
                    "On Pixel Home Screen, swipe up application list, to view all apps",
                    "On Pixel Home Screen, click 'Novelship' icon, to open the app",
                    "On Novelship Home Screen, click search bar, to begin a search",
                    "On Search Screen, click search field, to enter search term",
                    "On Search Screen, type 'addidas' in search field, to enter search term",
                    "On Search Screen, press enter, to view search results",
                    "On Search Results Page, click 'Filter' icon, to open filtering options",
                    "On Filter Screen, click 'Sneakers' button, to select product category",
                    "On Filter Screen, click 'Size (US)' option, to select shoe size",
                    "On Filter Screen, swipe up size list, to view more sizes",
                    "On Filter Screen, click '8' size option, to select size",
                    "On Filter Screen, click 'Apply' button, to apply filters",
                    "On Filter Screen, click 'Apply' button, to apply filters",
                    "On Search Results Page, click 'Sort by' dropdown, to open sorting options",
                    "On Search Results Page, click 'Most Popular' option, to sort by popularity",
                    "On Search Results Page, click first search result, to view product details",
                    "On Product Details Screen, click 'Wishlist' icon, to add product to wishlist",
                    "On Product Details Screen, swipe up size list, to view more sizes",
                    "On Product Details Page, click heart icon, to add product to wishlist",
                    "On Product Details Screen, complete task, successfully added item to wishlist",
                ],
            },
            {
                "instruction": "Open WhatsApp. Message Bob 'Hi there, how about a dinner tonight?'",
                "action_descriptions": [
                    "On Home Screen, swipe up application list, to view all apps",
                    "On Application List, swipe up application list, to view more apps",
                    "On Application List, click WhatsApp icon, to open WhatsApp",
                    "On WhatsApp Chats Screen, click Bob chat preview, to open chat with Bob",
                    "In Bob Chat Screen, type 'Hi there, how about a dinner tonight?' in message field, to compose a message",
                    "In Bob Chat Screen, click 'Send' icon, to send message",
                    "In Bob Chat Screen, complete task, successfully sent message",
                ],
            },
            {
                "instruction": "Open Hotels. Find accommodation in Sydney from July 20 to July 25. One adult in one room. Filter for hotels with breakfast included. Compare the first and second options.",
                "action_descriptions": [
                    "On Home Screen, swipe up application list, to view all apps",
                    "On Application List, swipe up, to view more apps",
                    "On Application List, click 'Hotels.com' icon, to open the app",
                    "On Hotels Home Screen, click 'Going to' field, to enter destination",
                    "On Destination Entry Screen, type 'Sydney' in destination field, to enter destination city",
                    "On Destination Search Results, click 'Sydney, New South Wales' item, to select the city",
                    "On Hotels Home Screen, click 'Dates' field, to select check-in and check-out dates",
                    "On Calendar Screen, swipe up, to view next month",
                    "On Calendar Screen, swipe up, to view next month",
                    "On Calendar Screen, swipe up, to view next month",
                    "On Calendar Screen, click '20' button, to select end date",
                    "On Calendar Date Picker, click '25' button, to select end date",
                    "On Calendar Date Picker, click 'Done' button, to confirm selected dates",
                    "On Hotels Home Screen, click 'Travelers' field, to adjust number of travelers",
                    "On Travelers Screen, click '+' button, to increase number of adults",
                    "On Travelers Screen, click 'Done' button, to confirm traveler details",
                    "On Hotels Home Screen, click 'Search' button, to view search results",
                    "On Hotel Search Results, click 'Sort & Filter' button, to open filtering options",
                    "In Filter Options Screen, click 'Breakfast included' checkbox, to select the filter",
                    "On Filter Options, click 'Done' button, to apply filters",
                    "On Hotel Search Results, swipe up, to view more hotels",
                    "On Hotel Search Results, complete task, successfully filtered hotels.",
                ],
            },
        ]

        return examples

    def _load_task_principles(self) -> List[str]:
        """加载任务生成原则 - 优化版本，强调AndroidWorld风格、考察点、任务复杂度和步数估算"""
        principles = [
            "AndroidWorld Style Priority: The generated tasks MUST comprehensively cover all examination points and similar tasks found in the AndroidWorld few-shot examples. If AndroidWorld examples show certain functionality, prioritize generating similar tasks that test the same capabilities.",
            "Task Complexity Priority (CRITICAL): Generate primarily medium-to-hard difficulty tasks. Simple single-operation tasks (like 'open settings', 'create one item', or 'delete one event') should constitute LESS THAN 20% of generated tasks. Prioritize: (1) Batch operations affecting multiple items, (2) Conditional filtering and actions, (3) Multi-step workflows combining search/filter/action, (4) Tasks requiring navigation through multiple screens.",
            "Accurate Step Estimation (CRITICAL): Count EVERY atomic UI interaction as one step. A 'simple' task like creating one calendar event requires 12-15 steps (open app, click +, click title field, type title, click date, navigate date picker, select date, click time, set hour, set minutes, confirm, save). Complex batch operations require 25-40 steps. NEVER underestimate step counts - minimum 8 steps for any task.",
            "Generic Action Space Testing: Follow AndroidWorld patterns for generic actions. If AndroidWorld examples include tasks like 'take a photo' or 'set a timer' without specifying exact app details, generate similar generic tasks that test basic app functionality in a straightforward manner.",
            "Agent Answer Space Testing: When AndroidWorld examples include question-answering tasks (e.g., 'What's my schedule for next Saturday afternoon in Simple Calendar app?'), prioritize generating similar tasks that require the agent to read information and provide answers based on current app state.",
            "Scroll/Drag Action Space Testing: When AndroidWorld examples involve scrolling or dragging operations, prioritize generating tasks that test the agent's ability to navigate through interfaces using scroll and drag gestures. Examples include: expense tracking tasks requiring scrolling through long lists to find specific categories or tags, calendar apps requiring swiping to navigate between months or weeks, music apps requiring scrolling through playlists, settings apps requiring scrolling to find specific configuration options.",
            "Long-press Action Space Testing: When AndroidWorld examples require long-press operations, prioritize generating tasks that test the agent's long-press interaction capabilities. Examples include: audio recording apps requiring long-press to select and rename default file names, expense apps requiring long-press to select and delete multiple expense entries, gallery apps requiring long-press to select multiple photos for batch operations, contacts apps requiring long-press to access contact editing or deletion options.",
            "Environment-Adaptive Parameter Usage: For tasks with parameters, adapt to the actual environment visible in screenshots. If AndroidWorld shows 'check yesterday's running duration in OpenTracks' but screenshots only show data from the day before yesterday, generate 'check the day before yesterday's running duration in OpenTracks'. If AndroidWorld shows 'change Li Hua's phone to 177-8888-9999' but screenshots show 'Zhang San' instead, generate 'change Zhang San's phone to 177-8888-9999'.",
            "Parameter-Free Task Replication: For AndroidWorld tasks without environment-dependent parameters (e.g., 'turn on Bluetooth'), directly replicate these tasks as they are universally executable functionality.",
            "Screenshot-Based Real Functionality: Only generate tasks for functionality actually visible in trajectory screenshots. However, trust that AndroidWorld examples reflect real app capabilities - if AndroidWorld shows certain functionality, the app definitely supports it, but current data might be limited.",
            "Core Functionality Comprehensive Coverage: Based on AndroidWorld examples, ensure comprehensive coverage of all core functionalities demonstrated. Don't just create parameter variations - focus on covering different functional areas shown in AndroidWorld tasks.",
            "Application Specification: Each task must clearly specify the target application, following AndroidWorld naming conventions (e.g., 'pro expense', 'simple calendar', etc.).",
            "Goal-Oriented Task Description: Describe what to achieve, not how to achieve it. Follow AndroidWorld's descriptive style focusing on end goals rather than step-by-step instructions.",
            "Step Length Progression: Create tasks with varying complexity (8-40 steps) representing different difficulty levels. Target distribution: 20% easy (8-12 steps), 50% medium (15-25 steps), 30% hard (25-40 steps). Step counts must be estimated by counting every atomic UI operation.",
        ]

        return principles

    def _save_visualized_screenshots(
        self, visualized_screenshots: List[Dict[str, Any]], debug_dir: Path
    ) -> None:
        """保存可视化截图到调试目录"""
        screenshots_dir = debug_dir / "visualized_screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        for i, screenshot_info in enumerate(visualized_screenshots):
            try:
                step_index = screenshot_info["step_index"]
                visualized_image = screenshot_info["visualized_image"]

                # 保存可视化截图
                image_path = screenshots_dir / f"step_{step_index:02d}_visualized.png"
                visualized_image.save(image_path)

                # 保存步骤信息
                info_path = screenshots_dir / f"step_{step_index:02d}_info.json"
                step_info = {
                    "step_index": step_index,
                    "action_type": screenshot_info.get("action_type", ""),
                    "step_summary": screenshot_info.get("step_summary", ""),
                    "action_coordinates": screenshot_info.get("action_coordinates"),
                    "target_element": screenshot_info.get("target_element"),
                }

                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(step_info, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"    Failed to save visualized screenshot {i}: {e}")

        print(f"    Visualized screenshots saved to: {screenshots_dir}")

    def _save_unified_debug_info(
        self,
        unified_result: Dict[str, Any],
        debug_dir: Path,
        fewshot_examples: List[Dict[str, Any]] = None,
    ) -> None:
        """保存统一处理的调试信息"""
        processing_dir = debug_dir / "unified_processing"
        processing_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整的统一处理结果
        result_path = processing_dir / "unified_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(unified_result, f, indent=2, ensure_ascii=False)

        # 保存评估结果
        evaluation = unified_result.get("evaluation", {})
        eval_path = processing_dir / "evaluation_result.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        # 保存生成的任务
        tasks = unified_result.get("generated_tasks", [])
        tasks_path = processing_dir / "generated_tasks.json"
        with open(tasks_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        # 保存token使用统计
        token_usage = self.unified_processor.get_token_usage()
        token_usage_path = processing_dir / "token_usage.json"
        with open(token_usage_path, "w", encoding="utf-8") as f:
            json.dump(token_usage, f, indent=2, ensure_ascii=False)

        # 保存few-shot示例信息
        if fewshot_examples:
            fewshot_path = processing_dir / "fewshot_examples.json"
            with open(fewshot_path, "w", encoding="utf-8") as f:
                json.dump(fewshot_examples, f, indent=2, ensure_ascii=False)

        # 保存LLM调用的详细调试信息
        llm_debug = self.unified_processor.get_last_llm_debug()
        if llm_debug:
            llm_debug_path = processing_dir / "llm_call_debug.json"
            with open(llm_debug_path, "w", encoding="utf-8") as f:
                json.dump(llm_debug, f, indent=2, ensure_ascii=False)

        print(f"    Unified processing debug info saved to: {processing_dir}")

    def _save_android_world_debug_info(
        self,
        app_package: str,
        app_name: str,
        app_specific_fewshot: List[Dict[str, Any]],
        debug_dir: Path,
    ) -> None:
        """保存AndroidWorld任务调试信息"""
        android_world_dir = debug_dir / "android_world"
        android_world_dir.mkdir(parents=True, exist_ok=True)

        # 保存应用特定的AndroidWorld任务
        app_fewshot_path = android_world_dir / "app_specific_tasks.json"
        with open(app_fewshot_path, "w", encoding="utf-8") as f:
            json.dump(app_specific_fewshot, f, indent=2, ensure_ascii=False)

        # 保存应用匹配信息
        match_info = {
            "app_package": app_package,
            "app_name": app_name,
            "matched_tasks_count": len(app_specific_fewshot),
            "tasks": app_specific_fewshot,
        }

        match_info_path = android_world_dir / "app_match_info.json"
        with open(match_info_path, "w", encoding="utf-8") as f:
            json.dump(match_info, f, indent=2, ensure_ascii=False)

        print(f"    AndroidWorld debug info saved to: {android_world_dir}")

    def _process_single_trajectory_parallel(
        self,
        app_package: str,
        app_name: str,
        trajectory_file: Path,
        generated_instructions: List[str],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """并行处理单条轨迹（线程安全版本）"""

        # 1. 解析轨迹数据
        trajectory_data = self.trajectory_parser.parse_trajectory(trajectory_file)
        if not trajectory_data:
            return None

        trajectory_id = trajectory_data.get("trajectory_id", "")
        goal = trajectory_data.get("goal", "")
        depth = trajectory_data.get("depth", 0)
        steps = trajectory_data.get("steps", [])

        # 创建轨迹专用的调试目录，使用session时间戳，放在generated_tasks下
        debug_dir = (
            Path(f"generated_tasks/debug_output_{self.session_timestamp}")
            / app_package
            / trajectory_id
        )
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 2. 获取和可视化截图
        screenshot_info = self.trajectory_parser.get_trajectory_screenshots(
            app_package, trajectory_id
        )

        if not screenshot_info:
            return None

        # 3. 生成动作可视化截图
        visualized_screenshots = self.action_visualizer.create_visualized_screenshots(
            trajectory_data, screenshot_info
        )

        # 保存可视化截图到调试目录
        self._save_visualized_screenshots(visualized_screenshots, debug_dir)

        # 4. 获取当前应用的AndroidWorld任务作为few-shot示例
        app_specific_fewshot = self.android_world_loader.get_app_fewshot_examples(
            app_package, app_name, max_examples=5
        )

        # 合并通用few-shot示例和应用特定的AndroidWorld任务
        combined_fewshot_examples = []

        # 优先使用AndroidWorld任务（如果存在）
        if app_specific_fewshot:
            combined_fewshot_examples.extend(app_specific_fewshot)
            print(
                f"  Using {len(app_specific_fewshot)} AndroidWorld tasks as few-shot examples"
            )

        # 补充通用示例（如果AndroidWorld任务不足）
        if len(combined_fewshot_examples) < 3:
            needed_general = 3 - len(combined_fewshot_examples)
            combined_fewshot_examples.extend(
                self.general_fewshot_examples[:needed_general]
            )
            print(f"  Added {needed_general} general few-shot examples")

        print(f"  Total few-shot examples: {len(combined_fewshot_examples)}")

        # 5. 使用统一处理器进行评估和生成
        unified_result = self.unified_processor.process_task(
            app_name=app_name,
            original_goal=goal,
            visualized_screenshots=visualized_screenshots,
            fewshot_examples=combined_fewshot_examples,
            task_principles=self.task_generation_principles,
            existing_tasks=generated_instructions,
        )

        evaluation_result = unified_result.get("evaluation", {})
        generated_tasks = unified_result.get("generated_tasks", [])

        # 保存统一处理的调试信息（包括few-shot示例信息）
        self._save_unified_debug_info(
            unified_result, debug_dir, combined_fewshot_examples
        )

        # 保存AndroidWorld调试信息
        self._save_android_world_debug_info(
            app_package, app_name, app_specific_fewshot, debug_dir
        )

        # 6. 构建结果
        result = {
            "trajectory_id": trajectory_id,
            "app_package": app_package,
            "app_name": app_name,
            "original_goal": goal,
            "depth": depth,
            "step_count": len(steps),
            "evaluation": evaluation_result,
            "generated_tasks": generated_tasks,
            "visualized_screenshots_count": len(visualized_screenshots),
            "processing_timestamp": self.result_saver.get_timestamp(),
            "debug_dir": str(debug_dir),
        }

        return result

    def _update_progress(self, success: bool) -> None:
        """更新进度统计（线程安全）"""
        with self.lock:
            if success:
                self.progress_stats["processed"] += 1
            else:
                self.progress_stats["failed"] += 1

    def _print_progress(self) -> None:
        """打印当前进度（线程安全）"""
        with self.lock:
            processed = self.progress_stats["processed"]
            failed = self.progress_stats["failed"]
            total = self.progress_stats["total"]
            completed = processed + failed

            if total > 0:
                percent = (completed / total) * 100
                success_rate = (processed / completed) * 100 if completed > 0 else 0

                # 计算预估时间
                elapsed_time = time.time() - self.progress_stats["start_time"]
                if completed > 0:
                    avg_time_per_task = elapsed_time / completed
                    remaining_tasks = total - completed
                    estimated_remaining = avg_time_per_task * remaining_tasks
                    eta_minutes = estimated_remaining / 60

                    print(
                        f"Progress: {completed}/{total} ({percent:.1f}%) | "
                        f"Success: {processed} ({success_rate:.1f}%) | "
                        f"Failed: {failed} | "
                        f"ETA: {eta_minutes:.1f}min"
                    )
                else:
                    print(f"Progress: {completed}/{total} ({percent:.1f}%)")

    def _print_final_stats(self) -> None:
        """打印最终统计信息"""
        with self.lock:
            processed = self.progress_stats["processed"]
            failed = self.progress_stats["failed"]
            total = self.progress_stats["total"]

            elapsed_time = time.time() - self.progress_stats["start_time"]

            print(f"\n{'=' * 60}")
            print(f"FINAL PROCESSING STATISTICS")
            print(f"{'=' * 60}")
            print(f"Total trajectories: {total}")
            print(f"Successfully processed: {processed}")
            print(f"Failed: {failed}")
            print(
                f"Success rate: {(processed / total) * 100:.1f}%" if total > 0 else "0%"
            )
            print(f"Total time: {elapsed_time / 60:.1f} minutes")
            print(
                f"Average time per trajectory: {elapsed_time / total:.1f} seconds"
                if total > 0
                else "N/A"
            )
            print(f"{'=' * 60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="重构后的课程生成器 - 基于exploration_output_vis数据"
    )

    parser.add_argument(
        "--vis_data_dir",
        type=str,
        default="./exploration_output_vis",
        help="可视化数据目录路径",
    )

    parser.add_argument(
        "--app_package",
        type=str,
        nargs="+",
        default=["all"],
        help="要处理的应用包名，可以是一个或多个，或使用 'all' 处理所有应用（默认）",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_tasks_refactored",
        help="输出目录路径",
    )

    parser.add_argument("--limit", type=int, default=None, help="限制处理的轨迹数量")

    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=10,
        help="并行处理的工作线程数量 (默认: 10)",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=1000,
        help="处理失败时的最大重试次数 (默认: 1000)",
    )

    args = parser.parse_args()

    # 检查数据目录是否存在
    if not Path(args.vis_data_dir).exists():
        print(f"Error: Data directory does not exist: {args.vis_data_dir}")
        return

    # 创建重构后的课程生成器
    generator = RefactoredCurriculumGenerator(args.vis_data_dir)

    # 创建带时间戳的输出目录，在generated_tasks下
    timestamped_output_dir = (
        f"generated_tasks/{args.output_dir}_{generator.session_timestamp}"
    )
    loaded_vars = vars(args)
    print(f"\n环境变量已加载:")
    for key, value in loaded_vars.items():
        print(f"  - {key}: {value}")
    print(f"Session timestamp: {generator.session_timestamp}")
    print(f"Output directory: {timestamped_output_dir}")

    # --- 统一处理所有、多个或单个应用 ---
    app_packages_to_process = []
    all_apps_mode = "all" in args.app_package

    if all_apps_mode:
        # "all" 模式下，获取所有应用目录
        print("Processing all available apps...")
        data_dir = Path(args.vis_data_dir)
        app_dirs = [
            d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        app_packages_to_process = [d.name for d in app_dirs]

        if not app_packages_to_process:
            print("No app directories found!")
            return

        print(f"Found {len(app_packages_to_process)} apps to process:")
        for app_package in app_packages_to_process:
            print(f"  - {app_package}")
    else:
        # 处理指定应用列表
        app_packages_to_process = args.app_package
        print(f"Processing {len(app_packages_to_process)} specified app(s):")
        for app_package in app_packages_to_process:
            print(f"  - {app_package}")

    # 设置并行处理
    if all_apps_mode:
        # 如果是all模式，可以开启与app数量相等的worker
        app_parallel_workers = len(app_packages_to_process)
    else:
        # 否则，尊重用户的parallel_workers设置，但不能超过任务数
        app_parallel_workers = min(len(app_packages_to_process), args.parallel_workers)

    print(f"Using {app_parallel_workers} workers for app-level parallelism.")

    def process_single_app(app_package: str, index: int) -> None:
        """处理单个应用的函数"""
        app_dir = Path(args.vis_data_dir) / app_package
        if not app_dir.exists() or not app_dir.is_dir():
            print(
                f"\nWarning: App directory for '{app_package}' not found or is not a directory. Skipping."
            )
            return

        print(f"\n{'=' * 60}")
        print(f"Processing app {index}/{len(app_packages_to_process)}: {app_package}")
        print(f"{'=' * 60}")

        try:
            generator.process_app_trajectories(
                app_package, timestamped_output_dir, args.limit, args.max_retries
            )
        except Exception as e:
            print(f"Failed to process app {app_package}: {e}")
            import traceback

            traceback.print_exc()

    # 使用ThreadPoolExecutor进行app之间的并行处理
    if len(app_packages_to_process) > 0:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=app_parallel_workers
        ) as executor:
            # 提交所有app处理任务
            future_to_app = {
                executor.submit(process_single_app, app_package, i): (
                    app_package,
                    i,
                )
                for i, app_package in enumerate(app_packages_to_process, 1)
            }

            # 等待所有app处理完成
            for future in concurrent.futures.as_completed(future_to_app):
                app_package, index = future_to_app[future]
                try:
                    future.result()
                    print(f"App {app_package} processing completed")
                except Exception as e:
                    print(f"Unexpected error processing app {app_package}: {e}")

        # 生成主汇总报告
        print(f"\n{'=' * 60}")
        print("Generating master summary report...")
        print(f"{'=' * 60}")
        generator.result_saver.create_master_summary(Path(timestamped_output_dir))


if __name__ == "__main__":
    main()
