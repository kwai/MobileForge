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

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import json
import os
from datetime import datetime

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import gui_owl
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import qwen3_vl
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.agents import ui_tars
from android_world.env import env_launcher
from android_world.env import interface
from android_world import multi_attempt_runner
import yaml
import pandas as pd
import numpy as np

logging.set_verbosity(logging.WARNING)

os.environ["GRPC_VERBOSITY"] = "ERROR"  # Only show errors
os.environ["GRPC_TRACE"] = "none"  # Disable tracing

_TASK_TEMPLATE_COLUMN = 'task_template'
_TASK_PROMPT_COLUMN = 'task_prompt'


class ResultSaver:
    """实时保存和更新实验结果到 JSON 文件."""

    def __init__(self, output_dir: str, session_name: str = None):
        self.output_dir = output_dir
        # 使用 session 名作为 JSON 文件名的一部分
        if session_name is None:
            session_name = os.path.basename(output_dir.rstrip('/'))
        self.session_name = session_name
        self.json_path = os.path.join(output_dir, f"{session_name}_results.json")
        self.results = {
            "experiment_info": {},
            "summary": {},
            "tasks": [],
            "by_tag": {},
            "by_difficulty": {},
            "pass_at_n": {},
        }
        self._load_existing()

    def _load_existing(self):
        """如果 JSON 文件已存在，加载它."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.results = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        self.results.setdefault("experiment_info", {})
        self.results.setdefault("summary", {})
        self.results.setdefault("tasks", [])
        self.results.setdefault("by_tag", {})
        self.results.setdefault("by_difficulty", {})
        self.results.setdefault("pass_at_n", {})

    def save(self):
        """保存结果到 JSON 文件."""
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

    def init_experiment(
        self,
        agent_name: str,
        suite_family: str,
        n_task_combinations: int,
        task_random_seed: int,
        total_tasks: int,
    ):
        """初始化实验信息."""
        self.results["experiment_info"] = {
            "agent_name": agent_name,
            "suite_family": suite_family,
            "n_task_combinations": n_task_combinations,
            "task_random_seed": task_random_seed,
            "total_tasks": total_tasks,
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }
        self.save()

    def update_summary(self, summary: dict):
        """更新汇总统计."""
        self.results["summary"] = summary
        self.save()

    def update_task_result(self, task_result: dict):
        """新增或更新单个任务结果."""
        task_results = self.results.setdefault("tasks", [])
        task_key = (
            task_result.get("task_template"),
            task_result.get("instance_id"),
        )

        for idx, existing_result in enumerate(task_results):
            existing_key = (
                existing_result.get("task_template"),
                existing_result.get("instance_id"),
            )
            if existing_key == task_key:
                task_results[idx] = task_result
                self.save()
                return

        task_results.append(task_result)
        self.save()

    def update_by_tag(self, by_tag: dict):
        """更新按标签分类的统计."""
        self.results["by_tag"] = by_tag
        self.save()

    def update_by_difficulty(self, by_difficulty: dict):
        """更新按难度分类的统计."""
        self.results["by_difficulty"] = by_difficulty
        self.save()

    def update_pass_at_n(self, pass_at_n: dict):
        """更新 pass@n 统计."""
        self.results["pass_at_n"] = pass_at_n
        self.save()

    def complete_experiment(self, final_summary: dict):
        """标记实验完成并保存最终结果."""
        self.results["experiment_info"]["status"] = "completed"
        self.results["experiment_info"]["end_time"] = datetime.now().isoformat()
        self.results["final_summary"] = final_summary
        self.save()


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path

    # Check if adb is in PATH
    import shutil

    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return adb_in_path

    raise EnvironmentError(
        "adb not found in the common Android SDK paths. Please install Android"
        " SDK and ensure adb is in one of the expected directories. If it's"
        " already installed, point to the installed location."
    )


_ADB_PATH = flags.DEFINE_string(
    "adb_path",
    _find_adb_directory(),
    "Path to adb. Set if not installed through SDK.",
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    "perform_emulator_setup",
    False,
    "Whether to perform emulator setup. This must be done once and only once"
    " before running Android World. After an emulator is setup, this flag"
    " should always be False.",
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    "console_port",
    5554,
    "The console port of the running Android device. This can usually be"
    " retrieved by looking at the output of `adb devices`. In general, the"
    " first connected device is port 5554, the second is 5556, and"
    " so on.",
)

_SUITE_FAMILY = flags.DEFINE_enum(
    "suite_family",
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    "Suite family to run. See registry.py for more information.",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    "task_random_seed", 30, "Random seed for task randomness."
)

_TASKS = flags.DEFINE_list(
    "tasks",
    None,
    "List of specific tasks to run in the given suite family. If None, run all"
    " tasks in the suite family.",
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    "n_task_combinations",
    1,
    "Number of task instances to run for each task template.",
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    "",
    "The directory to save checkpoints and resume evaluation from. If the"
    " directory contains existing checkpoint files, evaluation will resume from"
    " the latest checkpoint. If the directory is empty or does not exist, a new"
    " directory will be created.",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "./results",
    "The path to save results to if not resuming from a checkpoint is not provided.",
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string("agent_name", "Qwen3VL", help="Agent name.")

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    "fixed_task_seed",
    False,
    "Whether to use the same task seed when running multiple task combinations"
    " (n_task_combinations > 1).",
)

_MULTI_ATTEMPT_MODE = flags.DEFINE_boolean(
    "multi_attempt_mode",
    False,
    "Whether to enable multi-attempt mode with cross-attempt learning. "
    "When enabled, each task instance will be retried up to n_task_combinations times "
    "with self-reflection and long-term memory generation between attempts.",
)

_N_ATTEMPTS_PER_TASK = flags.DEFINE_integer(
    "n_attempts_per_task",
    3,
    "Number of attempts per task instance in multi-attempt mode. "
    "Only used when multi_attempt_mode is True.",
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        "This task is running in a mock app, you must stay in this app and"
        " DO NOT use the `navigate_home` action."
    ),
]


def load_config_with_override():
    """
    加载配置文件并允许命令行参数覆盖

    Returns:
        dict: 合并后的配置字典
    """
    config = {}

    # 尝试从config.yaml加载默认配置
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return config


def _get_agent(
    env: interface.AsyncEnv,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
    """Gets agent."""
    print("Initializing agent...")
    agent = None
    # 加载合并后的配置
    config = load_config_with_override()

    if _AGENT_NAME.value == "human_agent":
        agent = human_agent.HumanAgent(env)
    elif _AGENT_NAME.value == "random_agent":
        agent = random_agent.RandomAgent(env)
    # Gemini.
    elif _AGENT_NAME.value == "m3a_gemini_gcp":
        agent = m3a.M3A(env, infer.GeminiGcpWrapper(model_name="gemini-1.5-pro-latest"))
    elif _AGENT_NAME.value == "t3a_gemini_gcp":
        agent = t3a.T3A(env, infer.GeminiGcpWrapper(model_name="gemini-1.5-pro-latest"))
    # GPT.
    elif _AGENT_NAME.value == "t3a_gpt4":
        agent = t3a.T3A(env, infer.Gpt4Wrapper("gpt-4-turbo-2024-04-09"))
    elif _AGENT_NAME.value == "m3a_gpt4v":
        agent = m3a.M3A(env, infer.Gpt4Wrapper("gpt-4-turbo-2024-04-09"))
    # SeeAct.
    elif _AGENT_NAME.value == "seeact":
        agent = seeact.SeeAct(env)
    elif _AGENT_NAME.value == "UI_TARS":
        uitars_config = {
            "UITARS_BASE_URL": config["UITARS_BASE_URL"],
            "UITARS_API_KEY": config["UITARS_API_KEY"],
            "UITARS_MODEL": config["UITARS_MODEL"],
            "UITARS_HISTORY_N": config["UITARS_HISTORY_N"],
        }
        agent = ui_tars.UITARS(env, config=uitars_config)
    elif _AGENT_NAME.value == "Qwen3VL":
        qwen_config = {
            "QWEN_BASE_URL": config.get("QWEN_BASE_URL"),
            "QWEN_API_KEY": config.get("QWEN_API_KEY"),
            "QWEN_MODEL": config.get("QWEN_MODEL"),
            "QWEN_MAX_PIXELS": config.get("QWEN_MAX_PIXELS"),
        }
        agent = qwen3_vl.Qwen3VL(env, config=qwen_config)
    elif _AGENT_NAME.value == "GUIOwl15":
        # 与 Qwen3-VL 共用 config.yaml 中的 QWEN_* 键；GUIOwl15 内部也读这些键
        guiowl_config = {
            "QWEN_BASE_URL": config.get("QWEN_BASE_URL"),
            "QWEN_API_KEY": config.get("QWEN_API_KEY"),
            "QWEN_MODEL": config.get("QWEN_MODEL"),
        }
        agent = gui_owl.GUIOwl15(env, config=guiowl_config)

    if not agent:
        raise ValueError(f"Unknown agent: {_AGENT_NAME.value}")

    if (
        agent.name in ["M3A", "T3A", "SeeAct"]
        and family
        and family.startswith("miniwob")
        and hasattr(agent, "set_task_guidelines")
    ):
        agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
    agent.name = _AGENT_NAME.value

    return agent


def _allocate_step_budget_for_task(task_complexity: float) -> int:
    """Allocates number of steps dynamically based on the complexity score.

    Args:
        task_complexity: Complexity score of the task.

    Returns:
        Allocated number of steps for the task.
    """
    if task_complexity is None:
        raise ValueError("Task complexity must be provided.")
    return int(10 * (task_complexity))


def _run_multi_attempt_suite(
    suite: suite_utils.Suite,
    agent: base_agent.EnvironmentInteractingAgent,
    env,
    checkpoint_dir: str,
    n_attempts: int = 3,
    demo_mode: bool = False,
    result_saver: ResultSaver | None = None,
) -> list[dict]:
    """Run suite with multi-attempt mode.

    In multi-attempt mode:
    1. Each task instance runs up to n_attempts times
    2. After each failed attempt, the agent generates self-reflection
    3. Long-term memory from failed attempts is used in subsequent attempts
    4. Long-term memory is independent per task (not shared across tasks)

    Args:
        suite: The task suite.
        agent: The agent to run.
        env: The environment.
        checkpoint_dir: Directory for saving .pkl.gz checkpoint files.
        n_attempts: Maximum number of attempts per task instance.
        demo_mode: Whether running in demo mode.
        result_saver: Result saver for JSON output.

    Returns:
        List of episode results.
    """
    from android_world.task_evals.miniwob import miniwob_base

    # Create parser directory with -parser suffix
    parser_dir = checkpoint_dir.rstrip("/") + "-parser"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(parser_dir, exist_ok=True)

    print(f"Checkpoint directory (.pkl.gz): {checkpoint_dir}")
    print(f"Parser directory (screenshots/JSON): {parser_dir}")

    results = []
    total_tasks = sum(len(instances) for instances in suite.values())
    current_task_num = 0

    for task_name, instances in suite.items():
        print(f"\n{'=' * 60}")
        print(f"Running task: {task_name}")
        print(f"{'=' * 60}\n")

        for instance_id, task_instance in enumerate(instances):
            current_task_num += 1
            print(
                f"\n--- Task {current_task_num}/{total_tasks}: {task_name} (Instance {instance_id + 1}/{len(instances)}) ---"
            )

            # Determine max steps based on task complexity
            max_n_steps = _allocate_step_budget_for_task(task_instance.complexity)

            # Determine termination function
            termination_fn = None
            if task_name.lower().startswith("miniwob"):
                termination_fn = miniwob_base.is_episode_terminated

            # Run multi-attempt episode
            multi_result = multi_attempt_runner.run_multi_attempt_episode(
                task=task_instance,
                instance_id=instance_id,
                agent=agent,
                env=env,
                checkpoint_dir=checkpoint_dir,
                parser_dir=parser_dir,
                n_attempts=n_attempts,
                max_n_steps=max_n_steps,
                termination_fn=termination_fn,
                demo_mode=demo_mode,
            )

            # Convert to standard format
            episode_result = (
                multi_attempt_runner.convert_multi_attempt_to_episode_result(
                    multi_result
                )
            )
            episode_result["instance_id"] = instance_id
            episode_result["agent_name"] = agent.name
            episode_result["difficulty"] = getattr(task_instance, 'difficulty', None)
            episode_result["tags"] = getattr(task_instance, 'tags', None)
            episode_result["complexity"] = getattr(task_instance, 'complexity', None)

            results.append(episode_result)

            # 实时保存任务结果到 JSON
            if result_saver:
                task_result = suite_utils._merge_task_metadata(
                    pd.DataFrame([episode_result])
                ).to_dict('records')[0]
                result_saver.update_task_result(task_result)

            # 打印实时汇总统计（使用 pass@n 模式）
            suite_utils._print_and_save_realtime_summary(
                results=results,
                n_task_combinations=n_attempts,  # 多轮尝试模式用 n_attempts
                result_saver=result_saver,
            )

    return results


def _main() -> None:
    """Runs eval suite and gets rewards back."""
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )

    n_task_combinations = _N_TASK_COMBINATIONS.value
    task_registry = registry.TaskRegistry()

    # In multi-attempt mode, we only create 1 instance per task
    # (the n_task_combinations becomes n_attempts instead)
    if _MULTI_ATTEMPT_MODE.value:
        suite_n_combinations = 1
    else:
        suite_n_combinations = n_task_combinations

    suite = suite_utils.create_suite(
        task_registry.get_registry(family=_SUITE_FAMILY.value),
        n_task_combinations=suite_n_combinations,
        seed=_TASK_RANDOM_SEED.value,
        tasks=_TASKS.value,
        use_identical_params=_FIXED_TASK_SEED.value,
    )
    suite.suite_family = _SUITE_FAMILY.value

    agent = _get_agent(env, _SUITE_FAMILY.value)

    if _SUITE_FAMILY.value.startswith("miniwob"):
        # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
        agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
    else:
        agent.transition_pause = None

    if _CHECKPOINT_DIR.value:
        checkpoint_dir = _CHECKPOINT_DIR.value
    else:
        checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 从 checkpoint_dir 中提取 session 名
    session_name = os.path.basename(checkpoint_dir.rstrip('/'))

    print(
        f"Starting eval with agent {_AGENT_NAME.value} and writing to {checkpoint_dir}"
    )

    # 初始化结果保存器（使用 session 名作为 JSON 文件名）
    result_saver = ResultSaver(checkpoint_dir, session_name=session_name)

    # 计算总任务数
    total_tasks = sum(len(instances) for instances in suite.values())

    # 初始化实验信息
    result_saver.init_experiment(
        agent_name=_AGENT_NAME.value,
        suite_family=_SUITE_FAMILY.value,
        n_task_combinations=suite_n_combinations,
        task_random_seed=_TASK_RANDOM_SEED.value,
        total_tasks=total_tasks,
    )

    if _MULTI_ATTEMPT_MODE.value:
        # Use multi-attempt mode
        print(
            f"Multi-attempt mode enabled with {_N_ATTEMPTS_PER_TASK.value} attempts per task"
        )
        results = _run_multi_attempt_suite(
            suite=suite,
            agent=agent,
            env=env,
            checkpoint_dir=checkpoint_dir,
            n_attempts=_N_ATTEMPTS_PER_TASK.value,
            demo_mode=False,
            result_saver=result_saver,
        )

        # 打印最终汇总
        print(f"\n{'#' * 80}")
        print(f"{'FINAL EXPERIMENT RESULTS':^80}")
        print(f"{'#' * 80}")

        final_summary = suite_utils._print_and_save_realtime_summary(
            results=results,
            n_task_combinations=_N_ATTEMPTS_PER_TASK.value,
            result_saver=result_saver,
        )

        # 标记实验完成
        result_saver.complete_experiment(final_summary)

    else:
        # Use standard mode - 收集结果并实时打印
        results = suite_utils.run_with_realtime_summary(
            suite=suite,
            agent=agent,
            checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
            demo_mode=False,
            result_saver=result_saver,
            n_task_combinations=n_task_combinations,
        )

        # 打印最终汇总
        print(f"\n{'#' * 80}")
        print(f"{'FINAL EXPERIMENT RESULTS':^80}")
        print(f"{'#' * 80}")

        final_summary = suite_utils._print_and_save_realtime_summary(
            results=results,
            n_task_combinations=n_task_combinations,
            result_saver=result_saver,
        )

        # 标记实验完成
        result_saver.complete_experiment(final_summary)

    print(f"\n{'=' * 80}")
    print(f"JSON results saved to: {result_saver.json_path}")
    print(f"{'=' * 80}")

    print(
        f"Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}"
        f" family. Wrote to {checkpoint_dir}."
    )
    env.close()


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == "__main__":
    app.run(main)
