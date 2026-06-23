import os
import argparse
import subprocess
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import shutil

from framework import utils
from framework import task_scheduler
from framework.progress_monitor import print_realtime_progress as _print_progress
from framework.progress_monitor import set_start_time
from framework.device_initializer import (
    ComprehensiveDeviceSetup,
    initialize_devices_parallel,
)
from framework.hint_manager import get_hints_for_agent, HintManager
from concurrent_execution import run_task_with_multi_devices
from config_loader import load_config

load_dotenv(verbose=True, override=True)

# Load configuration with mode presets applied
config = load_config(verbose=True)

# Set start time for progress monitoring
set_start_time()

# 创建评估任务执行器
eval_executor = ThreadPoolExecutor(max_workers=config.get("MAX_EVAL_SUBPROCESS", 8))

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--agents", type=str, default=config["AGENT_NAME"])
parser.add_argument(
    "--mode", type=str, default="full", choices=["full", "exec", "eval"]
)
parser.add_argument("--session_id", type=str, default=config["SESSION_ID"])
parser.add_argument("--task_id", type=str, default=None)
parser.add_argument("--no_concurrent", action="store_true")
parser.add_argument("--setup_avd", action="store_true", default=True)
parser.add_argument("--setup_emulator", action="store_true")
parser.add_argument("--skip_key_components", type=bool, default=True)
parser.add_argument(
    "--reasoning_mode", type=str, default="direct", choices=["result_only", "direct"]
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="with_action",
    choices=["no_action", "with_action", "text_action"],
)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--overwrite_session", action="store_true")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=config["MAX_ATTEMPTS"],
    help="Maximum number of attempts for each task.",
)
parser.add_argument(
    "--reset_app_data",
    action="store_true",
    default=None,
    help="Enable app data reset before each task execution (overrides config.yaml).",
)
parser.add_argument(
    "--no_reset_app_data",
    action="store_true",
    help="Disable app data reset before task execution (overrides config.yaml).",
)
parser.add_argument(
    "--init_device",
    action="store_true",
    default=True,
    help="Initialize device after startup (install apps, set time, etc.).",
)
parser.add_argument(
    "--no_init_device",
    action="store_true",
    help="Disable device initialization after startup.",
)
parser.add_argument(
    "--install_apps",
    action="store_true",
    default=False,
    help="Install all 24 apps during device initialization (default: False, assumes apps are preinstalled).",
)
args = parser.parse_args()

# 处理reset_app_data参数
# 优先级：命令行参数 > config.yaml 配置
if args.no_reset_app_data:
    # 明确禁用
    args.reset_app_data = False
elif args.reset_app_data is None:
    # 未指定命令行参数，使用 config.yaml 中的配置（默认为 False）
    args.reset_app_data = config.get("RESET_APP_DATA", False)
# else: args.reset_app_data 已经是 True（用户明确指定了 --reset_app_data）

# 处理init_device参数
if args.no_init_device:
    args.init_device = False

# 当 RESET_APP_DATA=false 时，自动禁用设备初始化和应用安装
# 这意味着直接使用预配置的 AVD 镜像，不进行任何初始化操作
if not args.reset_app_data:
    args.init_device = False
    args.install_apps = False
    print("=" * 60)
    print("RESET_APP_DATA=false: 已禁用设备初始化和应用安装")
    print("将直接使用预配置的 AVD 镜像执行任务")
    print("=" * 60)

# 初始化输出目录和结果DataFrame
# 自动保存 config.yaml 到输出目录供后续查看
config_yaml_path = os.path.join(os.getcwd(), "config.yaml")
output_dir = utils.setup_output_directory(
    os.path.join(os.getcwd(), config["RESULTS_DIR"]),
    args.session_id,
    args.overwrite_session,
    config_path=config_yaml_path,
)

result_overwrite = args.overwrite
print("Overwrite:", result_overwrite)

# 设置设备
if args.mode in ("full", "exec"):
    if args.setup_avd:
        utils.setup_avd(
            config["SYS_AVD_HOME"],
            os.path.join(os.getcwd(), config["SOURCE_AVD_HOME"]),
            config["SOURCE_AVD_NAME"],
            config["NUM_OF_EMULATOR"],
            config["ANDROID_SDK_PATH"],
        )
        devices = utils.setup_emulator(
            config["EMULATOR_PATH"],
            config["SOURCE_AVD_NAME"],
            config["NUM_OF_EMULATOR"],
        )
    elif args.setup_emulator:
        devices = utils.setup_emulator(
            config["EMULATOR_PATH"],
            config["SOURCE_AVD_NAME"],
            config["NUM_OF_EMULATOR"],
        )
    else:
        devices = utils.setup_devices()

    # 设备初始化：设置时间、系统设置、可选安装应用
    # 使用并行初始化提高多设备启动效率
    if args.init_device:
        print("=" * 60)
        print(f"开始并行设备初始化 ({len(devices)} 个设备)...")
        print("=" * 60)

        # 使用并行初始化函数
        init_results = initialize_devices_parallel(
            devices, install_apps=args.install_apps, setup_datetime=True
        )

        # 打印每个设备的初始化结果
        for serial, success in init_results.items():
            status = "成功" if success else "失败"
            print(f"设备 {serial}: 初始化{status}")

        # 统计结果
        success_count = sum(1 for v in init_results.values() if v)
        print("=" * 60)
        print(f"设备初始化完成: {success_count}/{len(init_results)} 个设备成功")
        print("=" * 60)
else:
    devices = [{"serial": "eval_mode"}]

print(">>> [DIAGNOSTIC] After device init, before agent_scope", flush=True)

# 确定代理和任务范围
if args.agents is None:
    agent_scope = [agent_config["NAME"] for agent_config in config["AGENTS"]]
else:
    agent_scope = args.agents.split(",")

# 验证代理名称
if args.mode in ("full", "exec"):
    for agent_name in agent_scope:
        utils.get_agent(agent_name)(config)

# 设置结果CSV
print(">>> [DIAGNOSTIC] Before setup_results_csv", flush=True)
results_df = utils.setup_results_csv(
    output_dir,
    config["DATASET_PATH"],
    agent_scope,
    args.max_attempts,
    args.reasoning_mode,
    args.action_mode,
)
print(">>> [DIAGNOSTIC] setup_results_csv returned", flush=True)
config["output_dir"] = output_dir
# Only override API keys if environment variables are set, otherwise keep config.yaml values
if os.getenv("OPENAI_API_KEY"):
    config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("QWEN_API_KEY"):
    config["QWEN_API_KEY"] = os.getenv("QWEN_API_KEY")

# 确定任务范围
if args.task_id is None:
    task_scope = list(results_df.itertuples(index=False))
else:
    task_rows = results_df[results_df["task_identifier"] == args.task_id]
    if task_rows.empty:
        print(f"Error: Task ID '{args.task_id}' not found in the dataset.")
        exit(1)
    task_scope = list(task_rows.itertuples(index=False))

subprocess_list = []  # 存储并发子进程的列表

# Get early stop configuration
early_stop_on_success = config.get("EARLY_STOP_ON_SUCCESS", False)

# Get self-hint configuration
self_hint_enabled = config.get("SELF_HINT_ENABLED", False)
print(f"Self-hint enabled: {self_hint_enabled}")


def check_attempt_success(output_dir, task_identifier, agent_name, attempt):
    """
    Check if an attempt has succeeded (evaluation_summary.json final_result == 1).

    Args:
        output_dir: Output directory path
        task_identifier: Task identifier
        agent_name: Agent name
        attempt: Attempt number

    Returns:
        bool: True if attempt succeeded, False otherwise
    """
    import json

    attempt_dir = os.path.join(
        output_dir, task_identifier, agent_name, f"attempt_{attempt}"
    )
    eval_summary_path = os.path.join(attempt_dir, "evaluation_summary.json")

    if not os.path.exists(eval_summary_path):
        return False

    try:
        with open(eval_summary_path, "r") as f:
            eval_data = json.load(f)
            return eval_data.get("final_result", 0) == 1
    except Exception as e:
        print(f"Error reading evaluation summary for attempt {attempt}: {e}")
        return False


def check_task_has_success(output_dir, task_identifier, agent_name, max_attempts):
    """
    Check if any attempt for a task has succeeded.

    Returns:
        tuple: (has_success: bool, successful_attempt: int or None)
    """
    for attempt in range(1, max_attempts + 1):
        if check_attempt_success(output_dir, task_identifier, agent_name, attempt):
            return True, attempt
    return False, None


def print_realtime_progress(trigger=""):
    """Wrapper function: call progress_monitor module to print real-time progress"""
    _print_progress(
        task_scope=task_scope,
        agent_scope=agent_scope,
        output_dir=output_dir,
        max_attempts=args.max_attempts,
        trigger=trigger,
    )


def run_task_rollout(agent_name, task, subprocess_list, devices):
    """
    Execute task rollout logic: run attempts with different emulators.
    If EARLY_STOP_ON_SUCCESS is True, stop after first successful attempt.
    """
    print(
        f"=== Starting rollout execution for task {task.task_identifier} with {args.max_attempts} attempts ==="
    )

    # Check if task already has a successful attempt (early stop mode)
    if early_stop_on_success and not result_overwrite:
        has_success, successful_attempt = check_task_has_success(
            output_dir, task.task_identifier, agent_name, args.max_attempts
        )
        if has_success:
            print(
                f"Task {task.task_identifier} already has a successful attempt ({successful_attempt}). "
                f"Skipping remaining attempts (EARLY_STOP_ON_SUCCESS=True)."
            )
            return

    # Check if task is fully completed
    task_fully_completed = utils.is_task_fully_completed(
        output_dir,
        task.task_identifier,
        agent_name,
        args.max_attempts,
        args.reasoning_mode,
        args.action_mode,
    )

    if task_fully_completed:
        if result_overwrite:
            print(
                f"Task {task.task_identifier} is already fully completed. Overwriting results due to --overwrite flag."
            )
            utils.clear_task_results(
                output_dir,
                task.task_identifier,
                agent_name,
                args.max_attempts,
                args.reasoning_mode,
                args.action_mode,
            )
        else:
            print(f"Task {task.task_identifier} is already fully completed. Skipping.")
            return

    agent = utils.get_agent(agent_name=agent_name)(config)

    # Rollout逻辑：执行所有attempts，每个attempt使用不同的emulator
    for attempt in range(1, args.max_attempts + 1):
        print(
            f"--- Processing attempt {attempt}/{args.max_attempts} for task {task.task_identifier} ---"
        )

        # 选择设备：循环使用可用设备，确保不同attempt使用不同设备
        device_index = (attempt - 1) % len(devices)
        device = devices[device_index]

        print(f"Using device {device['serial']} for attempt {attempt}")

        # 检查当前attempt的状态
        attempt_status = utils.get_attempt_status(
            output_dir,
            task.task_identifier,
            agent_name,
            attempt,
            args.reasoning_mode,
            args.action_mode,
        )

        print(f"Attempt {attempt} status: {attempt_status}")

        if attempt_status == "executed_and_evaluated":
            if result_overwrite:
                print(
                    f"Attempt {attempt} already completed but overwriting due to --overwrite flag."
                )
            else:
                print(f"Attempt {attempt} is already executed and evaluated. Skipping.")
                continue
        elif attempt_status == "executed_not_evaluated":
            print(
                f"Attempt {attempt} is executed but not evaluated. Performing evaluation only."
            )
            # 只执行评估
            if args.mode == "full":
                print(f"Evaluating task {task.task_identifier}, attempt {attempt}...")
                utils.immediate_evaluate_and_update_pass_at_k(
                    output_dir,
                    task.task_identifier,
                    agent_name,
                    attempt,
                    args.reasoning_mode,
                    args.action_mode,
                    result_overwrite,
                    self_hint_enabled,
                )
                print_realtime_progress(
                    f"Evaluation Done: {task.task_identifier} (attempt {attempt})"
                )
            continue

        # 对于需要执行的attempt，确保设备准备就绪
        if args.setup_avd:
            print(
                f"Ensuring device {device['serial']} is ready for attempt {attempt}..."
            )
            if not utils.check_and_restart_device_if_needed(
                device, config["EMULATOR_PATH"], config["SOURCE_AVD_NAME"]
            ):
                print(
                    f"Failed to prepare device {device['serial']}, skipping attempt {attempt}"
                )
                continue

        # Reset app data (before each attempt execution)
        if args.reset_app_data:
            print(
                f"=== [RESET_APP_DATA] Starting app data reset for task {task.task_identifier} (attempt {attempt}) ==="
            )
            reset_result = utils.reset_app_data_for_task(
                device["serial"], task, inject_data=True
            )
            print(
                f"=== [RESET_APP_DATA] Reset result: {'success' if reset_result else 'failed'} ==="
            )
        else:
            print(
                f"=== [RESET_APP_DATA] App data reset disabled (args.reset_app_data={args.reset_app_data}) ==="
            )

        # 执行任务
        if args.mode in ("full", "exec"):
            attempt_dir = os.path.join(
                output_dir, task.task_identifier, agent_name, f"attempt_{attempt}"
            )

            # 清空attempt文件夹，防止残余文件影响实验
            if os.path.exists(attempt_dir):
                print(f"Clearing existing attempt directory: {attempt_dir}")
                shutil.rmtree(attempt_dir)

            # 创建新的attempt文件夹
            os.makedirs(attempt_dir)
            print(f"Created clean attempt directory: {attempt_dir}")

            # 保存任务元数据（摆脱对 results.csv 的依赖）
            utils.save_task_metadata(attempt_dir, task)

            # Get hints from previous failed attempts if self-hint is enabled
            hints = ""
            if self_hint_enabled and attempt > 1:
                hints = get_hints_for_agent(
                    output_dir, task.task_identifier, agent_name, attempt
                )
                if hints:
                    print(
                        f"Self-hint: Retrieved hints from previous attempts for attempt {attempt}"
                    )

            # 执行任务，处理重试逻辑
            max_crash_retries = 3
            for retry in range(max_crash_retries + 1):
                print(
                    f"Executing attempt {attempt}, retry {retry + 1}/{max_crash_retries + 1}"
                )

                task_completed, task_exit_code = agent.execute_task(
                    task, device, attempt_dir, hints=hints
                )

                # 如果执行成功或者不是设备崩溃问题，跳出重试循环
                if task_exit_code != 2:
                    break

                # 如果是设备崩溃且还有重试机会
                if retry < max_crash_retries:
                    print(
                        f"Attempt {attempt} failed due to execution error (exit code 2), retrying..."
                    )

                    # 清理失败的attempt目录
                    if os.path.exists(attempt_dir):
                        print(f"Clearing failed attempt directory: {attempt_dir}")
                        shutil.rmtree(attempt_dir)
                        os.makedirs(attempt_dir)

                    # 重启设备
                    if args.setup_avd:
                        if not utils.check_and_restart_device_if_needed(
                            device, config["EMULATOR_PATH"], config["SOURCE_AVD_NAME"]
                        ):
                            print(
                                f"Device {device['serial']} could not be restored. Failing attempt {attempt}"
                            )
                            break
                else:
                    print(
                        f"Max crash retries reached for attempt {attempt}. Recording failure."
                    )

            print(
                f"Finished execution for task: {task.task_identifier}, attempt: {attempt}"
            )
            utils.close_app_activity(device["serial"], None)

            # 兜底：若子进程被 kill 导致 log.json 缺失，生成最小 log.json 让评估能记为失败
            utils.ensure_fallback_log_json(attempt_dir)

            # 保存执行结果
            utils.save_result__completed_execution(
                output_dir,
                task.task_identifier,
                agent_name,
                task_completed,
                task_exit_code,
                device["serial"],
                attempt,
            )

            # Immediately evaluate (in full mode)
            if args.mode == "full":
                print(
                    f"Immediately evaluating task {task.task_identifier}, attempt {attempt}..."
                )
                utils.immediate_evaluate_and_update_pass_at_k(
                    output_dir,
                    task.task_identifier,
                    agent_name,
                    attempt,
                    args.reasoning_mode,
                    args.action_mode,
                    result_overwrite,
                    self_hint_enabled,
                )
                print_realtime_progress(
                    f"Evaluation Done: {task.task_identifier} (attempt {attempt})"
                )

                # Check for early stop on success
                if early_stop_on_success and check_attempt_success(
                    output_dir, task.task_identifier, agent_name, attempt
                ):
                    print(
                        f"*** Task {task.task_identifier} attempt {attempt} succeeded (final_result=1). "
                        f"Skipping remaining attempts ({attempt + 1}-{args.max_attempts}). ***"
                    )
                    break

    print(f"=== Completed rollout execution for task {task.task_identifier} ===")


def run_single_attempt(agent_name, task, attempt, device):
    """
    Execute a single attempt for concurrent execution.
    """
    print(
        f"[Device {device['serial']}] Starting attempt {attempt} for task {task.task_identifier}"
    )

    # Check if any previous attempt has succeeded (early stop mode)
    if early_stop_on_success and attempt > 1:
        for prev_attempt in range(1, attempt):
            if check_attempt_success(
                output_dir, task.task_identifier, agent_name, prev_attempt
            ):
                print(
                    f"[Device {device['serial']}] Task {task.task_identifier} already has a successful attempt ({prev_attempt}). "
                    f"Skipping attempt {attempt} (EARLY_STOP_ON_SUCCESS=True)."
                )
                return True

    # Check current attempt status
    attempt_status = utils.get_attempt_status(
        output_dir,
        task.task_identifier,
        agent_name,
        attempt,
        args.reasoning_mode,
        args.action_mode,
    )

    print(f"[Device {device['serial']}] Attempt {attempt} status: {attempt_status}")

    if attempt_status == "executed_and_evaluated":
        if result_overwrite:
            print(
                f"[Device {device['serial']}] Attempt {attempt} already completed but overwriting due to --overwrite flag."
            )
        else:
            print(
                f"[Device {device['serial']}] Attempt {attempt} is already executed and evaluated. Skipping."
            )
            return True
    elif attempt_status == "executed_not_evaluated":
        print(
            f"[Device {device['serial']}] Attempt {attempt} is executed but not evaluated. Performing evaluation only."
        )
        # 只执行评估
        if args.mode == "full":
            print(
                f"[Device {device['serial']}] Evaluating task {task.task_identifier}, attempt {attempt}..."
            )
            utils.immediate_evaluate_and_update_pass_at_k(
                output_dir,
                task.task_identifier,
                agent_name,
                attempt,
                args.reasoning_mode,
                args.action_mode,
                result_overwrite,
                self_hint_enabled,
            )
            print_realtime_progress(
                f"Evaluation Done: {task.task_identifier} (attempt {attempt})"
            )
        return True

    # 对于需要执行的attempt，确保设备准备就绪
    if args.setup_avd:
        print(
            f"[Device {device['serial']}] Ensuring device is ready for attempt {attempt}..."
        )
        if not utils.check_and_restart_device_if_needed(
            device, config["EMULATOR_PATH"], config["SOURCE_AVD_NAME"]
        ):
            print(
                f"[Device {device['serial']}] Failed to prepare device, skipping attempt {attempt}"
            )
            return False

    # Reset app data (before each attempt execution)
    if args.reset_app_data:
        print(
            f"[Device {device['serial']}] === [RESET_APP_DATA] Starting app data reset for task {task.task_identifier} (attempt {attempt}) ==="
        )
        reset_result = utils.reset_app_data_for_task(
            device["serial"], task, inject_data=True
        )
        print(
            f"[Device {device['serial']}] === [RESET_APP_DATA] Reset result: {'success' if reset_result else 'failed'} ==="
        )
    else:
        print(
            f"[Device {device['serial']}] === [RESET_APP_DATA] App data reset disabled (args.reset_app_data={args.reset_app_data}) ==="
        )

    # 执行任务
    if args.mode in ("full", "exec"):
        attempt_dir = os.path.join(
            output_dir, task.task_identifier, agent_name, f"attempt_{attempt}"
        )

        # 清空attempt文件夹，防止残余文件影响实验
        if os.path.exists(attempt_dir):
            print(
                f"[Device {device['serial']}] Clearing existing attempt directory: {attempt_dir}"
            )
            shutil.rmtree(attempt_dir)

        # 创建新的attempt文件夹
        os.makedirs(attempt_dir)
        print(
            f"[Device {device['serial']}] Created clean attempt directory: {attempt_dir}"
        )

        # 保存任务元数据（摆脱对 results.csv 的依赖）
        utils.save_task_metadata(attempt_dir, task)

        # 创建agent实例
        agent = utils.get_agent(agent_name=agent_name)(config)

        # Get hints from previous failed attempts if self-hint is enabled
        hints = ""
        if self_hint_enabled and attempt > 1:
            hints = get_hints_for_agent(
                output_dir, task.task_identifier, agent_name, attempt
            )
            if hints:
                print(
                    f"[Device {device['serial']}] Self-hint: Retrieved hints from previous attempts for attempt {attempt}"
                )

        # 执行任务，处理重试逻辑
        max_crash_retries = 3
        for retry in range(max_crash_retries + 1):
            print(
                f"[Device {device['serial']}] Executing attempt {attempt}, retry {retry + 1}/{max_crash_retries + 1}"
            )

            task_completed, task_exit_code = agent.execute_task(
                task, device, attempt_dir, hints=hints
            )

            # 如果执行成功或者不是设备崩溃问题，跳出重试循环
            if task_exit_code != 2:
                break

            # 如果是设备崩溃且还有重试机会
            if retry < max_crash_retries:
                print(
                    f"[Device {device['serial']}] Attempt {attempt} failed due to execution error (exit code 2), retrying..."
                )

                # 清理失败的attempt目录
                if os.path.exists(attempt_dir):
                    print(
                        f"[Device {device['serial']}] Clearing failed attempt directory: {attempt_dir}"
                    )
                    shutil.rmtree(attempt_dir)
                    os.makedirs(attempt_dir)

                # 重启设备
                if args.setup_avd:
                    if not utils.check_and_restart_device_if_needed(
                        device, config["EMULATOR_PATH"], config["SOURCE_AVD_NAME"]
                    ):
                        print(
                            f"[Device {device['serial']}] Device could not be restored. Failing attempt {attempt}"
                        )
                        break
            else:
                print(
                    f"[Device {device['serial']}] Max crash retries reached for attempt {attempt}. Recording failure."
                )

        print(
            f"[Device {device['serial']}] Finished execution for task: {task.task_identifier}, attempt: {attempt}"
        )
        utils.close_app_activity(device["serial"], None)

        # 兜底：若子进程被 kill 导致 log.json 缺失，生成最小 log.json 让评估能记为失败
        utils.ensure_fallback_log_json(attempt_dir)

        # 保存执行结果
        utils.save_result__completed_execution(
            output_dir,
            task.task_identifier,
            agent_name,
            task_completed,
            task_exit_code,
            device["serial"],
            attempt,
        )

        # Immediately evaluate (in full mode)
        if args.mode == "full":
            print(
                f"[Device {device['serial']}] Immediately evaluating task {task.task_identifier}, attempt {attempt}..."
            )
            utils.immediate_evaluate_and_update_pass_at_k(
                output_dir,
                task.task_identifier,
                agent_name,
                attempt,
                args.reasoning_mode,
                args.action_mode,
                result_overwrite,
                self_hint_enabled,
            )
            print_realtime_progress(
                f"Evaluation Done: {task.task_identifier} (attempt {attempt})"
            )

            # Check for early stop on success and print message
            if early_stop_on_success and check_attempt_success(
                output_dir, task.task_identifier, agent_name, attempt
            ):
                print(
                    f"[Device {device['serial']}] *** Task {task.task_identifier} attempt {attempt} succeeded (final_result=1). ***"
                )

    return True


def run_concurrent_rollout_mode():
    """
    Concurrent task mode: different tasks run in parallel across devices,
    but attempts of the same task run serially on a single device.
    This ensures self-hint and early-stop work correctly (which depend on
    previous attempt results) while maximizing device utilization.
    """
    from queue import Queue

    print(
        f">>> [DIAGNOSTIC] Entered run_concurrent_rollout_mode. devices={len(devices)}, task_scope={len(task_scope)}", flush=True
    )
    print(
        f"=== Concurrent tasks mode: {len(devices)} devices, {len(task_scope)} tasks, "
        f"{args.max_attempts} attempts/task ==="
    )
    print("=== Tasks run in PARALLEL, attempts of same task run SERIALLY ===")
    print(f"=== EARLY_STOP_ON_SUCCESS: {early_stop_on_success} ===")

    # Build work items: each item is a whole task (all attempts handled inside)
    work_items = []
    for agent_name in agent_scope:
        for task in task_scope:
            work_items.append((agent_name, task))

    if not work_items:
        print("No tasks to execute.")
        return

    # Thread-safe device pool: each task acquires a device, runs all attempts
    # serially on it, then returns it for the next task.
    device_queue = Queue()
    for device in devices:
        device_queue.put(device)

    max_workers = min(len(devices), len(work_items))
    print(f"Using {max_workers} concurrent workers for {len(work_items)} task(s)")

    def run_task_with_device(agent_name, task):
        """Acquire a device, run all attempts serially, then release."""
        device = device_queue.get()
        try:
            run_task_rollout(agent_name, task, subprocess_list, [device])
        finally:
            device_queue.put(device)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for agent_name, task in work_items:
            future = executor.submit(run_task_with_device, agent_name, task)
            future_to_task[future] = (agent_name, task)

        for future in future_to_task:
            agent_name, task = future_to_task[future]
            try:
                future.result()
                completed_count += 1
                print(
                    f"[{completed_count}/{len(work_items)}] Completed: task {task.task_identifier} with agent {agent_name}"
                )
            except Exception as e:
                completed_count += 1
                print(
                    f"[{completed_count}/{len(work_items)}] Error: task {task.task_identifier} with agent {agent_name}: {e}"
                )

    print("=== Concurrent tasks mode completed ===")


def run_eval_only_mode():
    """
    仅评估模式：批量评估所有任务
    """
    print("Running in eval-only mode...")
    eval_tasks = utils.collect_evaluation_tasks(
        output_dir,
        agent_scope,
        task_scope,
        args.max_attempts,
        args.reasoning_mode,
        args.action_mode,
        result_overwrite,
    )

    if not eval_tasks:
        print("No evaluation tasks found.")
        return

    eval_futures = []
    print(f"Submitting {len(eval_tasks)} evaluation tasks...")

    for eval_task in eval_tasks:
        future = eval_executor.submit(
            utils.execute_evaluation,
            eval_task["task_identifier"],
            output_dir,
            args.mode,
            eval_task["agent_name"],
            eval_task["attempt"],
            args.reasoning_mode,
            args.action_mode,
            eval_task["attempt_dir"],
            result_overwrite,
        )
        eval_futures.append(future)

    # 处理评估结果
    utils.process_evaluation_results(
        output_dir, eval_tasks, eval_futures, args.reasoning_mode, args.action_mode
    )


# 主执行逻辑
print(">>> [DIAGNOSTIC] Before print_realtime_progress(Startup)", flush=True)
print_realtime_progress("Startup")
print(">>> [DIAGNOSTIC] After print_realtime_progress(Startup)", flush=True)

if args.mode == "eval":
    run_eval_only_mode()
    print_realtime_progress("Evaluation Batch Done")
else:
    # 执行任务（full或exec模式）- 使用rollout逻辑
    print("=== Starting rollout execution mode ===")

    print(
        f"Available devices: {len(devices)}, Max attempts per task: {args.max_attempts}"
    )

    # 智能选择执行策略
    if not args.no_concurrent and len(task_scope) > 1:
        # 并发任务模式：不同任务并行执行（充分利用设备），同一任务的attempts串行执行
        print("Using concurrent tasks mode (tasks parallel, attempts serial)")
        run_concurrent_rollout_mode()
    else:
        # 串行模式：任务逐个执行，适用于单任务或明确要求串行的情况
        print("Using sequential rollout mode")

        # 按任务顺序逐个执行，每个任务的所有attempts都完成后再执行下一个任务
        for agent_name in agent_scope:
            for task in task_scope:
                print(
                    f"\n>>> Processing task {task.task_identifier} with agent {agent_name} <<<"
                )
                run_task_rollout(agent_name, task, subprocess_list, devices)

# 清理工作
if args.setup_avd or args.setup_emulator:
    utils.terminate_emulator([device["serial"] for device in devices])

if args.mode != "eval":
    print("All execution completed.")
    print_realtime_progress("Execution Done")
    if args.task_id is None:
        utils.print_execution_summary(output_dir, agent_scope)

# 等待所有子进程完成
for process in subprocess_list:
    process.wait()

# 关闭评估任务执行器
eval_executor.shutdown(wait=True)
print("All tasks completed.")

# 打印最终总结
if args.mode != "exec":
    print("All evaluation finished.")
    print_realtime_progress("Final Summary")
    utils.print_evaluation_summary(output_dir, agent_scope, args.max_attempts)
    print("Evaluation results have been saved to the results CSV file.")
