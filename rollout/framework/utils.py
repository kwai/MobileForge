import pandas as pd
from . import agents
import subprocess
import os
import shutil
import json
import time
import re
from filelock import FileLock, Timeout
from functools import wraps
from datetime import datetime


def get_apk(device_serial: str, package_name: str, local_apk_path: str):
    adb_command = f"adb -s {device_serial} shell pm path {package_name}"
    apk_path = execute_adb(adb_command)
    if apk_path == "ERROR":
        return "ERROR"
    apk_path = apk_path.split("package:")[1].strip()
    adb_command = f"adb -s {device_serial} pull {apk_path} {local_apk_path}"
    return execute_adb(adb_command)


def get_agent(agent_name):
    try:
        return getattr(agents, agent_name)
    except AttributeError:
        raise Exception(f"Required agent <{agent_name}> not implemented.")


def get_agent_config(config, agent_name):
    for agent in config["AGENTS"]:
        if agent["NAME"] == agent_name:
            return agent
    raise Exception("INVALID agent_name")


def execute_adb(adb_command, verbose=True):
    try:
        result = subprocess.run(adb_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        if verbose:
            print(f"Command execution failed: {adb_command}")
            print(result.stderr)
        return "ERROR"
    except UnicodeDecodeError:
        # Handle non-UTF-8 output by using binary mode and manual decoding
        try:
            result = subprocess.run(adb_command, shell=True, capture_output=True)
            if result.returncode == 0:
                return result.stdout.decode("utf-8", errors="replace").strip()
            if verbose:
                print(f"Command execution failed: {adb_command}")
                print(result.stderr.decode("utf-8", errors="replace"))
            return "ERROR"
        except Exception as e:
            if verbose:
                print(f"Command execution failed with exception: {adb_command}")
                print(f"Exception: {e}")
            return "ERROR"


def get_all_devices():
    adb_command = "adb devices"
    device_list = []
    result = execute_adb(adb_command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        for d in devices:
            device_list.append(d.split()[0])

    return device_list


def setup_devices():
    devices = get_all_devices()
    print(f"{len(devices)} device(s) found: {devices}")
    if len(devices) == 0:
        exit(1)
    elif len(devices) > 1:
        ans = input("Are you sure to run using all devices? (y/n)")
        if ans.strip().lower() != "y":
            exit(1)
    return [
        {"serial": serial, "console_port": None, "grpc_port": None}
        for serial in devices
    ]


def setup_avd(
    avd_home, source_avd_home, source_avd_name, num_of_copies, target_sdk_path
):
    from .utils_clone_avd import clone_avd

    for idx in range(num_of_copies):
        clone_avd(
            src_avd_dir=os.path.join(source_avd_home, source_avd_name + ".avd"),
            src_ini_file=os.path.join(source_avd_home, source_avd_name + ".ini"),
            src_avd_name=source_avd_name,
            tar_avd_name=f"{source_avd_name}_{idx}",
            src_android_avd_home=r"C:\Users\User\.android\avd",
            tar_android_avd_home=avd_home,
            src_sdk=r"C:\Users\User\AppData\Local\Android\Sdk",
            tar_sdk=target_sdk_path,
            target_linux=os.name == "posix",
        )


def parse_adb_devices(res) -> dict:
    devices = {}
    for line in res.split("\n")[1:]:
        serial, status = line.split("\t")
        devices[serial] = status
    return devices


def setup_emulator(emulator_exe, source_avd_name, num_of_emulators):
    sdk_path = os.path.dirname(os.path.dirname(emulator_exe))
    adb_path = os.path.join(sdk_path, "platform-tools")
    os.environ["PATH"] = f"{adb_path}{os.pathsep}{os.environ['PATH']}"
    devices = [
        {
            "serial": f"emulator-{5554 + (idx * 2)}",
            "console_port": 5554 + (idx * 2),
            "grpc_port": 8554 + (idx * 2),
        }
        for idx in range(num_of_emulators)
    ]
    devices_serial = [device["serial"] for device in devices]
    ready_devices = []
    for idx, device in enumerate(devices):
        command = [
            emulator_exe,
            "-avd",
            f"{source_avd_name}_{idx}",
            "-no-snapshot-save",
            "-no-window",
            "-no-audio",
            "-port",
            str(device["console_port"]),
            "-grpc",
            str(device["grpc_port"]),
        ]
        # add “no-window” to the command if need to run in headless mode
        http_proxy = os.environ.get("HTTP_PROXY")
        if http_proxy:
            command.extend(["-http-proxy", http_proxy])
        # 禁用 crash reporting，防止并行模拟器共享 /tmp/android-unknown/emu-crash-*.db/ 导致 lock 竞争
        emu_env = os.environ.copy()
        emu_env["ANDROID_EMU_ENABLE_CRASH_REPORTING"] = "0"
        subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.DEVNULL,  # to silence emulator output
            env=emu_env,
        )
    adb_command = "adb devices"
    while True:
        result = execute_adb(adb_command)
        if result == "ERROR":
            raise Exception("Error in executing ADB command")
        else:
            launched_devices = [
                serial
                for serial, status in parse_adb_devices(result).items()
                if status == "device" and serial in devices_serial
            ]
            print(
                f"{len(launched_devices)}/{num_of_emulators} device(s) launched; {len(ready_devices)}/{num_of_emulators} device(s) ready"
            )
            if len(launched_devices) == num_of_emulators:
                break
            else:
                time.sleep(1)
    while True:
        for serial in launched_devices:
            if serial in ready_devices:
                continue
            result = execute_adb(f"adb -s {serial} shell getprop sys.boot_completed")
            if result == "1":
                ready_devices.append(serial)
        print(
            f"{len(launched_devices)}/{num_of_emulators} device(s) launched; {len(ready_devices)}/{num_of_emulators} device(s) ready"
        )
        if len(ready_devices) == num_of_emulators:
            break
        else:
            time.sleep(1)

    return devices


def terminate_emulator(serial_list):
    for serial in serial_list:
        try:
            result = execute_adb(f"adb -s {serial} emu kill", verbose=False)
            if result != "ERROR":
                print(f"Successfully terminated emulator: {serial}")
            else:
                print(f"Failed to terminate emulator: {serial}")
        except Exception as e:
            print(f"Exception while terminating emulator {serial}: {e}")
            continue


def check_device_connectivity(device_serial):
    """
    检查设备是否在线并可访问
    :param device_serial: 设备序列号
    :return: True if device is online and accessible, False otherwise
    """
    try:
        # 检查设备是否在adb devices列表中
        result = execute_adb("adb devices", verbose=False)
        if result == "ERROR":
            return False

        # 解析设备状态
        devices = parse_adb_devices(result)
        if device_serial not in devices:
            return False

        if devices[device_serial] != "device":
            return False

        # 进一步检查设备是否响应
        response = execute_adb(
            f"adb -s {device_serial} shell echo 'alive'", verbose=False
        )
        return response == "alive"
    except Exception:
        return False


def restart_emulator(device_info, emulator_exe, source_avd_name):
    """
    重启单个emulator
    :param device_info: 设备信息字典，包含serial, console_port, grpc_port
    :param emulator_exe: emulator可执行文件路径
    :param source_avd_name: AVD名称
    :return: True if restart successful, False otherwise
    """
    try:
        device_serial = device_info["serial"]
        console_port = device_info["console_port"]
        grpc_port = device_info["grpc_port"]

        print(f"Restarting emulator {device_serial}...")

        # 1. 先尝试杀死旧的emulator进程
        execute_adb(f"adb -s {device_serial} emu kill", verbose=False)
        time.sleep(2)

        # 2. 提取设备索引
        if "emulator-" in device_serial:
            port_num = int(device_serial.split("-")[1])
            device_idx = (port_num - 5554) // 2
        else:
            return False

        # 3. 启动新的emulator
        command = [
            emulator_exe,
            "-avd",
            f"{source_avd_name}_{device_idx}",
            "-no-snapshot-save",
            "-no-window",
            "-no-audio",
            "-port",
            str(console_port),
            "-grpc",
            str(grpc_port),
        ]

        # 添加HTTP代理设置（如果有）
        http_proxy = os.environ.get("HTTP_PROXY")
        if http_proxy:
            command.extend(["-http-proxy", http_proxy])

        # 启动emulator，禁用 crash reporting 防止并行 lock 竞争
        emu_env = os.environ.copy()
        emu_env["ANDROID_EMU_ENABLE_CRASH_REPORTING"] = "0"
        subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=emu_env,
        )

        print(f"Emulator {device_serial} started, waiting for it to be ready...")

        # 4. 等待emulator启动完成
        max_wait_time = 120  # 最大等待120秒
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # 检查设备是否已启动
            result = execute_adb("adb devices", verbose=False)
            if result != "ERROR":
                devices = parse_adb_devices(result)
                if device_serial in devices and devices[device_serial] == "device":
                    # 检查设备是否完全启动
                    boot_result = execute_adb(
                        f"adb -s {device_serial} shell getprop sys.boot_completed",
                        verbose=False,
                    )
                    if boot_result == "1":
                        print(f"Emulator {device_serial} is ready!")
                        return True
            time.sleep(3)

        print(f"Timeout waiting for emulator {device_serial} to be ready")
        return False

    except Exception as e:
        print(f"Error restarting emulator {device_info}: {e}")
        return False


def check_and_restart_device_if_needed(device_info, emulator_exe, source_avd_name):
    """
    检查设备状态，如果掉线则重启
    :param device_info: 设备信息字典
    :param emulator_exe: emulator可执行文件路径
    :param source_avd_name: AVD名称
    :return: True if device is ready, False if restart failed
    """
    device_serial = device_info["serial"]

    # 检查设备连接性
    if check_device_connectivity(device_serial):
        return True

    print(f"Device {device_serial} is offline, attempting to restart...")

    # 尝试重启设备
    if restart_emulator(device_info, emulator_exe, source_avd_name):
        return True

    print(f"Failed to restart device {device_serial}")
    return False


def setup_app_activity(device_serial: str, adb_app: str, adb_home_page: str) -> bool:
    """Open the home page of the target app. Go to home-screen if failed or no info is given.

    Parameters:
    - device_serial (str): The android device serial number.
    - adb_app (str): The application package name.
    - adb_home_page (str): The activity class name.

    Returns:
    - bool: Whether the home page is successfully opened.
    """
    # Close app
    close_app_activity(device_serial, adb_app)

    # Start app
    launched = False
    if adb_app and adb_home_page:
        output = execute_adb(
            f"adb -s {device_serial} shell am start -n {adb_app}/{adb_home_page}",
            verbose=False,
        )
        if output != "ERROR":
            launched = True
    if not launched and adb_app:
        output = execute_adb(
            f"adb -s {device_serial} shell monkey -p {adb_app} -c android.intent.category.LAUNCHER 1"
        )
        if output != "ERROR":
            launched = True
    if launched:
        max_retry = 30
        trial = 0
        while trial < max_retry:
            windows = execute_adb(
                f'adb -s {device_serial} shell "dumpsys window | grep -E mCurrentFocus"',
                verbose=False,
            )
            if windows == "ERROR":
                break
            m = re.search(
                r"mCurrentFocus=Window{.*\s+(?P<package>[^\s]+)/(?P<activity>[^\s]+)\}",
                windows,
            )
            if m and m.group("package") == adb_app:
                break
            else:
                time.sleep(1)
                trial += 1
        time.sleep(10)  # For loading app content
        return True
    else:
        execute_adb(f"adb -s {device_serial} shell input keyevent KEYCODE_HOME")
        print(
            f"10 seconds are allowed to start the app `{adb_app}/{adb_home_page}` on {device_serial} manually:"
        )
        time.sleep(10)
        return False


def close_app_activity(
    device_serial: str, adb_app: str = None, kill_every_task: bool = True
) -> bool:
    """Kill every app.

    Parameters:
    - device_serial (str): The android device serial number.
    - adb_app (str): The application package name.
    - kill_every_task (bool): Whether to kill all running apps.

    Returns:
    - bool: Whether the app is successfully closed.
    """
    if kill_every_task:
        execute_adb(
            f'''adb -s {device_serial} shell "dumpsys activity | grep topActivity | sed -n 's/.*{{\\([^\\/]*\\)\\/.*/\\1/p' | while read -r package; do if [ -n \\"\\$package\\" ]; then am force-stop \\"\\$package\\"; fi; done"'''
        )
    # Kill specific app
    if adb_app:
        output = execute_adb(f"adb -s {device_serial} shell am force-stop {adb_app}")
        if output != "ERROR":
            time.sleep(5)
            return True
    return False


def set_adb_keyboard(device_serial):
    execute_adb(
        f"adb -s {device_serial} shell ime enable com.android.adbkeyboard/.AdbIME"
    )
    execute_adb(f"adb -s {device_serial} shell ime set com.android.adbkeyboard/.AdbIME")


def set_default_keyboard(device_serial, package):
    execute_adb(f"adb -s {device_serial} shell ime set {package}")


def setup_output_directory(
    results_dir: str, session_id: str, overwrite_session: bool, config_path: str = None
) -> str:
    output_dir = os.path.join(results_dir, f"session-{session_id}")

    if os.path.exists(output_dir):
        if overwrite_session:
            # Directory exists, prompt the user
            response = (
                input(
                    f"The results session <{session_id}> already exists. Do you want to erase its contents and restart the session? (y/n): "
                )
                .strip()
                .lower()
            )
            if response in ["yes", "y"]:
                # Erase the contents
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            else:
                pass
        else:
            pass
    else:
        # Create the directory
        os.makedirs(output_dir)

    # Save a copy of config.yaml to the output directory for future reference
    if config_path is None:
        # Auto-detect config.yaml location
        config_path = os.path.join(os.getcwd(), "config.yaml")

    if os.path.exists(config_path):
        config_backup_path = os.path.join(output_dir, "config.yaml")
        # Only copy if config.yaml doesn't exist in output_dir or if we're overwriting
        if not os.path.exists(config_backup_path) or overwrite_session:
            shutil.copy2(config_path, config_backup_path)
            print(f"✓ Configuration backup saved to: {config_backup_path}")
    else:
        print(f"⚠ Warning: config.yaml not found at {config_path}, skipping backup")

    return output_dir


def get_results_csv_path(output_dir: str) -> str:
    return os.path.join(output_dir, "results.csv")


def get_results_df(output_dir: str) -> pd.DataFrame:
    """Safely reads the results CSV into a DataFrame."""
    csv_path = get_results_csv_path(output_dir)
    return pd.read_csv(csv_path)


def get_col_name_from_template(
    template_name: str,
    agent_name: str = None,
    eval_name: str = None,
    sub_eval_name: str = None,
    attempt_num: int = None,
):
    """Generates a column name based on a template."""
    parts = []
    if agent_name:
        parts.append(agent_name)
    if eval_name:
        parts.append(eval_name)
    if sub_eval_name:
        parts.append(sub_eval_name)
    if attempt_num:
        parts.append(f"attempt_{attempt_num}")

    # Only append the main metric name (template_name) if it's not empty
    if template_name:
        parts.append(template_name)
    return "_".join(parts)


def get_exec_json_path(
    output_dir: str, task_id: str, agent_name: str, content: str
) -> str:
    return os.path.join(output_dir, task_id, agent_name, f"{content}.json")


def with_filelock(timeout: int = 30):
    """Decorator to add a simple file lock context to the wrapped function using the 'output_dir'
    argument.

    Args:
        timeout: Maximum seconds to wait for the lock (default 30). If lock cannot
                 be acquired within this time, it is treated as stale and removed.

    Returns:
    - A decorated function with a file lock.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            output_dir = kwargs.get("output_dir") or next(
                (
                    arg
                    for arg_name, arg in zip(func.__code__.co_varnames, args)
                    if arg_name == "output_dir"
                ),
                None,
            )

            if not output_dir:
                raise ValueError("Lock path argument output_dir is required.")

            csv_path = get_results_csv_path(output_dir)
            lock_path = csv_path + ".lock"
            lock = FileLock(lock_path, timeout=timeout)

            try:
                with lock:
                    result = func(*args, **kwargs)
                    return result
            except Timeout:
                print(f"WARNING: Lock file {lock_path} is held by another process "
                      f"(timeout={timeout}s). Removing stale lock and retrying...")
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                lock2 = FileLock(lock_path, timeout=timeout)
                with lock2:
                    result = func(*args, **kwargs)
                    return result

        return wrapper

    return decorator


def try_save_csv(
    dataframe: pd.DataFrame, path: str, max_retry: int = 5, retry_interval: int = 5
) -> bool:
    counter = 0
    while True:
        try:
            dataframe.to_csv(path, encoding="utf-8", index=False)
        except Exception as err:
            print("Failed to save to ", path)
            print(str(err))
            if counter < max_retry:
                counter += 1
                print(f"Retry in {retry_interval} seconds; {counter}/{max_retry}")
                time.sleep(retry_interval)
                continue
            else:
                return False
        return True


def _try_load_json(x):
    try:
        return json.loads(x.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_col_default(col: str) -> str:
    """Return a sensible default value for a results column based on its name."""
    if "successful_attempts" in col:
        return "[]"
    if "success_count" in col:
        return "0"
    if col.endswith("_completion") or col.endswith("_evaluation") or col.endswith("_exec_error"):
        return "N"
    if col.endswith("_device"):
        return "N"
    if col.endswith("_exit_code") or col.endswith("_finish_signal") or col.endswith("_failure_step"):
        return "-1"
    if any(x in col for x in ["_step_ratio", "_avg_prompt_tokens", "_avg_completion_tokens",
                                "_elapsed_time", "_total_steps", "_total_time", "_total_token_cost",
                                "_eval_prompt_tokens", "_eval_completion_tokens", "_eval_total_tokens",
                                "_eval_api_cost", "_step_desc_prompt_tokens", "_step_desc_completion_tokens",
                                "_step_desc_total_tokens", "_step_desc_api_cost", "_final_decision_prompt_tokens",
                                "_final_decision_completion_tokens", "_final_decision_total_tokens",
                                "_final_decision_api_cost"]):
        return "0.0"
    if any(x in col for x in ["_model_name", "_model_provider", "_evaluation_method",
                                "_task_feasible", "_task_feasible_reason", "_task_barriers"]):
        return ""
    if col.endswith("_details") or col.endswith("_step_analysis"):
        return "{}"
    if "reasonable_steps" in col or "unreasonable_steps" in col:
        return "[]"
    return ""


def _merge_missing_tasks(
    results_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    missing_ids: set,
    agent_list: list,
    max_attempts: int,
    reasoning_mode: str,
    action_mode: str,
) -> pd.DataFrame:
    """
    Merge tasks that exist in dataset_df but are missing from results_df.

    Preserves all existing results. For each missing task, fills in base columns
    from dataset_df and initializes all attempt/evaluation columns with defaults.
    """
    dataset_map = dataset_df.set_index("task_identifier").to_dict("index")
    results_cols = set(results_df.columns)

    for tid in missing_ids:
        if tid not in dataset_map:
            continue
        row_data = dataset_map[tid]

        new_row = {}
        for col in results_df.columns:
            if col == "task_identifier":
                new_row[col] = tid
            elif col in row_data:
                new_row[col] = row_data[col]
            elif col in dataset_df.columns:
                new_row[col] = row_data.get(col, "")
            else:
                new_row[col] = _get_col_default(col)

        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    return results_df


@with_filelock()
def setup_results_csv(
    output_dir: str,
    dataset_path: str,
    agent_list: list[str],
    max_attempts: int,
    reasoning_mode: str,
    action_mode: str,
) -> pd.DataFrame:
    """Setup the CSV file for storing results.

    This function will create a new results.csv file or load an existing one.
    It adds necessary columns for each agent and each attempt.

    Parameters:
    - output_dir (str): The directory where the results are stored.
    - dataset_path (str): The path to the original dataset CSV.
    - agent_list (list[str]): A list of agent names.
    - max_attempts (int): Maximum number of attempts for each task.

    Returns:
    - pd.DataFrame: The initialized or loaded DataFrame.
    """
    csv_path = get_results_csv_path(output_dir)
    if os.path.exists(csv_path):
        print("Loaded existing results.csv")
        for encoding in ["utf-8", "gbk", "gb18030", "utf-8-sig", "latin1"]:
            try:
                results_df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        # Check if dataset_path has tasks missing from results.csv
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
            existing_ids = set(results_df["task_identifier"].astype(str))
            dataset_ids = set(dataset_df["task_identifier"].astype(str))
            missing_ids = dataset_ids - existing_ids

            if missing_ids:
                print(
                    f"WARNING: results.csv has {len(existing_ids)} tasks, "
                    f"dataset has {len(dataset_ids)} tasks. "
                    f"Adding {len(missing_ids)} missing tasks."
                )
                results_df = _merge_missing_tasks(
                    results_df,
                    dataset_df,
                    missing_ids,
                    agent_list,
                    max_attempts,
                    reasoning_mode,
                    action_mode,
                )
                try_save_csv(results_df, csv_path)
                print(f"Updated results.csv with {len(results_df)} total tasks.")
        else:
            print(f"WARNING: dataset_path {dataset_path} not found, using existing results.csv as-is.")

        return results_df
    else:
        print("Created results.csv")
        results_df = pd.read_csv(dataset_path, keep_default_na=False)
        results_df.set_index("task_identifier", inplace=True)

        for agent_name in agent_list:
            # Add success tracking columns
            results_df[f"{agent_name}_successful_attempts"] = (
                "[]"  # Array of successful attempt numbers
            )
            results_df[f"{agent_name}_success_count"] = 0  # Total number of successes

            for i in range(1, max_attempts + 1):
                # Columns for each attempt
                exec_col_prefix = get_col_name_from_template(
                    "", agent_name=agent_name, attempt_num=i
                )
                eval_col_prefix = get_col_name_from_template(
                    "",
                    agent_name=agent_name,
                    eval_name=reasoning_mode,
                    sub_eval_name=action_mode,
                    attempt_num=i,
                )

                results_df[f"{exec_col_prefix}_completion"] = "N"
                results_df[f"{exec_col_prefix}_device"] = "N"
                results_df[f"{exec_col_prefix}_exit_code"] = -1
                results_df[f"{exec_col_prefix}_total_steps"] = 0
                results_df[f"{exec_col_prefix}_total_token_cost"] = 0.0
                results_df[f"{exec_col_prefix}_total_time"] = 0.0
                results_df[f"{exec_col_prefix}_finish_signal"] = 0
                results_df[f"{exec_col_prefix}_step_ratio"] = 0.0
                results_df[f"{exec_col_prefix}_elapsed_time_initial"] = 0.0
                results_df[f"{exec_col_prefix}_elapsed_time_exec"] = 0.0
                results_df[f"{exec_col_prefix}_avg_prompt_tokens"] = 0
                results_df[f"{exec_col_prefix}_avg_completion_tokens"] = 0
                results_df[f"{exec_col_prefix}_exec_error"] = "N"

                results_df[f"{eval_col_prefix}_evaluation"] = "N"
                results_df[f"{eval_col_prefix}_details"] = "{}"
                results_df[f"{eval_col_prefix}_evaluation_method"] = ""

                # 总计token使用情况
                results_df[f"{eval_col_prefix}_eval_prompt_tokens"] = 0
                results_df[f"{eval_col_prefix}_eval_completion_tokens"] = 0
                results_df[f"{eval_col_prefix}_eval_total_tokens"] = 0
                results_df[f"{eval_col_prefix}_eval_api_cost"] = 0.0
                results_df[f"{eval_col_prefix}_model_provider"] = ""
                results_df[f"{eval_col_prefix}_model_name"] = ""

                # 步骤描述生成的token使用情况
                results_df[f"{eval_col_prefix}_step_desc_prompt_tokens"] = 0
                results_df[f"{eval_col_prefix}_step_desc_completion_tokens"] = 0
                results_df[f"{eval_col_prefix}_step_desc_total_tokens"] = 0
                results_df[f"{eval_col_prefix}_step_desc_api_cost"] = 0.0
                results_df[f"{eval_col_prefix}_step_desc_model_name"] = ""
                results_df[f"{eval_col_prefix}_step_desc_model_provider"] = ""

                # 最终决策的token使用情况
                results_df[f"{eval_col_prefix}_final_decision_prompt_tokens"] = 0
                results_df[f"{eval_col_prefix}_final_decision_completion_tokens"] = 0
                results_df[f"{eval_col_prefix}_final_decision_total_tokens"] = 0
                results_df[f"{eval_col_prefix}_final_decision_api_cost"] = 0.0
                results_df[f"{eval_col_prefix}_final_decision_model_name"] = ""
                results_df[f"{eval_col_prefix}_final_decision_model_provider"] = ""

                # 步骤合理性分析
                results_df[f"{eval_col_prefix}_reasonable_steps"] = ""
                results_df[f"{eval_col_prefix}_unreasonable_steps"] = ""
                results_df[f"{eval_col_prefix}_step_analysis"] = "{}"

                # 任务合理性评估
                results_df[f"{eval_col_prefix}_task_feasible"] = "N"
                results_df[f"{eval_col_prefix}_task_feasible_reason"] = ""
                results_df[f"{eval_col_prefix}_task_barriers"] = ""

        results_df.reset_index(inplace=True)
        try_save_csv(results_df, csv_path)
        return results_df


def save_task_metadata(attempt_dir: str, task) -> None:
    """
    将数据集中的任务级元数据保存到 attempt 目录下的 task_metadata.json，
    使下游分析脚本无需依赖 results.csv。

    仅保存 evaluation_summary.json / final_decision.json 等已有 JSON 中
    **不包含**的字段，避免重复。
    """
    # 需要保存的字段列表（evaluation_summary 中已有 task_identifier / task_description，
    # final_decision 中已有 task_feasible 等，这里不再重复）
    FIELDS_TO_SAVE = [
        "app_name",
        "app_package",
        "golden_steps",
        "trajectory_id",
        "original_goal",
        "task_reasonable",
        "task_completed",
        "task_id",
        "difficulty_level",
        "core_functionality",
        "variation_type",
        "prerequisites",
    ]

    metadata = {}
    for field in FIELDS_TO_SAVE:
        val = getattr(task, field, None)
        if val is not None:
            # pandas 可能返回 numpy 类型，统一转为 Python 原生类型
            if hasattr(val, "item"):
                val = val.item()
            metadata[field] = val

    if not metadata:
        return

    meta_path = os.path.join(attempt_dir, "task_metadata.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save task_metadata.json to {attempt_dir}: {e}")


def ensure_fallback_log_json(attempt_dir: str) -> None:
    """
    当 benchmark_run.py 因超时/OOM 被 kill 后，attempt 目录只有截图、没有 log.json。
    此函数检查 log.json 是否存在，不存在则生成一个最小兜底版本，让 evaluator 能正常跑（记为失败）。
    """
    log_json_path = os.path.join(attempt_dir, "log.json")
    if os.path.exists(log_json_path):
        return

    if not os.path.isdir(attempt_dir):
        return

    png_files = [f for f in os.listdir(attempt_dir) if f.endswith(".png")]
    total_steps = len(png_files)

    fallback_log = [
        {
            "total_steps": total_steps,
            "finish_signal": 0,
            "elapsed_time_initial": 0.0,
            "elapsed_time_exec": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "error": "Process killed (timeout or OOM), no log.json generated",
        }
    ]

    try:
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(fallback_log, f, ensure_ascii=False, indent=2)
        print(f"Generated fallback log.json at {log_json_path} (total_steps={total_steps})")
    except Exception as e:
        print(f"Warning: Failed to write fallback log.json to {log_json_path}: {e}")


@with_filelock()
def save_result__completed_execution(
    output_dir: str,
    task_id: str,
    agent_name: str,
    task_completed: bool,
    exit_code: int,
    device: str,
    attempt_num: int,
) -> pd.DataFrame:
    """Save the task execution result to the CSV.

    This function is decorated with a file lock to ensure thread/process safety.
    """
    df = pd.read_csv(get_results_csv_path(output_dir))
    df.set_index("task_identifier", inplace=True)
    row_index = df.index.get_loc(task_id)

    prefix = get_col_name_from_template(
        "", agent_name=agent_name, attempt_num=attempt_num
    )
    df.loc[task_id, f"{prefix}_completion"] = "Y" if task_completed else "N"
    df.loc[task_id, f"{prefix}_exit_code"] = exit_code
    df.loc[task_id, f"{prefix}_device"] = device

    # Also log execution summary if exists
    log_path = os.path.join(
        output_dir, task_id, agent_name, f"attempt_{attempt_num}", "log.json"
    )
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_data = json.load(f)
            summary = log_data[-1]  # The summary is the last element
            df.loc[task_id, f"{prefix}_total_steps"] = summary.get("total_steps", 0)
            df.loc[task_id, f"{prefix}_finish_signal"] = summary.get("finish_signal", 0)
            df.loc[task_id, f"{prefix}_elapsed_time_initial"] = summary.get(
                "elapsed_time_initial", 0.0
            )
            df.loc[task_id, f"{prefix}_elapsed_time_exec"] = summary.get(
                "elapsed_time_exec", 0.0
            )

            # Calculate and record additional metrics
            total_prompt_tokens = summary.get("total_prompt_tokens", 0)
            total_completion_tokens = summary.get("total_completion_tokens", 0)
            total_steps = summary.get("total_steps", 0)
            elapsed_time_total = summary.get("elapsed_time_initial", 0.0) + summary.get(
                "elapsed_time_exec", 0.0
            )

            # Record token-related metrics
            df.loc[task_id, f"{prefix}_avg_prompt_tokens"] = (
                total_prompt_tokens // total_steps if total_steps > 0 else 0
            )
            df.loc[task_id, f"{prefix}_avg_completion_tokens"] = (
                total_completion_tokens // total_steps if total_steps > 0 else 0
            )

            # Calculate total token cost (basic estimation, adjust based on actual pricing)
            # Using rough estimates: $0.01 per 1K prompt tokens, $0.03 per 1K completion tokens
            prompt_cost = (total_prompt_tokens / 1000) * 0.01
            completion_cost = (total_completion_tokens / 1000) * 0.03
            df.loc[task_id, f"{prefix}_total_token_cost"] = (
                prompt_cost + completion_cost
            )

            # Record total time
            df.loc[task_id, f"{prefix}_total_time"] = elapsed_time_total

            # Calculate step ratio (ratio of actual steps to some expected maximum)
            # This could be based on golden_steps from the dataset if available
            # For now, using a simple completion ratio based on finish_signal
            finish_signal = summary.get("finish_signal", 0)
            if finish_signal == 1:  # Task completed successfully
                df.loc[task_id, f"{prefix}_step_ratio"] = 1.0
            else:
                # Calculate based on actual steps vs some reasonable maximum
                max_expected_steps = 50  # Reasonable maximum for most tasks
                df.loc[task_id, f"{prefix}_step_ratio"] = min(
                    total_steps / max_expected_steps, 1.0
                )

    df.reset_index(inplace=True)
    try_save_csv(df, get_results_csv_path(output_dir))
    return df


@with_filelock()
def save_result__completed_evaluation(
    output_dir: str,
    task_id: str,
    agent_name: str,
    success: int,
    evaluation_detail: dict,
    reasoning_mode: str,
    action_mode: str,
    attempt_num: int,
    evaluation_method: str = "",
    # 移除总计token使用参数，只保留分类的token统计
    # eval_prompt_tokens: int = 0,
    # eval_completion_tokens: int = 0,
    # eval_total_tokens: int = 0,
    # eval_api_cost: float = 0.0,
    # model_provider: str = "",
    # model_name: str = "",
    # 步骤描述生成的token使用
    step_desc_prompt_tokens: int = 0,
    step_desc_completion_tokens: int = 0,
    step_desc_total_tokens: int = 0,
    step_desc_api_cost: float = 0.0,
    step_desc_model_name: str = "",
    step_desc_model_provider: str = "",
    # 最终决策的token使用
    final_decision_prompt_tokens: int = 0,
    final_decision_completion_tokens: int = 0,
    final_decision_total_tokens: int = 0,
    final_decision_api_cost: float = 0.0,
    final_decision_model_name: str = "",
    final_decision_model_provider: str = "",
    # 失败步骤追踪
    failure_step: int = None,
    # 步骤合理性分析
    reasonable_steps: list = None,
    unreasonable_steps: list = None,
    step_analysis: dict = None,
    # 任务合理性评估
    task_feasible: bool = None,
    task_feasible_reason: str = "",
    task_barriers: list = None,
) -> pd.DataFrame:
    """Save the task evaluation result to the CSV.

    This function is decorated with a file lock to ensure thread/process safety.
    It converts numeric success codes (1, 0, -1) to string representations ('S', 'F', 'E').
    """
    df = pd.read_csv(get_results_csv_path(output_dir))
    df.set_index("task_identifier", inplace=True)

    prefix = get_col_name_from_template(
        "",
        agent_name=agent_name,
        eval_name=reasoning_mode,
        sub_eval_name=action_mode,
        attempt_num=attempt_num,
    )

    result_map = {1: "S", 0: "F", -1: "E"}
    evaluation_result = result_map.get(success, "E")  # Default to Error

    df.loc[task_id, f"{prefix}_evaluation"] = evaluation_result
    df.loc[task_id, f"{prefix}_details"] = str(evaluation_detail)
    df.loc[task_id, f"{prefix}_evaluation_method"] = evaluation_method

    # 移除总计token使用情况的存储，只保留分类的token统计
    # df.loc[task_id, f"{prefix}_eval_prompt_tokens"] = eval_prompt_tokens
    # df.loc[task_id, f"{prefix}_eval_completion_tokens"] = eval_completion_tokens
    # df.loc[task_id, f"{prefix}_eval_total_tokens"] = eval_total_tokens
    # df.loc[task_id, f"{prefix}_eval_api_cost"] = eval_api_cost
    # df.loc[task_id, f"{prefix}_model_provider"] = model_provider
    # df.loc[task_id, f"{prefix}_model_name"] = model_name

    # 步骤描述生成的token使用情况
    df.loc[task_id, f"{prefix}_step_desc_prompt_tokens"] = step_desc_prompt_tokens
    df.loc[task_id, f"{prefix}_step_desc_completion_tokens"] = (
        step_desc_completion_tokens
    )
    df.loc[task_id, f"{prefix}_step_desc_total_tokens"] = step_desc_total_tokens
    df.loc[task_id, f"{prefix}_step_desc_api_cost"] = step_desc_api_cost
    df.loc[task_id, f"{prefix}_step_desc_model_name"] = step_desc_model_name
    df.loc[task_id, f"{prefix}_step_desc_model_provider"] = step_desc_model_provider

    # 最终决策的token使用情况
    df.loc[task_id, f"{prefix}_final_decision_prompt_tokens"] = (
        final_decision_prompt_tokens
    )
    df.loc[task_id, f"{prefix}_final_decision_completion_tokens"] = (
        final_decision_completion_tokens
    )
    df.loc[task_id, f"{prefix}_final_decision_total_tokens"] = (
        final_decision_total_tokens
    )
    df.loc[task_id, f"{prefix}_final_decision_api_cost"] = final_decision_api_cost
    df.loc[task_id, f"{prefix}_final_decision_model_name"] = final_decision_model_name
    df.loc[task_id, f"{prefix}_final_decision_model_provider"] = (
        final_decision_model_provider
    )

    # 失败步骤追踪
    df.loc[task_id, f"{prefix}_failure_step"] = (
        failure_step if failure_step is not None else ""
    )

    # 步骤合理性分析
    df.loc[task_id, f"{prefix}_reasonable_steps"] = (
        str(reasonable_steps) if reasonable_steps is not None else ""
    )
    df.loc[task_id, f"{prefix}_unreasonable_steps"] = (
        str(unreasonable_steps) if unreasonable_steps is not None else ""
    )
    df.loc[task_id, f"{prefix}_step_analysis"] = (
        str(step_analysis) if step_analysis is not None else "{}"
    )

    # 任务合理性评估
    df.loc[task_id, f"{prefix}_task_feasible"] = (
        "Y" if task_feasible else "N" if task_feasible is not None else "N"
    )
    df.loc[task_id, f"{prefix}_task_feasible_reason"] = task_feasible_reason
    df.loc[task_id, f"{prefix}_task_barriers"] = (
        str(task_barriers) if task_barriers is not None else ""
    )

    df.reset_index(inplace=True)
    try_save_csv(df, get_results_csv_path(output_dir))
    return df


@with_filelock()
def update_success_tracking(
    output_dir: str, task_id: str, agent_name: str, attempt_num: int
) -> None:
    """Updates the success tracking columns for a given task and agent."""
    df = pd.read_csv(get_results_csv_path(output_dir))
    df.set_index("task_identifier", inplace=True)

    successful_attempts_col = f"{agent_name}_successful_attempts"
    success_count_col = f"{agent_name}_success_count"

    # Get current successful attempts list
    current_attempts_str = df.loc[task_id, successful_attempts_col]
    try:
        current_attempts = (
            json.loads(current_attempts_str)
            if current_attempts_str and current_attempts_str != "[]"
            else []
        )
    except (json.JSONDecodeError, TypeError):
        current_attempts = []

    # Add the new successful attempt if not already present
    if attempt_num not in current_attempts:
        current_attempts.append(attempt_num)
        current_attempts.sort()  # Keep attempts sorted

        # Update both columns
        df.loc[task_id, successful_attempts_col] = json.dumps(current_attempts)
        df.loc[task_id, success_count_col] = len(current_attempts)

        print(
            f"Updated success tracking for task {task_id}: successful attempts {current_attempts}, total successes: {len(current_attempts)}"
        )

    df.reset_index(inplace=True)
    try_save_csv(df, get_results_csv_path(output_dir))


def print_execution_summary(output_dir, agent_scope):
    df = pd.read_csv(get_results_csv_path(output_dir))
    for agent in agent_scope:
        print(f"Agent <{agent}>:")

        # Count total tasks for this agent
        total_tasks = len(df)

        # Count completed tasks (attempt 1)
        completion_col = get_col_name_from_template(
            "completion", agent_name=agent, attempt_num=1
        )
        if completion_col in df.columns:
            completed_count = df[completion_col].eq("Y").sum()
        else:
            completed_count = 0

        # Count abnormal exits (attempt 1)
        exit_code_col = get_col_name_from_template(
            "exit_code", agent_name=agent, attempt_num=1
        )
        if exit_code_col in df.columns:
            abnormal_exit_count = df[exit_code_col].ne(0).sum()
        else:
            abnormal_exit_count = 0

        print(f"  - Completed: {completed_count} / {total_tasks}")
        print(f"  - Abnormal exit: {abnormal_exit_count} / {total_tasks}")

        # Show success tracking if available
        success_count_col = f"{agent}_success_count"
        if success_count_col in df.columns:
            successful_tasks = df[success_count_col].gt(0).sum()
            print(f"  - Successful tasks: {successful_tasks} / {total_tasks}")
        else:
            print(f"  - No success tracking data available")


def print_evaluation_summary(output_dir, agent_scope, max_attempts):
    """Prints a summary of the evaluation results, including pass@k."""
    csv_path = get_results_csv_path(output_dir)
    df = pd.read_csv(csv_path)
    print("\n--- Evaluation Summary ---")
    for agent_name in agent_scope:
        print(f"Agent <{agent_name}>:")
        success_count_col = f"{agent_name}_success_count"
        successful_attempts_col = f"{agent_name}_successful_attempts"

        if success_count_col not in df.columns:
            print(f"  - No success tracking data available.")
            continue

        total_tasks = len(df)
        successful_tasks = df[success_count_col] > 0
        print(f"  - Overall Success Rate: {successful_tasks.sum() / total_tasks:.2%}")

        # Calculate pass@k based on successful attempts
        for k in range(1, max_attempts + 1):
            pass_at_k_count = 0
            for idx, row in df.iterrows():
                try:
                    attempts_str = row[successful_attempts_col]
                    successful_attempts = (
                        json.loads(attempts_str)
                        if attempts_str and attempts_str != "[]"
                        else []
                    )
                    # Check if any successful attempt is within the first k attempts
                    if any(attempt <= k for attempt in successful_attempts):
                        pass_at_k_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            pass_rate = pass_at_k_count / total_tasks
            print(f"  - Pass@{k}: {pass_at_k_count}/{total_tasks} ({pass_rate:.2%})")

        # Calculate detailed success counts
        print(f"  --- Detailed Success Counts (max_attempts = {max_attempts}) ---")
        success_counts = df[success_count_col]

        for i in range(max_attempts + 1):
            succeeded_n_times = (success_counts == i).sum()
            succeeded_n_times_rate = succeeded_n_times / total_tasks
            plural = "s" if i != 1 else ""
            print(
                f"    - Succeeded {i} time{plural}: {succeeded_n_times}/{total_tasks} ({succeeded_n_times_rate:.2%})"
            )

    print(f"\nEvaluation results have been saved to: {os.path.abspath(csv_path)}")


def is_task_completed(
    output_dir: str,
    task_id: str,
    agent_name: str,
    max_attempts: int,
    reasoning_mode: str,
    action_mode: str,
) -> bool:
    """
    Check if a task is already completed (either succeeded in ≤k attempts or failed all k attempts).

    Parameters:
    - output_dir (str): The directory where the results are stored.
    - task_id (str): The task identifier.
    - agent_name (str): The agent name.
    - max_attempts (int): Maximum number of attempts.
    - reasoning_mode (str): The reasoning mode.
    - action_mode (str): The action mode.

    Returns:
    - bool: True if the task is completed, False otherwise.
    """
    try:
        df = pd.read_csv(get_results_csv_path(output_dir))
        task_row = df[df["task_identifier"] == task_id]

        if task_row.empty:
            return False

        # Check if task already succeeded (has any successful attempts)
        success_count_col = f"{agent_name}_success_count"
        if success_count_col in df.columns:
            success_count = task_row.iloc[0][success_count_col]
            if success_count > 0:
                return True

        # Check if all attempts have been executed
        all_attempts_completed = True
        for attempt in range(1, max_attempts + 1):
            eval_col_prefix = get_col_name_from_template(
                "",
                agent_name=agent_name,
                eval_name=reasoning_mode,
                sub_eval_name=action_mode,
                attempt_num=attempt,
            )
            eval_col_name = f"{eval_col_prefix}_evaluation"

            if eval_col_name not in df.columns:
                all_attempts_completed = False
                break

            eval_result = task_row.iloc[0][eval_col_name]
            if (
                eval_result == "N" or eval_result == "E"
            ):  # Not evaluated yet or evaluation failed
                all_attempts_completed = False
                break

        return all_attempts_completed

    except Exception as e:
        print(f"Error checking task completion for {task_id}: {e}")
        return False


def clear_task_results(
    output_dir: str,
    task_id: str,
    agent_name: str,
    max_attempts: int,
    reasoning_mode: str,
    action_mode: str,
) -> None:
    """
    Clear all results for a specific task and agent.

    Parameters:
    - output_dir (str): The directory where the results are stored.
    - task_id (str): The task identifier.
    - agent_name (str): The agent name.
    - max_attempts (int): Maximum number of attempts.
    - reasoning_mode (str): The reasoning mode.
    - action_mode (str): The action mode.
    """
    try:
        # Clear CSV results
        df = pd.read_csv(get_results_csv_path(output_dir))
        df.set_index("task_identifier", inplace=True)

        if task_id not in df.index:
            print(f"Task {task_id} not found in results CSV.")
            return

        # Reset success tracking columns
        successful_attempts_col = f"{agent_name}_successful_attempts"
        success_count_col = f"{agent_name}_success_count"
        if successful_attempts_col in df.columns:
            df.loc[task_id, successful_attempts_col] = "[]"
        if success_count_col in df.columns:
            df.loc[task_id, success_count_col] = 0

        # Reset all attempt results
        for attempt in range(1, max_attempts + 1):
            # Reset execution columns
            exec_col_prefix = get_col_name_from_template(
                "", agent_name=agent_name, attempt_num=attempt
            )
            exec_columns = [
                f"{exec_col_prefix}_completion",
                f"{exec_col_prefix}_device",
                f"{exec_col_prefix}_exit_code",
                f"{exec_col_prefix}_total_steps",
                f"{exec_col_prefix}_total_token_cost",
                f"{exec_col_prefix}_total_time",
                f"{exec_col_prefix}_finish_signal",
                f"{exec_col_prefix}_step_ratio",
                f"{exec_col_prefix}_elapsed_time_initial",
                f"{exec_col_prefix}_elapsed_time_exec",
                f"{exec_col_prefix}_avg_prompt_tokens",
                f"{exec_col_prefix}_avg_completion_tokens",
                f"{exec_col_prefix}_exec_error",
            ]

            for col in exec_columns:
                if col in df.columns:
                    if (
                        col.endswith("_completion")
                        or col.endswith("_device")
                        or col.endswith("_exec_error")
                    ):
                        df.loc[task_id, col] = "N"
                    elif col.endswith("_exit_code"):
                        df.loc[task_id, col] = -1
                    else:
                        df.loc[task_id, col] = (
                            0 if col.endswith(("_steps", "_signal")) else 0.0
                        )

            # Reset evaluation columns
            eval_col_prefix = get_col_name_from_template(
                "",
                agent_name=agent_name,
                eval_name=reasoning_mode,
                sub_eval_name=action_mode,
                attempt_num=attempt,
            )
            eval_columns = [
                f"{eval_col_prefix}_evaluation",
                f"{eval_col_prefix}_details",
                f"{eval_col_prefix}_evaluation_method",
            ]

            for col in eval_columns:
                if col in df.columns:
                    if col.endswith("_evaluation"):
                        df.loc[task_id, col] = "N"
                    elif col.endswith("_evaluation_method"):
                        df.loc[task_id, col] = ""
                    else:  # details
                        df.loc[task_id, col] = "{}"

        df.reset_index(inplace=True)
        try_save_csv(df, get_results_csv_path(output_dir))

        # Clear output directories
        task_output_dir = os.path.join(output_dir, task_id, agent_name)
        if os.path.exists(task_output_dir):
            shutil.rmtree(task_output_dir)
            print(f"Cleared output directory: {task_output_dir}")

        print(f"Cleared all results for task {task_id} with agent {agent_name}")

    except Exception as e:
        print(f"Error clearing task results for {task_id}: {e}")


def execute_evaluation(
    task_identifier,
    output_dir,
    mode,
    agent_name,
    attempt,
    reasoning_mode,
    action_mode,
    attempt_dir,
    result_overwrite=False,
    self_hint_enabled=False,
):
    """
    执行单个评估任务
    """
    # 如果不是覆盖模式，先检查是否已有评估结果
    if not result_overwrite:
        current_results_df = get_results_df(output_dir)
        result_col_prefix = get_col_name_from_template(
            "",
            agent_name=agent_name,
            eval_name=reasoning_mode,
            sub_eval_name=action_mode,
            attempt_num=attempt,
        )
        result_col_name = f"{result_col_prefix}_evaluation"

        task_row = current_results_df[
            current_results_df["task_identifier"] == task_identifier
        ]

        if not task_row.empty and result_col_name in task_row.columns:
            eval_result_val = task_row.iloc[0][result_col_name]
            if eval_result_val in ["S", "F", "E"]:  # 已有评估结果
                print(
                    f"Task {task_identifier} agent {agent_name} attempt {attempt} already has evaluation result: {eval_result_val}. Skipping evaluation."
                )
                return True  # 返回True表示不需要重新评估

    # Check if log.json exists before attempting evaluation
    log_json_path = os.path.join(attempt_dir, "log.json")
    if not os.path.exists(log_json_path):
        print(
            f"No log.json found at {log_json_path}. Skipping evaluation for attempt {attempt}."
        )
        return False

    # Get conda path from config_loader
    from config_loader import get_config

    config = get_config(verbose=False)
    conda_path = config["CONDA_PATH"]

    # 构建包含conda路径的命令
    command = (
        f'export PATH="{conda_path}/bin:$PATH" && conda run -n MobileForge python {os.path.join(os.getcwd(), "mobilegym_critic/evaluator.py")} '
        f"--task_identifier {task_identifier} "
        f"--result_dir {output_dir} "
        f"--mode {mode} "
        f"--agent {agent_name} "
        f"--attempt_num {attempt} "
        f"--reasoning_mode {reasoning_mode} "
        f"--action_mode {action_mode}"
    )
    
    # Add self_hint_enabled flag if enabled
    if self_hint_enabled:
        command += " --self_hint_enabled"
    print(f"Evaluating task: {task_identifier}, attempt: {attempt}...")

    eval_process = subprocess.run(command, shell=True, capture_output=True, text=True)

    if eval_process.returncode != 0:
        print(
            f"Evaluation script for attempt {attempt} failed with exit code {eval_process.returncode}."
        )
        print(f"Stderr: {eval_process.stderr}")
        return False

    return True


def immediate_evaluate_and_update_pass_at_k(
    output_dir,
    task_identifier,
    agent_name,
    attempt,
    reasoning_mode,
    action_mode,
    result_overwrite=False,
    self_hint_enabled=False,
):
    """
    立即评估任务并更新pass@k状态
    """
    attempt_dir = os.path.join(
        output_dir, task_identifier, agent_name, f"attempt_{attempt}"
    )

    # 执行评估
    eval_success = execute_evaluation(
        task_identifier,
        output_dir,
        "full",
        agent_name,
        attempt,
        reasoning_mode,
        action_mode,
        attempt_dir,
        result_overwrite,
        self_hint_enabled,
    )

    if not eval_success:
        print(f"Evaluation failed for task {task_identifier}, attempt {attempt}")
        return False

    # 读取评估结果
    current_results_df = get_results_df(output_dir)
    result_col_prefix = get_col_name_from_template(
        "",
        agent_name=agent_name,
        eval_name=reasoning_mode,
        sub_eval_name=action_mode,
        attempt_num=attempt,
    )
    result_col_name = f"{result_col_prefix}_evaluation"

    task_row = current_results_df[
        current_results_df["task_identifier"] == task_identifier
    ]

    if task_row.empty or result_col_name not in task_row.columns:
        print(
            f"Could not find result for task {task_identifier} attempt {attempt} in CSV."
        )
        return False

    eval_result_val = task_row.iloc[0][result_col_name]

    if eval_result_val == "S":
        print(f"Task {task_identifier} Attempt {attempt} was successful!")
        update_success_tracking(output_dir, task_identifier, agent_name, attempt)
        return True
    elif eval_result_val == "E":
        print(
            f"Task {task_identifier} Attempt {attempt} resulted in an evaluation error."
        )
    elif eval_result_val == "F":
        print(f"Task {task_identifier} Attempt {attempt} failed evaluation.")
    else:
        print(
            f"Task {task_identifier} Attempt {attempt} has unknown evaluation result: {eval_result_val}"
        )

    return False


def get_valid_attempts(output_dir, task_identifier, agent_name, max_attempts):
    """
    Get list of valid attempts (those with log.json files) for a given task and agent.
    Returns a list of attempt numbers.
    """
    valid_attempts = []
    task_dir = os.path.join(output_dir, task_identifier, agent_name)

    if not os.path.exists(task_dir):
        return valid_attempts

    for attempt in range(1, max_attempts + 1):
        attempt_dir = os.path.join(task_dir, f"attempt_{attempt}")
        log_json_path = os.path.join(attempt_dir, "log.json")

        if os.path.exists(log_json_path):
            valid_attempts.append(attempt)

    return valid_attempts


def get_attempt_status(
    output_dir, task_identifier, agent_name, attempt, reasoning_mode, action_mode
):
    """
    检查单个attempt的状态

    Returns:
    - "not_started": attempt还没有开始（没有log.json）
    - "executed_not_evaluated": 已经执行但未评估（有log.json但没有评估结果）
    - "executed_and_evaluated": 已经执行且已评估（有log.json且有评估结果）
    - "error": 检查过程中出现错误
    """
    try:
        # 检查是否有log.json文件
        attempt_dir = os.path.join(
            output_dir, task_identifier, agent_name, f"attempt_{attempt}"
        )
        log_json_path = os.path.join(attempt_dir, "log.json")

        if not os.path.exists(log_json_path):
            return "not_started"

        # 检查是否已经评估
        df = pd.read_csv(get_results_csv_path(output_dir))
        task_row = df[df["task_identifier"] == task_identifier]

        if task_row.empty:
            return "executed_not_evaluated"

        eval_col_prefix = get_col_name_from_template(
            "",
            agent_name=agent_name,
            eval_name=reasoning_mode,
            sub_eval_name=action_mode,
            attempt_num=attempt,
        )
        eval_col_name = f"{eval_col_prefix}_evaluation"

        if eval_col_name not in df.columns:
            return "executed_not_evaluated"

        eval_result = task_row.iloc[0][eval_col_name]

        if eval_result in ["S", "F", "E"]:  # 已有评估结果
            return "executed_and_evaluated"
        else:
            return "executed_not_evaluated"

    except Exception as e:
        print(
            f"Error checking attempt {attempt} status for task {task_identifier}: {e}"
        )
        return "error"


def is_task_fully_completed(
    output_dir, task_identifier, agent_name, max_attempts, reasoning_mode, action_mode
):
    """
    检查任务是否已经完全完成（所有k次attempts都已经执行和评估完成）

    注意：根据rollout策略，无论前面的attempts是否成功，都必须执行满k次attempts

    Returns:
    - True: 所有k次attempts都已完全完成，不需要继续执行
    - False: 还有未完成的attempts，需要继续执行
    """
    try:
        df = pd.read_csv(get_results_csv_path(output_dir))
        task_row = df[df["task_identifier"] == task_identifier]

        if task_row.empty:
            return False

        # 检查所有attempts是否都已经评估完成
        # 注意：移除了早期成功停止的逻辑，确保所有k次attempts都被执行
        for attempt in range(1, max_attempts + 1):
            status = get_attempt_status(
                output_dir,
                task_identifier,
                agent_name,
                attempt,
                reasoning_mode,
                action_mode,
            )
            if status in ["not_started", "executed_not_evaluated"]:
                return False

        return True  # 所有attempts都已评估完成

    except Exception as e:
        print(f"Error checking task completion for {task_identifier}: {e}")
        return False


def collect_evaluation_tasks(
    output_dir,
    agent_scope,
    task_scope,
    max_attempts,
    reasoning_mode,
    action_mode,
    result_overwrite=False,
):
    """
    收集所有需要评估的任务
    """
    eval_tasks = []
    skipped_count = 0

    # 读取当前的结果文件
    current_results_df = get_results_df(output_dir)

    for agent_name in agent_scope:
        for task in task_scope:
            valid_attempts = get_valid_attempts(
                output_dir, task.task_identifier, agent_name, max_attempts
            )

            for attempt in valid_attempts:
                attempt_dir = os.path.join(
                    output_dir, task.task_identifier, agent_name, f"attempt_{attempt}"
                )

                # 检查是否已有评估结果
                if not result_overwrite:
                    result_col_prefix = get_col_name_from_template(
                        "",
                        agent_name=agent_name,
                        eval_name=reasoning_mode,
                        sub_eval_name=action_mode,
                        attempt_num=attempt,
                    )
                    result_col_name = f"{result_col_prefix}_evaluation"

                    # 检查该任务是否已经有评估结果
                    task_row = current_results_df[
                        current_results_df["task_identifier"] == task.task_identifier
                    ]

                    if not task_row.empty and result_col_name in task_row.columns:
                        eval_result_val = task_row.iloc[0][result_col_name]
                        if eval_result_val in ["S", "F", "E"]:  # 已有评估结果
                            print(
                                f"Task {task.task_identifier} agent {agent_name} attempt {attempt} already has evaluation result: {eval_result_val}. Skipping."
                            )
                            skipped_count += 1
                            continue

                # 检查log.json文件是否存在
                log_json_path = os.path.join(attempt_dir, "log.json")
                if not os.path.exists(log_json_path):
                    print(
                        f"No log.json found for task {task.task_identifier} agent {agent_name} attempt {attempt}. Skipping evaluation."
                    )
                    skipped_count += 1
                    continue

                eval_tasks.append(
                    {
                        "task_identifier": task.task_identifier,
                        "agent_name": agent_name,
                        "attempt": attempt,
                        "attempt_dir": attempt_dir,
                        "task": task,
                    }
                )

    print(
        f"收集到 {len(eval_tasks)} 个需要评估的任务，跳过 {skipped_count} 个已有结果或缺少文件的任务"
    )
    return eval_tasks


def process_evaluation_results(
    output_dir, eval_tasks, eval_futures, reasoning_mode, action_mode
):
    """
    处理评估结果并更新成功状态
    """
    print(f"处理 {len(eval_futures)} 个评估任务的结果...")

    for i, future in enumerate(eval_futures):
        eval_task = eval_tasks[i]
        task_identifier = eval_task["task_identifier"]
        agent_name = eval_task["agent_name"]
        attempt = eval_task["attempt"]

        try:
            eval_success = future.result()

            if not eval_success:
                print(
                    f"Evaluation failed for task {task_identifier}, attempt {attempt}"
                )
                continue

            # 读取评估结果
            current_results_df = get_results_df(output_dir)
            result_col_prefix = get_col_name_from_template(
                "",
                agent_name=agent_name,
                eval_name=reasoning_mode,
                sub_eval_name=action_mode,
                attempt_num=attempt,
            )
            result_col_name = f"{result_col_prefix}_evaluation"

            task_row = current_results_df[
                current_results_df["task_identifier"] == task_identifier
            ]

            if task_row.empty or result_col_name not in task_row.columns:
                print(
                    f"Could not find result for task {task_identifier} attempt {attempt} in CSV."
                )
                continue

            eval_result_val = task_row.iloc[0][result_col_name]

            if eval_result_val == "S":
                print(f"Task {task_identifier} Attempt {attempt} was successful!")
                update_success_tracking(
                    output_dir, task_identifier, agent_name, attempt
                )
            elif eval_result_val == "E":
                print(
                    f"Task {task_identifier} Attempt {attempt} resulted in an evaluation error."
                )
            elif eval_result_val == "F":
                print(f"Task {task_identifier} Attempt {attempt} failed evaluation.")
            else:
                print(
                    f"Task {task_identifier} Attempt {attempt} has unknown evaluation result: {eval_result_val}"
                )

        except Exception as e:
            print(
                f"Error processing evaluation result for task {task_identifier}, attempt {attempt}: {e}"
            )


# ==================== App数据重置相关函数 ====================

def get_task_app_package(task) -> list:
    """
    从任务对象获取涉及的app包名列表
    
    支持多种任务格式和列名：
    - app_package 列
    - source_app 列
    - task_app 列
    - adb_app 列
    - 从task_description推断
    
    Args:
        task: 任务对象 (namedtuple)
        
    Returns:
        包名列表
    """
    from .app_data_manager import get_task_app_packages
    return get_task_app_packages(task)


def reset_app_data_for_task(device_serial: str, task, inject_data: bool = True) -> bool:
    """
    Reset app data for the task using AndroidWorld native tools.
    
    Args:
        device_serial: Device serial number
        task: Task object
        inject_data: Whether to inject initial data
        
    Returns:
        Whether successful
    """
    from .app_data_manager import reset_app_for_task, get_task_app_packages
    from .device_initializer import create_env_from_serial
    
    # Debug info: print detailed info about task object
    task_fields = getattr(task, '_fields', None)
    task_id = getattr(task, 'task_identifier', 'unknown')
    print(f"[reset_app_data] Task: {task_id}, task fields: {task_fields}")
    
    # Check if app_package field exists
    if hasattr(task, 'app_package'):
        app_pkg_value = getattr(task, 'app_package')
        print(f"[reset_app_data] app_package field value: {repr(app_pkg_value)}")
    else:
        print(f"[reset_app_data] WARNING: task object does not have app_package field!")
    
    packages = get_task_app_packages(task)
    
    if not packages:
        print(f"[reset_app_data] Unable to identify app package from task, skipping data reset")
        # Print more debug info
        if hasattr(task, 'task_description'):
            desc = getattr(task, 'task_description', '')
            print(f"[reset_app_data] task_description: {desc[:100]}...")
        return True
    
    print(f"[reset_app_data] Identified apps for task: {packages}")
    
    # Create AndroidWorld env for data reset
    env = None
    try:
        print(f"[reset_app_data] Creating AndroidWorld env for device: {device_serial}")
        env = create_env_from_serial(device_serial)
        
        success = True
        for package in packages:
            print(f"Resetting app data: {package}")
            if not reset_app_for_task(env, package, inject_data):
                print(f"Failed to reset app data: {package}")
                success = False
        
        # 等待模拟器 telephony 完成所有 SMS 的投递和持久化，
        # 然后再关闭 env（断开 gRPC 连接）。
        # Reference 使用同一个 env 贯穿整个任务生命周期，不存在此问题；
        # 当前项目为每次 reset 创建/销毁独立 env，过早关闭可能导致
        # 最后几条 SMS 的投递未完成。
        # 从 3 秒增加到 10 秒，给 telephony 框架充足的时间完成异步投递。
        import time
        time.sleep(10)
        
        return success
        
    except Exception as e:
        print(f"[reset_app_data] Failed to create env or reset data: {e}")
        return False
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass


def should_reset_app_data(task, previous_task=None) -> bool:
    """
    Determine whether app data needs to be reset
    
    Strategy:
    - If current task's app differs from previous task's app, reset is needed
    - If this is the first task, reset is needed
    
    Args:
        task: Current task object
        previous_task: Previous task object (optional)
        
    Returns:
        Whether reset is needed
    """
    from .app_data_manager import get_task_app_packages
    
    current_packages = get_task_app_packages(task)
    
    if not current_packages:
        return False
    
    if previous_task is None:
        return True
    
    previous_packages = get_task_app_packages(previous_task)
    
    # If current task's app differs from previous task, reset is needed
    if set(current_packages) != set(previous_packages):
        return True
    
    return True  # Conservative strategy: always reset
