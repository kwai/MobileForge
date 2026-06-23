import os
from android_world.agents import infer
from android_world.agents import (
    t3a,
    m3a,
    seeact,
    ui_tars,
    ui_tars_1_5,
    m3a_multiturn,
    qwen3_vl,
    gui_owl,
)
from android_world.env import env_launcher
from PIL import Image
import json
import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--app", type=str, default="contact")
parser.add_argument(
    "--task",
    type=str,
    default="Open the Contact app, go to the new contact screen and enter the following details: First Name: Grace, Last Name: Taylor, Phone: 799-802-1530, Phone Label: Work. Do NOT hit save.",
)
# parser.add_argument(
#     "--task",
#     type=str,
#     default="Up swipe the screen to find the Camera app, open it, and take a photo.",
# )
parser.add_argument("--lang", type=str, default="ENG")
parser.add_argument("--openai_api_model", type=str, default="gpt-4o-2024-05-13")
parser.add_argument(
    "--openai_api_key",
    type=str,
    default="YOUR_OPENAI_API_KEY",
)
parser.add_argument("--vivo_gemini_api_model", type=str, default="gemini-2.5-pro")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results/debug/attempt_1",
)
parser.add_argument("--max_rounds", type=int, default=20)
parser.add_argument(
    "--agent",
    type=str,
    choices=[
        "T3A",
        "M3A",
        "SeeAct",
        "M3A_vivo_gemini",
        "M3A_MultiTurn",
        "T3A_vivo_gemini",
        "UITARS",
        "UITARS_1_5",
        "Qwen3VL",
        "GUIOwl15",
    ],
    default="M3A_MultiTurn",
)
parser.add_argument("--adb_path", type=str, default="adb")
parser.add_argument("--device_console_port", type=int, default=5554)
parser.add_argument("--device_grpc_port", type=int, default=8554)
parser.add_argument("--device_serial", type=str, default=None)
# UI-TARS specific configuration parameters - no defaults, conditionally required
parser.add_argument("--uitars_base_url", type=str, required=False)
parser.add_argument("--uitars_api_key", type=str, required=False)
parser.add_argument("--uitars_model", type=str, required=False)
parser.add_argument("--uitars_history_n", type=int, required=False)
# M3A specific configuration parameters - no defaults, conditionally required
parser.add_argument(
    "--m3a_model", type=str, required=False, default="app-nu0fg7-1754119470355380516"
)
parser.add_argument(
    "--m3a_base_url",
    type=str,
    required=False,
    default="https://your-openai-compatible-endpoint/v1",
)
parser.add_argument(
    "--m3a_api_key",
    type=str,
    required=False,
    default="YOUR_API_KEY",
)
# Qwen3VL specific configuration parameters - no defaults, conditionally required
parser.add_argument("--qwen_base_url", type=str, required=False)
parser.add_argument("--qwen_api_key", type=str, required=False)
parser.add_argument("--qwen_model", type=str, required=False)
parser.add_argument("--qwen_max_pixels", type=int, required=False)
# Self-hint feature: hints from previous failed attempts
parser.add_argument("--hints_file", type=str, required=False, default=None,
                    help="Path to JSON file containing hints from previous failed attempts")
args = parser.parse_args()

# Load hints from file if provided
hints_text = ""
if args.hints_file:
    try:
        with open(args.hints_file, "r", encoding="utf-8") as f:
            hints_data = json.load(f)
            hints_text = hints_data.get("hints", "")
            if hints_text:
                print(f"Loaded hints from {args.hints_file}")
    except Exception as e:
        print(f"Warning: Failed to load hints file {args.hints_file}: {e}")

# os.environ['OPENAI_API_KEY'] = args.openai_api_key
_EMULATOR_SETUP = False

# 确保output_dir存在
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def print_and_log_error(error_message):
    """
    打印错误信息并记录到错误日志文件

    Args:
        error_message: 错误信息
    """
    print(error_message)
    error_log = [{"error_message": error_message}]
    filename = args.output_dir + "/error.json"
    # Check if the file already exists
    if not os.path.exists(filename):
        # If the file does not exist, create it and write the JSON data
        with open(filename, "w", encoding="utf-8") as logfile:
            json.dump(error_log, logfile, ensure_ascii=False)


def setup_agent(env):
    """
    根据参数配置适当的代理

    Args:
        env: 环境对象

    Returns:
        tuple: (agent, screenshot_key, grounded_action_key, log_keys, raw_response_key)
    """
    # Validate required parameters for specific agents
    if args.agent in ["UITARS", "UITARS_1_5"]:
        if not args.uitars_base_url:
            raise ValueError(f"--uitars_base_url is required for {args.agent} agent")
        if not args.uitars_api_key:
            raise ValueError(f"--uitars_api_key is required for {args.agent} agent")
        if not args.uitars_model:
            raise ValueError(f"--uitars_model is required for {args.agent} agent")
        if args.uitars_history_n is None:
            raise ValueError(f"--uitars_history_n is required for {args.agent} agent")

    if args.agent in ["M3A", "M3A_vivo_gemini", "M3A_MultiTurn"]:
        if not args.m3a_model:
            raise ValueError(f"--m3a_model is required for {args.agent} agent")
        if not args.m3a_base_url:
            raise ValueError(f"--m3a_base_url is required for {args.agent} agent")
        if not args.m3a_api_key:
            raise ValueError(f"--m3a_api_key is required for {args.agent} agent")

    if args.agent in ["Qwen3VL", "GUIOwl15"]:
        if not args.qwen_base_url:
            raise ValueError(f"--qwen_base_url is required for {args.agent} agent")
        if not args.qwen_api_key:
            raise ValueError(f"--qwen_api_key is required for {args.agent} agent")
        if not args.qwen_model:
            raise ValueError(f"--qwen_model is required for {args.agent} agent")

    if args.agent == "M3A":
        # Create M3A config dict from command line arguments
        m3a_config = {
            "M3A_MODEL": args.m3a_model,
            "M3A_BASE_URL": args.m3a_base_url,
            "M3A_API_KEY": args.m3a_api_key,
        }
        agent = m3a.M3A(env, infer.VivoGeminiWrapper(args.m3a_model, config=m3a_config))
        screenshot_key = "raw_screenshot"
        grounded_action_key = "action_output_json"
        log_keys = ["action_output", "summary"]
        raw_response_key = ["action_raw_response", "summary_raw_response"]
    elif args.agent == "M3A_vivo_gemini":
        # Create M3A config dict from command line arguments
        m3a_config = {
            "M3A_MODEL": args.m3a_model,
            "M3A_BASE_URL": args.m3a_base_url,
            "M3A_API_KEY": args.m3a_api_key,
        }
        agent = m3a.M3A(env, infer.VivoGeminiWrapper(args.m3a_model, config=m3a_config))
        screenshot_key = "raw_screenshot"
        grounded_action_key = "action_output_json"
        log_keys = ["action_output", "summary"]
        raw_response_key = ["action_raw_response", "summary_raw_response"]
    elif args.agent == "T3A":
        agent = t3a.T3A(env, infer.Gpt4Wrapper(args.openai_api_model))
        screenshot_key = "before_screenshot"
        grounded_action_key = "converted_action"
        log_keys = ["action_output", "summary"]
        raw_response_key = ["action_raw_response", "summary_raw_response"]
    elif args.agent == "T3A_vivo_gemini":
        agent = t3a.T3A(env, infer.VivoGeminiWrapper(args.vivo_gemini_api_model))
        screenshot_key = "before_screenshot"
        grounded_action_key = "converted_action"
        log_keys = ["action_output", "summary"]
        raw_response_key = ["action_raw_response", "summary_raw_response"]
    elif args.agent == "M3A_MultiTurn":
        # Create M3A config dict from command line arguments
        m3a_config = {
            "M3A_MODEL": args.m3a_model,
            "M3A_BASE_URL": args.m3a_base_url,
            "M3A_API_KEY": args.m3a_api_key,
        }
        agent = m3a_multiturn.M3A_MultiTurn(
            env, infer.VivoGeminiWrapper(args.m3a_model, config=m3a_config)
        )
        screenshot_key = "before_screenshot"
        grounded_action_key = "action_output_json"
        log_keys = [
            "instruction",
            "action_prompt",
            "action_output",
            "summary_prompt",
            "summary",
        ]
        raw_response_key = ["action_raw_response", "summary_raw_response"]
    elif args.agent == "SeeAct":
        agent = seeact.SeeAct(env, model=args.openai_api_model)
        screenshot_key = "screenshot"
        grounded_action_key = "action"
        log_keys = ["action_gen_response", "action_ground_response"]
        raw_response_key = ["raw_action_gen_response", "raw_action_ground_response"]
    elif args.agent == "UITARS":
        # Create config dict from command line arguments
        uitars_config = {
            "UITARS_BASE_URL": args.uitars_base_url,
            "UITARS_API_KEY": args.uitars_api_key,
            "UITARS_MODEL": args.uitars_model,
            "UITARS_HISTORY_N": args.uitars_history_n,
        }
        agent = ui_tars.UITARS(env, config=uitars_config)
        screenshot_key = "raw_screenshot"
        grounded_action_key = "action_output_json"
        log_keys = ["action_output", "raw_response", "raw_action"]
        raw_response_key = ["action_raw_response"]
    elif args.agent == "UITARS_1_5":
        # Create config dict from command line arguments
        uitars_config = {
            "UITARS_BASE_URL": args.uitars_base_url,
            "UITARS_API_KEY": args.uitars_api_key,
            "UITARS_MODEL": args.uitars_model,
            "UITARS_HISTORY_N": args.uitars_history_n,
        }
        agent = ui_tars_1_5.UITARS_1_5(env, config=uitars_config)
        screenshot_key = "raw_screenshot"
        grounded_action_key = "action_output_json"
        log_keys = ["action_output", "raw_response"]
        raw_response_key = ["action_raw_response"]
    elif args.agent == "Qwen3VL":
        # Create config dict from command line arguments
        qwen_config = {
            "QWEN_BASE_URL": args.qwen_base_url,
            "QWEN_API_KEY": args.qwen_api_key,
            "QWEN_MODEL": args.qwen_model,
        }
        if args.qwen_max_pixels:
            qwen_config["QWEN_MAX_PIXELS"] = args.qwen_max_pixels
        agent = qwen3_vl.Qwen3VL(env, config=qwen_config)
        screenshot_key = "before_screenshot"
        grounded_action_key = "parsed_action"
        log_keys = ["action_output", "raw_response"]
        raw_response_key = ["raw_response"]
    elif args.agent == "GUIOwl15":
        # GUIOwl15 uses the same QWEN_* config keys as Qwen3VL
        guiowl_config = {
            "QWEN_BASE_URL": args.qwen_base_url,
            "QWEN_API_KEY": args.qwen_api_key,
            "QWEN_MODEL": args.qwen_model,
        }
        agent = gui_owl.GUIOwl15(env, config=guiowl_config)
        screenshot_key = "screenshot"
        grounded_action_key = "action"
        log_keys = ["action_response"]
        raw_response_key = ["action_response"]
    return agent, screenshot_key, grounded_action_key, log_keys, raw_response_key


def process_action(response, grounded_action_key):
    """
    处理代理执行的动作

    Args:
        response: 代理响应
        grounded_action_key: 操作的键名

    Returns:
        tuple: (action_log, action_type)
    """
    action_log = [
        "",  # action type
        {
            "detail_type": "string",  # "string" or "coordinates"
            "detail": "",  # "Task completed." or [x, y] or f"The text \"{input_str}\" has been inputted."
            # or f"The coordinates ({x},{y}) have been swiped to the {swipe_direction}."
            # or f"The swipe action has been performed starting from coordinates ({start_x},{start_y}) to ({end_x},{end_y})."
        },
    ]  # second element for action details based action_type

    action_type = ""
    converted_action = response.data[grounded_action_key]

    # 添加对converted_action为None的处理
    if converted_action is None:
        action_type = "error"
        action_log[0] = action_type
        action_log[1]["detail"] = "Action parsing failed - converted_action is None"
        return action_log, action_type

    if type(converted_action) is str:
        action_type = "wait"
    else:
        # Support both dict and object formats
        if isinstance(converted_action, dict):
            # Dictionary format (e.g., from Qwen3VL)
            action_type = converted_action.get("action_type", "")
            if not action_type:
                action_type = "error"
                action_log[0] = action_type
                action_log[1]["detail"] = (
                    "Action parsing failed - no action_type in dict"
                )
                return action_log, action_type
        elif hasattr(converted_action, "action_type"):
            # Object format (e.g., from other agents)
            action_type = converted_action.action_type
        else:
            action_type = "error"
            action_log[0] = action_type
            action_log[1]["detail"] = "Action parsing failed - no action_type attribute"
            return action_log, action_type

        if action_type == "click":
            action_log[1]["detail_type"] = "coordinates"
            action_log[1]["detail"] = response.data["actual_action_coordinates"]
        elif action_type == "double_tap":
            action_log[1]["detail_type"] = "coordinates"
            action_log[1]["detail"] = response.data["actual_action_coordinates"]
        elif action_type == "input_text":
            # Handle both dict and object format
            if isinstance(converted_action, dict):
                text = converted_action.get("text", "")
            else:
                text = converted_action.text
            action_log[1]["detail"] = f'The text "{text}" has been inputted.'
        elif action_type == "keyboard_enter":
            pass
        elif action_type == "long_press":
            action_log[1]["detail_type"] = "coordinates"
            action_log[1]["detail"] = response.data["actual_action_coordinates"]
        elif action_type == "navigate_back":
            action_log[1]["detail"] = "Back to the previous page."
        elif action_type == "navigate_home":
            action_log[1]["detail"] = "Return to home page."
        elif action_type == "open_app":
            # Handle both dict and object format
            if isinstance(converted_action, dict):
                app_name = converted_action.get("app_name", "")
            else:
                app_name = converted_action.app_name
            action_log[1]["detail"] = app_name
        elif action_type == "scroll":
            x1, y1, x2, y2 = response.data["actual_action_coordinates"]
            # Handle both dict and object format
            if isinstance(converted_action, dict):
                direction = converted_action.get("direction", "")
            else:
                direction = converted_action.direction
            action_log[1]["detail"] = (
                f"The scroll action has been performed starting from coordinates ({x1},{y1}) to ({x2},{y2}). Direction: {direction}."
            )
        elif action_type == "status":
            # Handle both dict and object format
            if isinstance(converted_action, dict):
                goal_status = converted_action.get("goal_status", "")
            else:
                goal_status = converted_action.goal_status
            if goal_status in ["complete", "infeasible"]:
                action_log[1]["detail"] = f"Task {goal_status}."
            else:
                action_log[1]["detail"] = (
                    f"The agent thinks the task has finished and answered with '{goal_status}'."
                )
        elif action_type == "swipe":
            x1, y1, x2, y2 = response.data["actual_action_coordinates"]
            action_log[1]["detail"] = (
                f"The swipe action has been performed starting from coordinates ({x1},{y1}) to ({x2},{y2})."
            )
        elif action_type == "drag":
            x1, y1, x2, y2 = response.data["actual_action_coordinates"]
            action_log[1]["detail"] = (
                f"The drag action has been performed starting from coordinates ({x1},{y1}) to ({x2},{y2})."
            )
        elif action_type == "answer":
            # Handle both dict and object format
            if isinstance(converted_action, dict):
                text = converted_action.get("text", "")
            else:
                text = converted_action.text
            action_log[1]["detail"] = f'The answer is "{text}".'
        else:  # [ 'unknown', 'wait']
            action_type = "wait"
            action_log[1]["detail"] = "No action has been taken."
    action_log[0] = action_type
    return action_log, action_type


def main():
    """主函数，执行基准测试"""
    log_path = os.path.join(args.output_dir, "log.json")

    def _persist_log(log_list):
        """每步增量持久化 log.json，防止子进程被 SIGKILL 时丢失。

        使用 default=str 兜底所有不可序列化对象，并写到临时文件后原子替换，
        避免 kill 发生在 write 中途留下损坏的 json。
        """
        try:
            tmp_path = log_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(log_list, f, ensure_ascii=False, default=str)
            os.replace(tmp_path, log_path)
        except Exception as e:
            print(f"[benchmark_run] Warning: failed to persist log.json: {e}")

    start_time_initial = time.time()
    env = env_launcher.load_and_setup_env(
        console_port=args.device_console_port,
        emulator_setup=_EMULATOR_SETUP,
        freeze_datetime=False,
        adb_path=args.adb_path,
        grpc_port=args.device_grpc_port,
    )
    # Benchmark: Remove api level check
    # env_launcher.verify_api_level(env)
    try:
        env.reset(go_home=False)

        # 设置代理
        agent, screenshot_key, grounded_action_key, log_keys, raw_response_key = (
            setup_agent(env)
        )
        
        # Prepare goal with hints if available (for self-hint feature)
        goal_with_hints = args.task
        if hints_text:
            # Append hints to the task description for agent to consider
            goal_with_hints = f"{args.task}\n{hints_text}"
            print("=" * 60)
            print("Self-hint: Previous attempt hints included in goal")
            print("=" * 60)
        
        print("Goal: " + args.task)
        if hints_text:
            print("Hints from previous attempts: " + hints_text[:200] + "..." if len(hints_text) > 200 else hints_text)

        is_done = False
        benchmark_log = []
        error_code = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        end_time_initial = time.time()
        elapsed_time_initial = end_time_initial - start_time_initial
        start_time_exec = time.time()
        for action_cnt in range(1, args.max_rounds + 1):
            try:
                response = agent.step(goal_with_hints)
            except Exception as e:
                print_and_log_error(f"Error taking step: {str(e)}")
                error_code = 2
                break
            try:
                # Store screenshot
                screenshot = Image.fromarray(response.data[screenshot_key], "RGB")
                screenshot.save(os.path.join(args.output_dir, f"{action_cnt - 1}.png"))

                # 处理动作
                action_log, action_type = process_action(response, grounded_action_key)

                # 计算tokens和获取增强日志
                if args.agent in [
                    "M3A",
                    "T3A",
                    "M3A_vivo_gemini",
                    "T3A_vivo_gemini",
                    "M3A_MultiTurn",
                ]:
                    prompt_tokens = sum(
                        [
                            response.data[key].json()["usage"]["prompt_tokens"]
                            for key in raw_response_key
                            if response.data[key] is not None
                        ]
                    )
                    completion_tokens = sum(
                        [
                            response.data[key].json()["usage"]["completion_tokens"]
                            for key in raw_response_key
                            if response.data[key] is not None
                        ]
                    )
                    enhanced_log_data = {}
                elif args.agent in ["UITARS", "UITARS_1_5", "Qwen3VL", "GUIOwl15"]:
                    # 获取增强日志数据，包括详细的模型调用记录
                    if hasattr(agent, "get_enhanced_log_data"):
                        enhanced_log_data = agent.get_enhanced_log_data()
                        prompt_tokens = enhanced_log_data.get("total_prompt_tokens", 0)
                        completion_tokens = enhanced_log_data.get(
                            "total_completion_tokens", 0
                        )
                    else:
                        # 兼容旧版本
                        enhanced_log_data = {}
                        prompt_tokens = 0
                        completion_tokens = 0
                else:
                    prompt_tokens = sum(
                        [
                            response.data[key]["usage"]["prompt_tokens"]
                            for key in raw_response_key
                            if response.data[key] is not None
                        ]
                    )
                    completion_tokens = sum(
                        [
                            response.data[key]["usage"]["completion_tokens"]
                            for key in raw_response_key
                            if response.data[key] is not None
                        ]
                    )
                    enhanced_log_data = {}

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                # 记录本次操作
                benchmark_log.append(
                    {
                        "step": action_cnt,
                        **{key: response.data[key] for key in log_keys},
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "action": action_log,
                    }
                )
                # 每步增量写盘，防止子进程被 kill（SIGKILL 无法 finally）后丢失
                _persist_log(benchmark_log)
                if response.done:
                    is_done = True
                    break

            except Exception as e:
                print_and_log_error(f"Error handling log: {str(e)}")
                error_code = 1
                break
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[benchmark_run] env.close() failed: {e}")

        # finally 块写入最终 summary。注意：SIGKILL 不会进入 finally，
        # 但 SIGTERM / 普通异常都会经过这里，配合上面每步增量写，
        # 最差情况也能保证盘上有一份带 step 的 log.json。
        try:
            end_time_exec = time.time()
            elapsed_time_exec = end_time_exec - start_time_exec
            # 如果循环根本没进，action_cnt 不存在
            try:
                steps_done = action_cnt - 1 if "action_cnt" in dir() else 0  # noqa: F821
            except Exception:
                steps_done = 0

            # 移除可能残留的旧 summary（避免重复尾部）
            if benchmark_log and isinstance(benchmark_log[-1], dict) and "total_steps" in benchmark_log[-1]:
                benchmark_log.pop()

            benchmark_log.append(
                {
                    "total_steps": steps_done,
                    "finish_signal": int(is_done),
                    "elapsed_time_initial": elapsed_time_initial,
                    "elapsed_time_exec": elapsed_time_exec,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "error_code": error_code,
                }
            )
            _persist_log(benchmark_log)
        except Exception as e:
            print(f"[benchmark_run] Warning: failed to write final summary: {e}")

    # 保存增强的模型调用日志（如果有的话）
    if args.agent in ["UITARS", "UITARS_1_5", "Qwen3VL", "GUIOwl15"] and hasattr(
        agent, "get_enhanced_log_data"
    ):
        enhanced_log_data = agent.get_enhanced_log_data()
        if enhanced_log_data.get("detailed_model_logs"):
            with open(
                args.output_dir + "/detailed_model_logs.json", "w", encoding="utf-8"
            ) as logfile:
                json.dump(
                    enhanced_log_data["detailed_model_logs"],
                    logfile,
                    ensure_ascii=False,
                    indent=2,
                )

            # 保存模型调用统计摘要
            model_stats = {
                "total_model_calls": enhanced_log_data.get("total_model_calls", 0),
                "total_prompt_tokens": enhanced_log_data.get("total_prompt_tokens", 0),
                "total_completion_tokens": enhanced_log_data.get(
                    "total_completion_tokens", 0
                ),
                "average_prompt_tokens_per_call": 0,
                "average_completion_tokens_per_call": 0,
                "total_api_call_duration": 0,
                "average_api_call_duration": 0,
            }

            if model_stats["total_model_calls"] > 0:
                model_stats["average_prompt_tokens_per_call"] = (
                    model_stats["total_prompt_tokens"]
                    / model_stats["total_model_calls"]
                )
                model_stats["average_completion_tokens_per_call"] = (
                    model_stats["total_completion_tokens"]
                    / model_stats["total_model_calls"]
                )

                # 计算API调用时间统计
                api_durations = [
                    log.get("api_call_duration", 0)
                    for log in enhanced_log_data["detailed_model_logs"]
                ]
                model_stats["total_api_call_duration"] = sum(api_durations)
                model_stats["average_api_call_duration"] = (
                    model_stats["total_api_call_duration"]
                    / model_stats["total_model_calls"]
                )

            with open(
                args.output_dir + "/model_call_stats.json", "w", encoding="utf-8"
            ) as logfile:
                json.dump(model_stats, logfile, ensure_ascii=False, indent=2)

    # 确定退出代码
    if error_code in [2, 3]:
        sys.exit(error_code)

    if is_done:
        print("Task completed successfully")
        sys.exit(0)
    elif action_cnt == args.max_rounds:
        print("Task finished due to reaching max rounds")
        sys.exit(4)
    else:
        print("Task finished unexpectedly")
        sys.exit(1)


if __name__ == "__main__":
    main()
