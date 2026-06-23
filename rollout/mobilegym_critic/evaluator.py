import os
import re
import glob
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from framework import utils
from mobilegym_critic.utils.data import get_dataset
from mobilegym_critic.utils.visualize_actions import (
    visualize_and_save_actions,
    create_llm_puzzle,
)
from mobilegym_critic.utils.vivo.vivo_api import (
    inference_chat_gemini_1_image,
    inference_chat_gemini_wo_image,
)
from mobilegym_critic.utils.prompts import (
    get_describe_step_prompt,
    get_final_decision_prompt,
    get_describe_final_step_prompt,
    get_task_feasibility_prompt,
    get_eval_hint_prompt,
)
from mobilegym_critic.utils.common import (
    parse_json_from_response,
    log_and_save_interaction,
)
from mobilegym_critic.utils.image_utils import stitch_images_horizontally

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _generate_description_for_step(
    step_num,
    last_step_num,
    visualize_dir,
    single_actions_dir,
    log_data,
    task_description,
    target_dir,
):
    """
    Worker function to generate a description for a single step.
    This function is designed to be run in a separate thread.
    """
    try:
        is_last_step = step_num == last_step_num

        if is_last_step:
            image_path = os.path.join(single_actions_dir, f"step_{step_num}.png")
        else:
            image_path = os.path.join(visualize_dir, f"step_{step_num}.png")

        if not os.path.exists(image_path):
            print(
                f"Warning: Image for step {step_num} not found at {image_path}. Skipping description."
            )
            return step_num, None

        step_log_data = next(
            (item for item in log_data if item.get("step") == step_num), {}
        )

        action_list = step_log_data.get("action", ["N/A", {}])
        action_type_str = action_list[0]
        action_detail_obj = action_list[1]

        action_detail_str = "N/A"
        if isinstance(action_detail_obj, dict):
            detail_type = action_detail_obj.get("detail_type")
            detail_value = action_detail_obj.get("detail")
            if detail_type == "coordinates":
                action_detail_str = f"Clicked at coordinates: {detail_value}"
            elif detail_value is not None:
                action_detail_str = str(detail_value)
        elif action_detail_obj is not None:
            action_detail_str = str(action_detail_obj)

        if is_last_step:
            (
                system_prompt_desc,
                user_prompt_desc,
            ) = get_describe_final_step_prompt(
                task_description, action_type_str, action_detail_str
            )
        else:
            (
                system_prompt_desc,
                user_prompt_desc,
            ) = get_describe_step_prompt(
                task_description, action_type_str, action_detail_str
            )

        # print(f"  - Calling VLM for step {step_num}...")
        from mobilegym_critic.utils.vivo.llm_config import (
            DEFAULT_DESC_MODEL,
            DEFAULT_DESC_PROVIDER,
            DEFAULT_DESC_API_URL,
        )

        response = inference_chat_gemini_1_image(
            system_prompt_desc,
            user_prompt_desc,
            image_path,
            model=DEFAULT_DESC_MODEL,
            provider=DEFAULT_DESC_PROVIDER,
            api_url=DEFAULT_DESC_API_URL,
        )

        # Handle both old (string) and new (dict) response formats
        if isinstance(response, dict):
            desc_str = response["content"]
            usage_info = response.get("usage", {})
            model_info = {
                "model": response.get("model", DEFAULT_DESC_MODEL),
                "provider": response.get("provider", DEFAULT_DESC_PROVIDER),
                "api_cost": response.get("api_cost", 0.0),
            }
        else:
            desc_str = response
            usage_info = {}
            model_info = {
                "model": DEFAULT_DESC_MODEL,
                "provider": DEFAULT_DESC_PROVIDER,
                "api_cost": 0.0,
            }

        log_and_save_interaction(
            target_dir,
            f"step_{step_num}_description",
            system_prompt_desc,
            user_prompt_desc,
            desc_str,
        )
        desc = parse_json_from_response(desc_str)

        # Add usage information to description
        if desc:
            desc["_usage_info"] = usage_info
            desc["_model_info"] = model_info
            desc["_raw_action"] = {
                "type": action_type_str,
                "detail": action_detail_str,
            }
        # print(f"  - Description for step {step_num} generated.")
        return step_num, desc

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Error generating or parsing description for step {step_num}: {e}")
        return step_num, None


def get_last_screenshot(target_dir):
    """Finds the last screenshot in a directory."""
    screenshots = [
        f for f in os.listdir(target_dir) if f.endswith(".png") and f[:-4].isdigit()
    ]
    if not screenshots:
        return None
    screenshots.sort(key=lambda x: int(x[:-4]))
    return os.path.join(target_dir, screenshots[-1])


def generate_eval_hint(
    task_identifier,
    task_description,
    failure_reason,
    step_descriptions,
    step_analysis,
    target_dir,
    enable_hints=True,
):
    """
    Generates self-reflection hints from attempts with issues to guide future attempts.

    Args:
        task_identifier: The task ID
        task_description: Description of the task
        failure_reason: Reason for failure or issues found
        step_descriptions: Dictionary of step descriptions
        step_analysis: Dictionary of step reasonableness analysis from evaluator
        target_dir: Directory where attempt results are stored
        enable_hints: Whether to generate hints (default True)

    Returns:
        Dictionary containing hint data, or None if hints are disabled or error occurred
    """
    if not enable_hints:
        logging.info(
            f"[{task_identifier}] Self-hint disabled, skipping hint generation"
        )
        return None

    logging.info(
        f"[{task_identifier}] Generating self-reflection hint with LLM..."
    )


    try:
        # Import model configuration (same as final decision)
        from mobilegym_critic.utils.vivo.llm_config import (
            DEFAULT_MODEL,
            DEFAULT_PROVIDER,
            DEFAULT_API_URL,
        )

        # Check for required image files (located in puzzle/ subdirectory)
        puzzle_dir = os.path.join(target_dir, "puzzle")
        pre_eval_puzzle_path = os.path.join(puzzle_dir, "pre_eval_puzzle.png")
        supplemental_puzzle_path = os.path.join(puzzle_dir, "supplemental_puzzle.png")

        if not os.path.exists(pre_eval_puzzle_path):
            logging.warning(
                f"[{task_identifier}] pre_eval_puzzle.png not found, cannot generate hint"
            )
            return None

        # Determine which image to use
        # If supplemental_puzzle exists, use it (more detailed); otherwise use pre_eval_puzzle
        if os.path.exists(supplemental_puzzle_path):
            image_path = supplemental_puzzle_path
            logging.info(
                f"[{task_identifier}] Using supplemental_puzzle.png for hint generation"
            )
        else:
            image_path = pre_eval_puzzle_path
            logging.info(
                f"[{task_identifier}] Using pre_eval_puzzle.png for hint generation"
            )

        # Get prompts for hint generation
        system_prompt, user_prompt = get_eval_hint_prompt(
            task_description=task_description,
            failure_reason=failure_reason,
            step_descriptions=step_descriptions,
            step_analysis=step_analysis,
        )

        # Call LLM to generate hint with image (same parameters as final decision)
        response = inference_chat_gemini_1_image(
            system_prompt,
            user_prompt,
            image_path,
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
            api_url=DEFAULT_API_URL,
        )

        if not response:
            logging.warning(
                f"[{task_identifier}] No response from LLM for hint generation"
            )
            return None

        # Handle response format and extract usage info
        if isinstance(response, dict):
            response_str = response["content"]
            token_usage = response.get("usage", {})
            model_info = {
                "model": response.get("model", DEFAULT_MODEL),
                "provider": response.get("provider", DEFAULT_PROVIDER),
                "api_cost": response.get("api_cost", 0.0),
            }
        else:
            response_str = response
            token_usage = {}
            model_info = {
                "model": DEFAULT_MODEL,
                "provider": DEFAULT_PROVIDER,
                "api_cost": 0.0,
            }

        # Parse JSON response
        hint_data = parse_json_from_response(response_str)

        if hint_data:
            # Add metadata
            # Extract attempt number from target_dir (e.g., ".../attempt_1/")
            attempt_dir_name = os.path.basename(target_dir.rstrip('/'))
            if attempt_dir_name.startswith("attempt_"):
                try:
                    attempt_num_extracted = int(attempt_dir_name.split("_")[1])
                    hint_data["generated_at_attempt"] = attempt_num_extracted
                except (IndexError, ValueError):
                    pass
            hint_data["task_identifier"] = task_identifier
            hint_data["token_usage"] = token_usage
            hint_data["model_info"] = model_info
            hint_data["image_used"] = os.path.basename(image_path)

            # Save hint to a dedicated file
            hint_path = os.path.join(target_dir, "eval_hint.json")
            with open(hint_path, "w", encoding="utf-8") as f:
                json.dump(hint_data, f, indent=4, ensure_ascii=False)

            # Save the complete interaction (prompt + response) for debugging/analysis
            log_and_save_interaction(
                target_dir,
                "eval_hint_generation",
                system_prompt,
                user_prompt,
                response_str,
            )

            logging.info(
                f"[{task_identifier}] Self-reflection hint generated successfully. "
                f"Key mistake: {hint_data.get('key_mistake', 'N/A')[:80]}..."
            )

            return hint_data
        else:
            logging.warning(
                f"[{task_identifier}] Failed to parse hint JSON from LLM response"
            )
            # Still save the interaction even if parsing failed for debugging
            log_and_save_interaction(
                target_dir,
                "eval_hint_generation_PARSE_FAILED",
                system_prompt,
                user_prompt,
                response_str,
            )
            return None

    except Exception as e:
        logging.error(
            f"[{task_identifier}] Error generating eval hint: {e}"
        )
        import traceback

        traceback.print_exc()
        return None


def mobilegym_critic_evaluator(
    task_identifier,
    result_dir,
    mode="eval",
    agent=None,
    attempt_num=None,
    reasoning_mode="direct",
    action_mode="with_action",
    self_hint_enabled=False,
):
    """
    Main function for evaluating a task.
    """
    dataset = get_dataset(utils.get_results_csv_path(result_dir))
    task_description = dataset.loc[task_identifier]["task_description"]

    # The path should always include the agent if it's provided, regardless of mode.
    if agent:
        base_dir = os.path.join(result_dir, task_identifier, agent)
    else:
        base_dir = os.path.join(result_dir, task_identifier)

    # If an attempt number is given, use the attempt-specific subdirectory
    if attempt_num:
        target_dir = os.path.join(base_dir, f"attempt_{attempt_num}")
    else:
        target_dir = base_dir

    if not os.path.isdir(target_dir):
        logging.error(f"Target directory not found: {target_dir}")
        # Save error at the task level if attempt dir doesn't exist
        utils.save_result__completed_evaluation(
            result_dir,
            task_identifier,
            agent,
            -1,  # Error code
            {"error": f"Target directory not found: {target_dir}"},
            reasoning_mode,
            action_mode,
            attempt_num,
            "directory_not_found",
            failure_step=None,
        )
        return -1

    evaluation_detail = {}

    # Step 1: Visualize actions
    # print(f"[{task_identifier}] Step 1: Visualizing actions...")
    try:
        log_data = visualize_and_save_actions(
            target_dir, task_identifier, task_description
        )
        evaluation_detail["visualization"] = "Success"
    except Exception as e:
        print(f"Error during visualization for {task_identifier}: {e}")
        evaluation_detail["visualization"] = f"Failed: {e}"
        utils.save_result__completed_evaluation(
            result_dir,
            task_identifier,
            agent,
            -1,
            evaluation_detail,
            reasoning_mode,
            action_mode,
            attempt_num,
            "visualization_error",
            failure_step=None,
        )
        return -1

    # ==================================================================================
    # Proceed directly to detailed evaluation for all tasks
    # ==================================================================================
    logging.info(f"[{task_identifier}] Proceeding directly to detailed evaluation.")

    # ==================================================================================
    # 2. Detailed Step-by-Step Description Generation
    # ==================================================================================
    # logging.info(
    #     f"[{task_identifier}] Step 2: Generating step descriptions with VLM (in parallel)..."
    # )
    step_descriptions = {}  # Use a dictionary keyed by step number

    # Find all steps from the log data
    if log_data is None:
        logging.error(
            f"[{task_identifier}] log_data is None, cannot proceed with evaluation."
        )
        utils.save_result__completed_evaluation(
            result_dir,
            task_identifier,
            agent,
            -1,
            {"error": "log_data is None"},
            reasoning_mode,
            action_mode,
            attempt_num,
            "log_data_error",
            failure_step=None,
        )
        return -1

    all_steps_in_log = sorted(
        [item["step"] for item in log_data if "step" in item and item.get("action")]
    )
    if not all_steps_in_log:
        logging.warning(
            f"[{task_identifier}] No valid steps with actions found in log.json."
        )
        last_step_num = 0
    else:
        last_step_num = all_steps_in_log[-1]
        visualize_dir = os.path.join(target_dir, "visualize_actions")
        single_actions_dir = os.path.join(target_dir, "single_actions")

        # Use ThreadPoolExecutor to run description generation in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_step = {
                executor.submit(
                    _generate_description_for_step,
                    step_num,
                    last_step_num,
                    visualize_dir,
                    single_actions_dir,
                    log_data,
                    task_description,
                    target_dir,
                ): step_num
                for step_num in all_steps_in_log
            }

            for future in as_completed(future_to_step):
                step_num = future_to_step[future]
                try:
                    step_num_result, desc = future.result()
                    if desc:
                        step_descriptions[step_num_result] = desc
                except Exception as e:
                    logging.error(
                        f"An exception occurred for step {step_num} in the future: {e}"
                    )

    evaluation_detail["step_descriptions"] = step_descriptions

    # ==================================================================================
    # 3. Final Decision (Multi-phase) - Now includes task feasibility assessment
    # ==================================================================================
    # logging.info(f"[{task_identifier}] Step 3: Making final decision with LLM...")
    result = 0
    try:
        # Convert dict to a sorted list of descriptions for the prompt
        prompt_descriptions = []
        for step_num in sorted(
            [k for k in step_descriptions.keys() if isinstance(k, int)]
        ):
            prompt_descriptions.append(step_descriptions[step_num])
        if "final" in step_descriptions:
            prompt_descriptions.append(step_descriptions["final"])

        # --- Final Decision with Last 3 Screenshots ---
        from mobilegym_critic.utils.vivo.llm_config import (
            DEFAULT_MODEL,
            DEFAULT_PROVIDER,
            DEFAULT_API_URL,
        )

        # Create pre_eval_puzzle.png with last 3 screenshots
        single_actions_dir = os.path.join(target_dir, "single_actions")
        puzzle_dir = os.path.join(target_dir, "puzzle")
        os.makedirs(puzzle_dir, exist_ok=True)

        # Get last 3 screenshots for final decision
        pre_eval_puzzle_path = os.path.join(puzzle_dir, "pre_eval_puzzle.png")
        if os.path.exists(single_actions_dir):
            screenshot_files = sorted(
                glob.glob(os.path.join(single_actions_dir, "*.png"))
            )

            # Sort files numerically based on step number
            def get_step_num(filepath):
                match = re.search(r"step_(\d+)\.png", os.path.basename(filepath))
                return int(match.group(1)) if match else -1

            screenshot_files.sort(key=get_step_num)

            # Get last 3 screenshots
            last_screenshots_paths = screenshot_files[-3:]
            last_screenshots_labels = [
                f"Step {get_step_num(p)}" for p in last_screenshots_paths
            ]

            if last_screenshots_paths:
                try:
                    stitched_image = stitch_images_horizontally(
                        last_screenshots_paths, labels=last_screenshots_labels
                    )
                    stitched_image.save(pre_eval_puzzle_path)
                    logging.info(
                        f"[{task_identifier}] Created pre_eval_puzzle.png with last {len(last_screenshots_paths)} screenshots"
                    )
                except Exception as e:
                    logging.error(
                        f"[{task_identifier}] Error creating pre_eval_puzzle.png: {e}"
                    )
                    # Fallback to original puzzle.png if pre_eval_puzzle creation fails
                    pre_eval_puzzle_path = os.path.join(puzzle_dir, "puzzle.png")
            else:
                logging.warning(
                    f"[{task_identifier}] No screenshots found in single_actions, falling back to puzzle.png"
                )
                pre_eval_puzzle_path = os.path.join(puzzle_dir, "puzzle.png")
        else:
            logging.warning(
                f"[{task_identifier}] single_actions directory not found, falling back to puzzle.png"
            )
            pre_eval_puzzle_path = os.path.join(puzzle_dir, "puzzle.png")

        (
            system_prompt_decision,
            user_prompt_decision,
        ) = get_final_decision_prompt(task_description, prompt_descriptions, "")
        response = inference_chat_gemini_1_image(
            system_prompt_decision,
            user_prompt_decision,
            pre_eval_puzzle_path,
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
            api_url=DEFAULT_API_URL,
        )

        # Handle response format and extract usage info
        if isinstance(response, dict):
            response_str = response["content"]
            final_decision_usage = response.get("usage", {})
            final_decision_model_info = {
                "model": response.get("model"),
                "provider": response.get("provider"),
                "api_cost": response.get("api_cost", 0.0),
            }
        else:
            response_str = response
            final_decision_usage = {}
            final_decision_model_info = {
                "model": "unknown",
                "provider": "unknown",
                "api_cost": 0.0,
            }
        log_and_save_interaction(
            target_dir,
            "final_decision",
            system_prompt_decision,
            user_prompt_decision,
            response_str,
        )
        decision_data = parse_json_from_response(response_str)

        result = int(decision_data.get("decision", -1))
        reason = decision_data.get("reason", "No reason provided.")

        # Extract failure step information if task failed
        failure_step = None
        if result == 0:  # Task failed
            failure_step = decision_data.get("failure_step")
            if failure_step is not None:
                logging.info(f"[{task_identifier}] Task failed at step: {failure_step}")

        # Extract step reasonableness analysis
        reasonable_steps = decision_data.get("reasonable_steps", [])
        unreasonable_steps = decision_data.get("unreasonable_steps", [])
        step_analysis = decision_data.get("step_analysis", {})

        # Extract task feasibility assessment from decision_data
        task_feasible = decision_data.get("task_feasible", True)
        task_feasible_reason = decision_data.get("task_feasible_reason", "")
        task_barriers = decision_data.get("task_barriers", [])

        logging.info(
            f"[{task_identifier}] Evaluation result: {result}, Reason: {reason}"
        )
        logging.info(
            f"[{task_identifier}] Step analysis - Reasonable: {len(reasonable_steps)}, Unreasonable: {len(unreasonable_steps)}"
        )
        logging.info(
            f"[{task_identifier}] Task feasibility: {task_feasible}, Reason: {task_feasible_reason}"
        )

        evaluation_detail["final_decision_response"] = decision_data
        evaluation_detail["final_result"] = result
        evaluation_detail["failure_step"] = failure_step
        evaluation_detail["reasonable_steps"] = reasonable_steps
        evaluation_detail["unreasonable_steps"] = unreasonable_steps
        evaluation_detail["step_analysis"] = step_analysis
        evaluation_detail["task_feasible"] = task_feasible
        evaluation_detail["task_feasible_reason"] = task_feasible_reason
        evaluation_detail["task_barriers"] = task_barriers

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logging.error(f"Error during final decision for {task_identifier}: {e}")
        result = -1
        failure_step = None
        evaluation_detail["final_decision_response"] = {
            "reason": f"Failed: {e}",
            "decision": -1,
        }
        evaluation_detail["final_result"] = -1
        evaluation_detail["failure_step"] = None

    # Calculate separated usage and cost tracking
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    model_used = None
    provider_used = None

    # Step description usage tracking
    step_desc_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    step_desc_cost = 0.0
    step_desc_model_name = None
    step_desc_model_provider = None

    # Final decision usage tracking
    final_decision_usage_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    final_decision_cost_total = 0.0
    final_decision_model_name = None
    final_decision_model_provider = None

    # Accumulate from step descriptions
    for desc in step_descriptions.values():
        if isinstance(desc, dict) and "_usage_info" in desc:
            usage_info = desc["_usage_info"]
            model_info = desc.get("_model_info", {})

            # Add to step description totals
            step_desc_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
            step_desc_usage["completion_tokens"] += usage_info.get(
                "completion_tokens", 0
            )
            step_desc_usage["total_tokens"] += usage_info.get("total_tokens", 0)
            step_desc_cost += model_info.get("api_cost", 0.0)

            # Set step description model info (use the first one encountered)
            if step_desc_model_name is None:
                step_desc_model_name = model_info.get("model")
                step_desc_model_provider = model_info.get("provider")

            # Add to overall totals
            total_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage_info.get("total_tokens", 0)
            total_cost += model_info.get("api_cost", 0.0)

            if model_used is None:
                model_used = model_info.get("model")
                provider_used = model_info.get("provider")

    # Add final decision usage
    if "final_decision_usage" in locals():
        # Add to final decision totals
        final_decision_usage_total["prompt_tokens"] = final_decision_usage.get(
            "prompt_tokens", 0
        )
        final_decision_usage_total["completion_tokens"] = final_decision_usage.get(
            "completion_tokens", 0
        )
        final_decision_usage_total["total_tokens"] = final_decision_usage.get(
            "total_tokens", 0
        )
        final_decision_cost_total = final_decision_model_info.get("api_cost", 0.0)

        # Set final decision model info
        final_decision_model_name = final_decision_model_info.get("model")
        final_decision_model_provider = final_decision_model_info.get("provider")

        # Add to overall totals
        total_usage["prompt_tokens"] += final_decision_usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += final_decision_usage.get(
            "completion_tokens", 0
        )
        total_usage["total_tokens"] += final_decision_usage.get("total_tokens", 0)
        total_cost += final_decision_model_info.get("api_cost", 0.0)

        if model_used is None:
            model_used = final_decision_model_info.get("model")
            provider_used = final_decision_model_info.get("provider")

    # Extract reason for saving.
    final_decision_data = {
        "evaluation_method": evaluation_detail.get(
            "evaluation_method", "detailed_evaluation"
        ),
        **evaluation_detail.get("final_decision_response", {}),
    }

    # Add failure step information if available
    if evaluation_detail.get("failure_step") is not None:
        final_decision_data["failure_step"] = evaluation_detail.get("failure_step")

    # Add step reasonableness analysis to final decision data
    if evaluation_detail.get("reasonable_steps") is not None:
        final_decision_data["reasonable_steps"] = evaluation_detail.get(
            "reasonable_steps"
        )
    if evaluation_detail.get("unreasonable_steps") is not None:
        final_decision_data["unreasonable_steps"] = evaluation_detail.get(
            "unreasonable_steps"
        )
    if evaluation_detail.get("step_analysis") is not None:
        final_decision_data["step_analysis"] = evaluation_detail.get("step_analysis")

    if not isinstance(final_decision_data, dict) or "reason" not in final_decision_data:
        reason = str(
            evaluation_detail.get("final_decision_response", "Error in final decision")
        )
    else:
        reason = final_decision_data.get("reason", "No reason provided.")

    # Save results with separated usage information
    reason_for_csv = reason
    utils.save_result__completed_evaluation(
        result_dir,
        task_identifier,
        agent,
        result,
        reason_for_csv,
        reasoning_mode,
        action_mode,
        attempt_num,
        final_decision_data.get("evaluation_method", "detailed_evaluation"),
        # 移除总计token使用情况的传递，只保留分类的token统计
        # eval_prompt_tokens=total_usage["prompt_tokens"],
        # eval_completion_tokens=total_usage["completion_tokens"],
        # eval_total_tokens=total_usage["total_tokens"],
        # eval_api_cost=total_cost,
        # model_provider=provider_used,
        # model_name=model_used,
        # 步骤描述生成的token使用情况
        step_desc_prompt_tokens=step_desc_usage["prompt_tokens"],
        step_desc_completion_tokens=step_desc_usage["completion_tokens"],
        step_desc_total_tokens=step_desc_usage["total_tokens"],
        step_desc_api_cost=step_desc_cost,
        step_desc_model_name=step_desc_model_name or "",
        step_desc_model_provider=step_desc_model_provider or "",
        # 最终决策的token使用情况
        final_decision_prompt_tokens=final_decision_usage_total["prompt_tokens"],
        final_decision_completion_tokens=final_decision_usage_total[
            "completion_tokens"
        ],
        final_decision_total_tokens=final_decision_usage_total["total_tokens"],
        final_decision_api_cost=final_decision_cost_total,
        final_decision_model_name=final_decision_model_name or "",
        final_decision_model_provider=final_decision_model_provider or "",
        # 失败步骤追踪
        failure_step=evaluation_detail.get("failure_step"),
        # 步骤合理性分析
        reasonable_steps=evaluation_detail.get("reasonable_steps"),
        unreasonable_steps=evaluation_detail.get("unreasonable_steps"),
        step_analysis=evaluation_detail.get("step_analysis"),
        # 任务合理性评估
        task_feasible=task_feasible,
        task_feasible_reason=task_feasible_reason,
        task_barriers=task_barriers,
    )

    # Save a detailed JSON summary
    # logging.info(f"[{task_identifier}] Saving detailed evaluation summary to JSON...")

    final_decision_path = os.path.join(target_dir, "final_decision.json")
    with open(final_decision_path, "w") as f:
        json.dump(final_decision_data, f, indent=4)

    # logging.info(f"[{task_identifier}] Saved final decision to {final_decision_path}")

    summary_data = {
        "task_identifier": task_identifier,
        "task_description": task_description,
        "final_result": evaluation_detail.get("final_result", -1),
        "final_reason": reason,
        "step_by_step_analysis": step_descriptions,
        "step_reasonableness_analysis": {
            "reasonable_steps": evaluation_detail.get("reasonable_steps", []),
            "unreasonable_steps": evaluation_detail.get("unreasonable_steps", []),
            "step_analysis": evaluation_detail.get("step_analysis", {}),
        },
    }

    summary_path = os.path.join(target_dir, "evaluation_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        # logging.info(f"[{task_identifier}] Detailed summary saved to {summary_path}")
    except Exception as e:
        logging.error(f"Error saving JSON summary for {task_identifier}: {e}")

    # Generate and save hint for self-hint functionality
    # Hint is generated when: task failed OR unreasonable steps exist (even if task succeeded)
    hint_result = None
    unreasonable_steps = evaluation_detail.get("unreasonable_steps", [])
    has_unreasonable_steps = len(unreasonable_steps) > 0 if unreasonable_steps else False
    step_analysis = evaluation_detail.get("step_analysis", {})
    
    if self_hint_enabled and (result == 0 or has_unreasonable_steps):
        try:
            # Determine the hint reason based on outcome
            if result == 0:
                hint_reason = reason  # Task failed, use failure reason
            else:
                # Task succeeded but has unreasonable steps
                # Build detailed description for each unreasonable step
                unreasonable_details = []
                for step_num in unreasonable_steps:
                    step_num_str = str(step_num)
                    if step_num_str in step_analysis:
                        analysis = step_analysis[step_num_str]
                        explanation = analysis.get("explanation", "No explanation provided")
                        impact = analysis.get("impact", "unknown")
                        unreasonable_details.append(
                            f"  - Step {step_num}: {explanation} (Impact: {impact})"
                        )
                    else:
                        unreasonable_details.append(f"  - Step {step_num}: Marked as unreasonable")
                
                unreasonable_desc = "\n".join(unreasonable_details) if unreasonable_details else str(unreasonable_steps)
                hint_reason = (
                    f"Task completed successfully, but the execution was inefficient. "
                    f"The following {len(unreasonable_steps)} step(s) were unreasonable and should be improved:\n"
                    f"{unreasonable_desc}\n\n"
                    f"Please analyze these inefficiencies and suggest better approaches."
                )
            
            hint_result = generate_eval_hint(
                task_identifier=task_identifier,
                task_description=task_description,
                failure_reason=hint_reason,
                step_descriptions=step_descriptions,
                step_analysis=step_analysis,
                target_dir=target_dir,
                enable_hints=True,
            )
            if hint_result:
                hint_trigger = "failure" if result == 0 else "unreasonable steps"
                logging.info(f"[{task_identifier}] Self-reflection hint generated (trigger: {hint_trigger})")
                final_decision_data["eval_hint"] = hint_result
        except Exception as e:
            logging.error(f"Error generating hint for {task_identifier}: {e}")

    return final_decision_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MemGUI evaluator for a given task."
    )
    parser.add_argument(
        "--task_identifier",
        type=str,
        required=False,
        default="001-FindProductAndFilter",
        help="The unique identifier for the task.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=False,
        default="./results/debug-session",
        help="The directory where the results are stored.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval",
        choices=["full", "eval"],
        help="Evaluation mode: 'full' includes agent name in path, 'eval' does not.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="GUIOWL",
        help="The name of the agent being evaluated. Required if mode is 'full'.",
    )
    parser.add_argument(
        "--attempt_num",
        type=int,
        default=1,
        help="The attempt number for the task run.",
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="direct",
        choices=["result_only", "direct"],
        help="The reasoning mode for the evaluator.",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="with_action",
        choices=["no_action", "with_action", "text_action"],
        help="The action mode for the evaluator.",
    )
    parser.add_argument(
        "--self_hint_enabled",
        action="store_true",
        default=False,
        help="Enable self-hint: generate hints from failed attempts for subsequent attempts.",
    )

    args = parser.parse_args()

    if args.mode == "full" and not args.agent:
        raise ValueError("Argument --agent is required when mode is 'full'.")

    mobilegym_critic_evaluator(
        task_identifier=args.task_identifier,
        result_dir=args.result_dir,
        mode=args.mode,
        agent=args.agent,
        attempt_num=args.attempt_num,
        reasoning_mode=args.reasoning_mode,
        action_mode=args.action_mode,
        self_hint_enabled=args.self_hint_enabled,
    )
