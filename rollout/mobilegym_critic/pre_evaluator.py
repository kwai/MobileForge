import glob
import os
import logging
import re
from mobilegym_critic.utils.image_utils import stitch_images_horizontally
from mobilegym_critic.utils.prompts import get_pre_evaluation_prompt
from mobilegym_critic.utils.vivo.vivo_api import inference_chat_gemini_1_image
from mobilegym_critic.utils.common import (
    parse_json_from_response,
    log_and_save_interaction,
)

logging.basicConfig(level=logging.INFO)


def run_pre_evaluation(
    task_identifier: str, target_dir: str, task_description: str, log_data: list
) -> dict:
    """
    Runs a pre-evaluation step to quickly determine task success or failure.

    Args:
        task_identifier (str): The unique identifier for the task.
        target_dir (str): The specific result directory for the task run.
        task_description (str): The description of the task.
        log_data (list): The raw log data from log.json.

    Returns:
        dict: A dictionary containing the decision ('Success', 'Failure', 'Uncertain')
              and the reasoning. Returns None if pre-evaluation cannot be run.
    """
    # logging.info(f"[{task_identifier}] Running pre-evaluation...")
    single_actions_dir = os.path.join(target_dir, "single_actions")
    if not os.path.exists(single_actions_dir):
        logging.warning(
            f"[{task_identifier}] single_actions directory not found at {single_actions_dir}. Skipping pre-evaluation."
        )
        return None

    screenshot_files = sorted(glob.glob(os.path.join(single_actions_dir, "*.png")))
    if not screenshot_files:
        logging.warning(
            f"[{task_identifier}] No screenshots found in single_actions. Skipping pre-evaluation."
        )
        return None

    # Sort files numerically based on step number
    def get_step_num(filepath):
        match = re.search(r"step_(\d+)\.png", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    screenshot_files.sort(key=get_step_num)

    # Get last 3 screenshots, or fewer if not available
    last_screenshots_paths = screenshot_files[-3:]

    # Extract step numbers for labels
    last_screenshots_labels = [
        f"Step {get_step_num(p)}" for p in last_screenshots_paths
    ]

    try:
        stitched_image = stitch_images_horizontally(
            last_screenshots_paths, labels=last_screenshots_labels
        )
    except Exception as e:
        logging.error(f"[{task_identifier}] Error stitching images: {e}")
        return None

    puzzle_dir = os.path.join(target_dir, "puzzle")
    os.makedirs(puzzle_dir, exist_ok=True)
    puzzle_path = os.path.join(puzzle_dir, "pre_eval_puzzle.png")
    stitched_image.save(puzzle_path)
    # logging.info(f"[{task_identifier}] Saved pre-evaluation puzzle to {puzzle_path}")

    # Prepare data for the prompt, filtering for only valid steps with actions
    valid_logs = [d for d in log_data if "step" in d and d.get("action")]
    raw_action_logs = [f"Step {d['step']}: {d['action']}" for d in valid_logs]
    total_steps = len(raw_action_logs)

    system_prompt, user_prompt = get_pre_evaluation_prompt(
        task_description, raw_action_logs, total_steps
    )

    try:
        response = inference_chat_gemini_1_image(
            system_prompt, user_prompt, image1=puzzle_path
        )
        
        # Handle both old (string) and new (dict) response formats
        if isinstance(response, dict):
            response_str = response["content"]
            usage_info = response.get("usage", {})
            model_info = {
                "model": response.get("model"),
                "provider": response.get("provider"),
                "api_cost": response.get("api_cost", 0.0)
            }
        else:
            response_str = response
            usage_info = {}
            model_info = {"model": None, "provider": None, "api_cost": 0.0}

        log_and_save_interaction(
            target_dir, "pre_evaluation", system_prompt, user_prompt, response_str
        )

        # The return value is a raw string, which needs to be parsed into JSON.
        json_response = parse_json_from_response(response_str)
        if not json_response or "decision" not in json_response:
            logging.error(
                f"[{task_identifier}] Failed to parse valid JSON from pre-evaluation response: {response_str}"
            )
            return None

        # Add usage and model information to the response
        json_response["_usage_info"] = usage_info
        json_response["_model_info"] = model_info

        # logging.info(
        #     f"[{task_identifier}] Pre-evaluation decision: {json_response['decision']}"
        # )
        return json_response

    except Exception as e:
        logging.error(
            f"[{task_identifier}] Exception during pre-evaluation LLM call: {e}"
        )
        return None
