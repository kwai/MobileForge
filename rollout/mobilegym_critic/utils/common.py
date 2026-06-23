import re
import json
import os
import datetime
import logging


def parse_json_from_response(response: str) -> dict:
    """
    Extracts and parses a JSON object from a model's string response,
    which might be embedded in markdown or missing surrounding braces.
    """
    # Try to find a JSON object using regex
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        # If no JSON object is found, clean up markdown and try to add braces
        clean_str = re.sub(
            r"```(json)?\n|```", "", response, flags=re.IGNORECASE
        ).strip()
        if clean_str.startswith('"'):
            json_str = "{" + clean_str + "}"
        else:
            # If it's not a clear-cut case, raise an error
            raise ValueError(
                f"Could not find a valid JSON object in the response: '{response}'"
            )
    return json.loads(json_str)


def log_and_save_interaction(
    target_dir: str,
    prompt_stage: str,
    system_prompt: str,
    user_prompt: str,
    llm_response: str,
):
    """
    Logs prompts and their responses to the console and saves them to a JSON file.
    """
    # logging.info(f"\033[95m\n--- Interaction for: {prompt_stage} ---\033[0m")
    # logging.info("\033[96mSystem Prompt:\033[0m\n%s", system_prompt)
    # logging.info("\033[96mUser Prompt:\033[0m\n%s", user_prompt)
    # logging.info("\033[93mLLM Raw Response:\033[0m\n%s", llm_response)

    log_file = os.path.join(target_dir, "prompt_logs.json")

    new_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": prompt_stage,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_response,
    }

    try:
        if os.path.exists(log_file):
            with open(log_file, "r+") as f:
                logs = json.load(f)
                logs.append(new_log)
                f.seek(0)
                json.dump(logs, f, indent=4)
        else:
            with open(log_file, "w") as f:
                json.dump([new_log], f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save prompt log: {e}")
