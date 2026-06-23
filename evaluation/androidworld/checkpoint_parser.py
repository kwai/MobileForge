import argparse
import json
import pandas as pd
import os
from collections import Counter
import gzip
import pickle
import io
import base64
from PIL import Image
import numpy as np
from android_world import checkpointer
import zlib
import re
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_TEMPLATE_COLUMN = "task_template"
TASK_PROMPT_COLUMN = "task_prompt"


def process_numpy_array(arr, compress=True):
    """处理numpy数组，返回包含元数据和压缩数据的字典"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if compress:
        # 使用zlib压缩二进制数据
        compressed = zlib.compress(arr.tobytes())
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": base64.b64encode(compressed).decode("utf-8"),
            "compressed": True,
        }
    else:
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
            "compressed": False,
        }


def restore_numpy_array(data_dict):
    """从字典恢复numpy数组"""
    if not data_dict.get("__ndarray__", False):
        return data_dict

    buffer = base64.b64decode(data_dict["data"])
    if data_dict.get("compressed", False):
        buffer = zlib.decompress(buffer)

    arr = np.frombuffer(buffer, dtype=np.dtype(data_dict["dtype"]))
    return arr.reshape(data_dict["shape"])


def deep_process_structure(data, process_fn):
    """深度遍历数据结构并应用处理函数"""
    if isinstance(data, np.ndarray):
        return process_fn(data)
    elif isinstance(data, list):
        return [deep_process_structure(item, process_fn) for item in data]
    elif isinstance(data, dict):
        return {k: deep_process_structure(v, process_fn) for k, v in data.items()}
    return data


def save_screenshot_from_numpy(arr, output_path):
    """从numpy数组保存为PNG文件"""
    try:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # 假设数组是HWC格式（高度、宽度、通道）
        if arr.shape[-1] == 3:  # RGB
            img = Image.fromarray(arr, "RGB")
        elif arr.shape[-1] == 4:  # RGBA
            img = Image.fromarray(arr, "RGBA")
        else:
            print(f"Unsupported channel format: {arr.shape}")
            return False

        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error saving numpy array to {output_path}: {e}")
        return False


def list_directory_contents(directory):
    """List all files and subdirectories in the directory"""
    print(f"\nContents of directory '{directory}':")
    if not os.path.exists(directory):
        print(f"Error: Directory does not exist!")
        return

    files = os.listdir(directory)
    if not files:
        print("Directory is empty")
        return

    for item in sorted(files):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            size = os.path.getsize(item_path)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            print(f"  📄 {item} ({size_str})")


def _unzip_and_read_pickle(file_path):
    """Read a gzipped pickle file"""
    try:
        with open(file_path, "rb") as f:
            compressed = _dechunk_bytes(f.read())

        gzip_start = compressed.find(b"\x1f\x8b")
        if gzip_start > 0:
            compressed = compressed[gzip_start:]

        with gzip.open(io.BytesIO(compressed), "rb") as f_in:
            return pickle.load(f_in)
    except Exception as e:
        print(f"Error: Cannot parse file {file_path}: {e}")
        return None


def _dechunk_bytes(raw):
    """Decode HTTP chunked-style files when checkpoint artifacts were saved that way."""
    line_end = raw.find(b"\n")
    if line_end < 0:
        return raw

    first_line = raw[:line_end].strip()
    try:
        int(first_line.split(b";", 1)[0], 16)
    except ValueError:
        return raw

    chunks = []
    pos = 0
    try:
        while pos < len(raw):
            line_end = raw.find(b"\n", pos)
            if line_end < 0:
                return raw
            size_line = raw[pos:line_end].strip()
            size = int(size_line.split(b";", 1)[0], 16)
            pos = line_end + 1

            if size == 0:
                return b"".join(chunks)

            chunk = raw[pos:pos + size]
            if len(chunk) != size:
                return raw
            chunks.append(chunk)
            pos += size

            if raw[pos:pos + 2] == b"\r\n":
                pos += 2
            elif raw[pos:pos + 1] == b"\n":
                pos += 1

        return b"".join(chunks) if chunks else raw
    except Exception:
        return raw


def _load_json_file(path):
    """Load normal JSON or files with a short prefix before the JSON payload."""
    with open(path, "rb") as f:
        text = _dechunk_bytes(f.read()).decode("utf-8")

    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(stripped)
        return payload

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON payload found", text, 0)
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[min(starts):])
    return payload


def save_screenshot(screenshot_array, output_path):
    """将numpy数组保存为PNG文件"""
    try:
        # 确保数组类型为uint8
        if screenshot_array is None:
            return False
        if screenshot_array.dtype != np.uint8:
            screenshot_array = screenshot_array.astype(np.uint8)

        # 检查数组维度是否符合图像格式
        if len(screenshot_array.shape) not in (3, 4):
            print(f"无效的数组维度：{screenshot_array.shape}")
            return False

        # 转换为PIL图像
        if screenshot_array.shape[-1] == 3:  # RGB格式
            img = Image.fromarray(screenshot_array, "RGB")
        elif screenshot_array.shape[-1] == 4:  # RGBA格式
            img = Image.fromarray(screenshot_array, "RGBA")
        else:
            print(f"不支持的通道数：{screenshot_array.shape[-1]}")
            return False

        # 创建目录并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"保存截图失败：{e}")
        return False


def parse_response_text(response_text):
    """Parse raw_response text to extract Thought, Action/Tool Call, Memory Call, and Conclusion separately"""
    if not response_text or not isinstance(response_text, str):
        return {
            "thought": "",
            "action": "",
            "action_text": "",
            "conclusion": "",
            "memory": "",
        }

    thought_content = ""
    action_content = ""
    action_text_content = ""
    conclusion_content = ""
    memory_content = ""

    # 1. Try to extract tags first (High priority)
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
    tool_call_match = re.search(
        r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL
    )
    memory_call_match = re.search(
        r"<memory_call>(.*?)</memory_call>", response_text, re.DOTALL
    )
    conclusion_match = re.search(
        r"<conclusion>(.*?)</conclusion>", response_text, re.DOTALL
    )

    thinking_tag_text = thinking_match.group(1).strip() if thinking_match else ""
    tool_call_json = tool_call_match.group(1).strip() if tool_call_match else ""
    memory_call_json = memory_call_match.group(1).strip() if memory_call_match else ""
    conclusion_tag_text = conclusion_match.group(1).strip() if conclusion_match else ""

    # 2. If tags found, use them.
    if thinking_tag_text:
        thought_content = thinking_tag_text

    if tool_call_json:
        try:
            parsed_json = json.loads(tool_call_json)
            formatted_tool_call = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            action_content = f"```json\n{formatted_tool_call}\n```"
        except:
            if not tool_call_json.strip().startswith("```"):
                action_content = f"```json\n{tool_call_json}\n```"
            else:
                action_content = tool_call_json

    if memory_call_json:
        try:
            parsed_json = json.loads(memory_call_json)
            formatted_memory_call = json.dumps(
                parsed_json, indent=2, ensure_ascii=False
            )
            memory_content = f"```json\n{formatted_memory_call}\n```"
        except:
            if not memory_call_json.strip().startswith("```"):
                memory_content = f"```json\n{memory_call_json}\n```"
            else:
                memory_content = memory_call_json

    if conclusion_tag_text:
        conclusion_content = conclusion_tag_text

    # Extract Action Text (text between thinking and tool_call, or starting with Action:)
    # Remove thinking block to search for action text
    text_without_thinking = re.sub(
        r"<thinking>.*?</thinking>", "", response_text, flags=re.DOTALL
    )

    # Search for text starting with "Action:" up to <tool_call> or end
    action_text_match = re.search(
        r"Action:\s*(.*?)\s*(?=<tool_call>|$)",
        text_without_thinking,
        re.DOTALL | re.IGNORECASE,
    )
    if action_text_match:
        action_text_content = action_text_match.group(1).strip()
    else:
        # Fallback: try to find text before tool_call if no "Action:" label
        # But be careful not to capture empty space or just newlines
        pre_tool_match = re.search(
            r"(.*?)\s*(?=<tool_call>)", text_without_thinking, re.DOTALL
        )
        if pre_tool_match:
            candidate = pre_tool_match.group(1).strip()
            if candidate and not candidate.startswith(
                "<"
            ):  # Avoid capturing other tags
                action_text_content = candidate

    # 3. If no tags found (or partial), try to parse "Thought:" and "Action:" patterns
    # Only if we didn't find the corresponding tag content

    remaining_text = response_text
    if thinking_match:
        remaining_text = remaining_text.replace(thinking_match.group(0), "")
    if tool_call_match:
        remaining_text = remaining_text.replace(tool_call_match.group(0), "")
    if conclusion_match:
        remaining_text = remaining_text.replace(conclusion_match.group(0), "")

    # Regex for Thought: ... Action: ...
    # Look for "Thought:" followed by content until "Action:" or end
    if not thought_content:
        thought_pattern = re.search(
            r"Thought:\s*(.*?)(?=Action:|Conclusion:|$)",
            remaining_text,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_pattern:
            thought_content = thought_pattern.group(1).strip()

    if not action_content:
        # Look for "Action:" followed by content until "Conclusion:" or end
        action_pattern = re.search(
            r"Action:\s*(.*?)(?=Conclusion:|$)",
            remaining_text,
            re.DOTALL | re.IGNORECASE,
        )
        if action_pattern:
            action_text = action_pattern.group(1).strip()
            # Check if it looks like JSON
            if action_text.startswith("{") or action_text.startswith("```"):
                action_content = action_text
            else:
                # If it's just text, treat it as action text if we haven't found one yet
                if not action_text_content:
                    action_text_content = action_text
                # And maybe it implies an action? For now let's leave action_content empty if no JSON found

    # Also check for markdown json block if we still don't have action
    if not action_content:
        json_match = re.search(r"```json\s*(.*?)\s*```", remaining_text, re.DOTALL)
        if json_match:
            action_content = json_match.group(0)  # Keep the code block

    return {
        "thought": thought_content,
        "action": action_content,
        "action_text": action_text_content,
        "conclusion": conclusion_content,
        "memory": memory_content,
    }


def load_task_metadata(task_name, output_dir):
    """Load task metadata from JSON file"""
    task_dir = os.path.join(output_dir, task_name)
    json_file = os.path.join(task_dir, f"{task_name}.json")

    if not os.path.exists(json_file):
        return None

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data and len(data) > 0:
            episode = data[0]  # Get first episode
            # Sanitize fields to avoid NaN/invalid types
            import math

            goal = episode.get("goal", "")
            is_successful = bool(episode.get("is_successful", 0))
            run_time = episode.get("run_time", 0)
            try:
                run_time = float(run_time)
            except Exception:
                run_time = 0.0
            if isinstance(run_time, float) and math.isnan(run_time):
                run_time = 0.0

            ep_data = episode.get("episode_data", {})
            if not isinstance(ep_data, dict):
                # Some JSONs may contain NaN or other non-dict values
                ep_data = {}

            return {
                "goal": goal,
                "is_successful": is_successful,
                "run_time": run_time,
                "episode_data": ep_data,
            }
    except Exception as e:
        print(f"Error loading metadata for {task_name}: {e}")

    return None


def extract_screenshots_from_task(task_data, task_name, output_dir):
    """Extract screenshots from task data and save as PNG files"""
    screenshots_dir = os.path.join(output_dir, task_name, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    screenshot_count = 0
    episode_data = task_data[0].get("episode_data")
    if not isinstance(episode_data, dict):
        print(
            f"  Warning: episode_data is not a dict for task {task_name}; skipping screenshot extraction"
        )
        return 0

    # Define screenshot types to extract
    screenshot_types = [
        "raw_screenshot",
        "before_screenshot",
        "after_screenshot",
        "before_screenshot_with_som",
        "after_screenshot_with_som",
        "combined_screenshot",
    ]

    print(f"Processing screenshots for task {task_name}...")

    for screenshot_type in screenshot_types:
        if screenshot_type in episode_data:
            screenshot_list = episode_data[screenshot_type]
            if screenshot_list:
                print(f"  Found {len(screenshot_list)} {screenshot_type} screenshots")

                for screenshot_idx, screenshot_data in enumerate(screenshot_list):
                    filename = f"{screenshot_type}_{screenshot_idx}.png"
                    save_path = os.path.join(screenshots_dir, filename)

                    # Handle both numpy arrays and string representations
                    screenshot_array = screenshot_data
                    if isinstance(screenshot_data, str):
                        try:
                            # Parse string representation back to numpy array
                            import ast

                            screenshot_array = np.array(
                                ast.literal_eval(screenshot_data)
                            )
                        except (ValueError, SyntaxError) as e:
                            print(
                                f"  Warning: Could not parse screenshot {screenshot_idx} of type {screenshot_type}: {e}"
                            )
                            continue

                    if save_screenshot(screenshot_array, save_path):
                        screenshot_count += 1
                    else:
                        print(f"  Failed to save {filename}")

    print(f"  Successfully extracted {screenshot_count} screenshots")
    return screenshot_count


def extract_pkl_gz_to_readable(
    file_path, output_dir="extracted_data", extract_images=True
):
    """Extract a pkl.gz file to a readable JSON format and extract screenshots"""
    try:
        data = _unzip_and_read_pickle(file_path)
        if data is None:
            return False

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename without extension
        base_name = os.path.basename(file_path).replace(".pkl.gz", "")
        output_path = os.path.join(output_dir, base_name)
        os.makedirs(output_path, exist_ok=True)

        # Create a clean version of data without binary screenshots for JSON
        clean_data = []
        for episode in data:
            clean_episode = episode.copy()

            clean_data.append(clean_episode)

        # Save as JSON
        output_file = os.path.join(output_path, f"{base_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, default=str)

        # Extract screenshots if requested
        screenshot_count = 0
        if extract_images:
            screenshot_count = extract_screenshots_from_task(
                data, base_name, output_dir
            )
            print(f"Extracted {screenshot_count} screenshots from {file_path}")

        print(f"Extracted {file_path} to {output_file}")
        return True
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False


def get_metadata_df():
    """Load the metadata CSV file"""
    metadata_candidates = [
        os.path.join(
            SCRIPT_DIR,
            "docs",
            "androidworld-filter-memory - task_metadata_filled.csv",
        ),
        os.path.join("docs", "androidworld-filter-memory - task_metadata_filled.csv"),
    ]
    metadata_path = next((p for p in metadata_candidates if os.path.exists(p)), None)

    if metadata_path:
        try:
            df = pd.read_csv(metadata_path)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            if "tags" in df.columns:
                df["tags"] = df["tags"].apply(normalize_tags)
            return df
        except Exception as e:
            print(f"Error loading metadata: {e}")
    else:
        print("Metadata file not found. Tried:")
        for path in metadata_candidates:
            print(f"  - {path}")
    return None


def normalize_tags(tags):
    """Normalize metadata tags to the same list form used by run.py summaries."""
    if isinstance(tags, np.ndarray):
        tags = tags.tolist()
    if isinstance(tags, list):
        normalized = []
        for tag in tags:
            normalized.extend(normalize_tags(tag))
        return normalized or ["untagged"]
    if isinstance(tags, str):
        stripped = tags.strip()
        if not stripped:
            return ["untagged"]
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(stripped)
                return normalize_tags(parsed)
            except Exception:
                pass
        parts = [part.strip() for part in stripped.split(";") if part.strip()]
        return parts or ["untagged"]
    if pd.isna(tags):
        return ["untagged"]
    return [str(tags)]


def extract_base_task_name(task_name):
    """Extract base task name by removing suffix after underscore (e.g., TaskName_0 -> TaskName)"""
    if "_" in task_name:
        # Find the last underscore and check if what follows is a digit
        parts = task_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return task_name


def get_memory_tasks_set():
    """Get a set of task names that are memory tasks"""
    df = get_metadata_df()
    if df is not None:
        print(f"Metadata columns: {df.columns.tolist()}")
        if "memory-task" in df.columns and "task_name" in df.columns:
            # Ensure memory-task is numeric
            df["memory-task"] = pd.to_numeric(
                df["memory-task"], errors="coerce"
            ).fillna(0)
            memory_tasks = set(df[df["memory-task"] == 1]["task_name"].values)
            print(f"Identified {len(memory_tasks)} memory tasks: {memory_tasks}")
            return memory_tasks
        else:
            print("Required columns 'memory-task' or 'task_name' not found in metadata")
    else:
        print("Failed to load metadata dataframe")
    return set()


def enrich_df_with_metadata(df, task_name_col):
    """Merge metadata columns into the DataFrame"""
    metadata_df = get_metadata_df()
    if metadata_df is None or df.empty:
        return df

    try:
        # Select required columns
        cols_to_merge = [
            "task_name",
            "task_template",
            "difficulty",
            "tags",
            "optimal_steps",
            "memory-task",
        ]
        # Ensure columns exist
        cols_to_merge = [c for c in cols_to_merge if c in metadata_df.columns]

        metadata_subset = metadata_df[cols_to_merge].copy()

        # Merge
        # We need to merge on task_name_col from df and "task_name" from metadata
        merged_df = pd.merge(
            df, metadata_subset, left_on=task_name_col, right_on="task_name", how="left"
        )

        return merged_df
    except Exception as e:
        print(f"Error merging metadata: {e}")
        return df


def extract_all_pkl_gz_files(directory, output_dir="extracted_data"):
    """Extract all pkl.gz files in a directory to readable JSON format"""
    print(f"\nExtracting all pkl.gz files in '{directory}' to '{output_dir}'...")

    task_files = [f for f in os.listdir(directory) if f.endswith(".pkl.gz")]

    if not task_files:
        print("No pkl.gz files found")
        return 0

    print(f"Found {len(task_files)} pkl.gz files")

    success_count = 0
    for i, file_name in enumerate(task_files):
        if i % 10 == 0:
            print(f"Processing: {i + 1}/{len(task_files)} files...")

        file_path = os.path.join(directory, file_name)
        if extract_pkl_gz_to_readable(file_path, output_dir):
            success_count += 1

    print(f"Successfully extracted {success_count} out of {len(task_files)} files")
    return success_count


def create_html_report(
    task_name, output_dir="extracted_data", experiment_name="", summary_output_dir=None
):
    """Create an enhanced HTML report with task metadata and step information, matching the reference template"""
    task_dir = os.path.join(output_dir, task_name)
    screenshots_dir = os.path.join(task_dir, "screenshots")

    # Load task metadata
    metadata = load_task_metadata(task_name, output_dir)
    if not metadata:
        print(f"No metadata found for task {task_name}")
        return False

    html_file = os.path.join(task_dir, "report.html")

    # Determine back link
    back_link = f"../{experiment_name}.html"
    if summary_output_dir:
        summary_file = os.path.join(summary_output_dir, f"{experiment_name}.html")
        back_link = os.path.relpath(summary_file, task_dir)

    # Prepare data for template
    is_successful = bool(metadata.get("is_successful", 0))
    status_text = "Success" if is_successful else "Failure"

    # Check if this is a memory task
    memory_tasks = get_memory_tasks_set()
    base_task_name = extract_base_task_name(task_name)
    is_memory_task = base_task_name in memory_tasks
    memory_badge = '<span class="memory-badge">Memory</span>' if is_memory_task else ""

    episode_data = metadata.get("episode_data", {})
    if not isinstance(episode_data, dict):
        episode_data = {}

    raw_responses = episode_data.get("raw_response", [])
    raw_actions = episode_data.get("raw_action", [])

    # Get screenshots
    screenshot_files = []
    if os.path.exists(screenshots_dir):
        screenshot_files = os.listdir(screenshots_dir)

    # Prefer raw_screenshot for the main view as per template
    raw_screenshots = sorted(
        [f for f in screenshot_files if f.startswith("raw_screenshot_")],
        key=lambda x: int(x.split("_")[-1].replace(".png", "")),
    )

    before_screenshots = sorted(
        [f for f in screenshot_files if f.startswith("before_screenshot_")],
        key=lambda x: int(x.split("_")[-1].replace(".png", "")),
    )

    # Check for Qwen3-VL specific data
    qwen3_vl_step_data = episode_data.get("qwen3_vl_step_data")

    steps_data = []

    if (
        qwen3_vl_step_data
        and isinstance(qwen3_vl_step_data, list)
        and len(qwen3_vl_step_data) > 0
    ):
        # Use Qwen3-VL structured data
        # The last element contains the full history of steps
        full_steps = qwen3_vl_step_data[-1]

        for i, step in enumerate(full_steps):
            step_info = {
                "prefix": step.get("step_id", i + 1),
                "image": None,
                "image_width": "",
                "image_height": "",
                "x": "",
                "y": "",
                "thinking_content": "",
                "tool_call_content": "",
                "action_text": "",
                "conclusion_content": step.get("conclusion", ""),
                "memory_call_content": "",
                "raw_response": "",
                "user_prompt": "",
                "system_prompt": "",
            }

            # Parse action output for thinking and tool call
            action_output = step.get("action_output", "")
            if action_output:
                parsed = parse_response_text(action_output)
                step_info["thinking_content"] = parsed["thought"]
                step_info["tool_call_content"] = parsed["action"]
                step_info["action_text"] = parsed["action_text"]
                step_info["memory_call_content"] = parsed.get("memory", "")
                step_info["raw_response"] = action_output

            # Get user prompt and system prompt if available
            step_info["user_prompt"] = step.get("action_prompt", "")
            step_info["system_prompt"] = step.get("system_prompt", "System prompt not stored in checkpoint data")

            # Coordinates from parsed_action
            parsed_action = step.get("parsed_action", {})

            # Fallback if tool_call_content is empty but parsed_action exists
            if not step_info["tool_call_content"] and parsed_action:
                try:
                    step_info["tool_call_content"] = (
                        f"```json\n{json.dumps(parsed_action, indent=2, ensure_ascii=False)}\n```"
                    )
                except:
                    pass

            # Image
            # Map step index to before_screenshot
            # Assuming step_id starts at 1, so index is step_id - 1
            # Or just use loop index i
            img_name_before = f"before_screenshot_{i}.png"
            if img_name_before in screenshot_files:
                step_info["image"] = img_name_before

            # Coordinates from parsed_action
            parsed_action = step.get("parsed_action", {})
            if parsed_action and isinstance(parsed_action, dict):
                if parsed_action.get("action_type") == "click":
                    step_info["x"] = parsed_action.get("x", "")
                    step_info["y"] = parsed_action.get("y", "")

            steps_data.append(step_info)

        # Check if we missed the final termination step (or any trailing steps)
        # This happens because qwen3_vl_step_data might not include the final 'terminate' action
        # if it didn't produce a new observation/step in the agent's internal tracking
        if len(steps_data) < len(raw_responses):
            for i in range(len(steps_data), len(raw_responses)):
                step_info = {
                    "prefix": i + 1,
                    "image": None,
                    "image_width": "",
                    "image_height": "",
                    "x": "",
                    "y": "",
                    "thinking_content": "",
                    "tool_call_content": "",
                    "action_text": "",
                    "conclusion_content": "",
                    "memory_call_content": "",
                    "raw_response": "",
                    "user_prompt": "",
                    "system_prompt": "",
                }

                # Parse response
                parsed = parse_response_text(raw_responses[i])
                step_info["thinking_content"] = parsed["thought"]
                step_info["tool_call_content"] = parsed["action"]
                step_info["action_text"] = parsed["action_text"]
                step_info["conclusion_content"] = parsed["conclusion"]
                step_info["memory_call_content"] = parsed.get("memory", "")
                step_info["raw_response"] = (
                    raw_responses[i] if i < len(raw_responses) else ""
                )
                step_info["user_prompt"] = "Not available in checkpoint data"
                step_info["system_prompt"] = "Not available in checkpoint data"

                # Try to find an image if available
                img_name_before = f"before_screenshot_{i}.png"
                if img_name_before in screenshot_files:
                    step_info["image"] = img_name_before

                steps_data.append(step_info)

    else:
        # Fallback to generic parsing logic
        # Determine max steps
        max_steps = max(
            len(raw_responses),
            len(raw_actions),
            len(raw_screenshots),
            len(before_screenshots),
        )

        for i in range(max_steps):
            step_info = {
                "prefix": i + 1,
                "image": None,
                "image_width": "",
                "image_height": "",
                "x": "",
                "y": "",
                "thinking_content": "",
                "tool_call_content": "",
                "action_text": "",
                "conclusion_content": "",
                "memory_call_content": "",
                "raw_response": "",
                "user_prompt": "",
                "system_prompt": "",
            }

            # Content (Extract first to use for coordinate extraction)
            if i < len(raw_responses):
                parsed = parse_response_text(raw_responses[i])
                step_info["thinking_content"] = parsed["thought"]
                step_info["tool_call_content"] = parsed["action"]
                step_info["action_text"] = parsed["action_text"]
                step_info["conclusion_content"] = parsed["conclusion"]
                step_info["memory_call_content"] = parsed.get("memory", "")
                step_info["raw_response"] = raw_responses[i]
                step_info["user_prompt"] = "Not available in checkpoint data"
                step_info["system_prompt"] = "Not available in checkpoint data"

                # If action was not in response but in raw_actions (e.g. pure code agent)
                if not step_info["tool_call_content"] and i < len(raw_actions):
                    step_info["tool_call_content"] = raw_actions[i]
            elif i < len(raw_actions):
                step_info["tool_call_content"] = raw_actions[i]

            # Image
            # Try raw_screenshot first, then before_screenshot
            img_name_raw = f"raw_screenshot_{i}.png"
            img_name_before = f"before_screenshot_{i}.png"

            if img_name_raw in screenshot_files:
                step_info["image"] = img_name_raw
            elif img_name_before in screenshot_files:
                step_info["image"] = img_name_before

            if step_info["image"] and step_info["tool_call_content"]:
                # Try to extract click coordinates from action
                try:
                    # Try parsing as JSON first
                    action_data = None
                    content = step_info["tool_call_content"]

                    # If content is wrapped in code blocks, strip them
                    if content.startswith("```json"):
                        content = (
                            content.replace("```json", "").replace("```", "").strip()
                        )
                    elif content.startswith("```"):
                        content = content.replace("```", "").strip()

                    try:
                        action_data = json.loads(content)
                    except:
                        pass

                    if action_data and isinstance(action_data, dict):
                        # Check for mobile_use format
                        # {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}
                        if action_data.get("name") == "mobile_use":
                            args = action_data.get("arguments", {})
                            if args.get("action") == "click":
                                coords = args.get("coordinate")
                                if (
                                    coords
                                    and isinstance(coords, list)
                                    and len(coords) >= 2
                                ):
                                    step_info["x"] = coords[0]
                                    step_info["y"] = coords[1]

                        # Check for other formats if needed
                        # e.g. {"action": "click", "coordinate": [x, y]}
                        elif action_data.get("action") == "click":
                            coords = action_data.get("coordinate")
                            if coords and isinstance(coords, list) and len(coords) >= 2:
                                step_info["x"] = coords[0]
                                step_info["y"] = coords[1]

                except Exception as e:
                    print(f"Error extracting coordinates for step {i}: {e}")

            steps_data.append(step_info)

    # CSS from reference template
    css_content = """
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --success-color: #2ecc71;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
            --border-color: #dee2e6;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f8fa;
        }

        header {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        
        .nav-links {
            margin-top: 10px;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-right: 15px;
            font-weight: bold;
        }
        
        .nav-links a:hover {
            text-decoration: underline;
        }

        .task-overview {
            background-color: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid var(--primary-color);
        }

        .goal {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--secondary-color);
        }

        .status {
            padding: 15px;
            border-radius: 6px;
            background-color: var(--light-bg);
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.5; 
        }
        
        .status.success {
            border-left: 5px solid var(--success-color);
            background-color: #e8f5e9;
        }
        
        .status.failure {
            border-left: 5px solid var(--danger-color);
            background-color: #ffebee;
        }

        .steps-container {
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        thead th {
            background-color: var(--secondary-color);
            color: white;
            padding: 15px 20px;
            text-align: left;
            position: sticky;
            top: 0;
        }

        tbody td {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
        }

        .step-header {
            background-color: var(--light-bg);
            font-weight: 600;
            padding: 8px 15px;
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .image-container {
            max-width: 100%;
            text-align: center;
        }

        .step-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            display: block;
            margin: 0 auto;
        }

        .text-content {
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            padding: 15px;
            background-color: var(--light-bg);
            border-radius: 4px;
            line-height: 1.8;
        }

        .no-image {
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9rem;
        }

        .no-data {
            text-align: center;
            padding: 30px;
            background-color: #f8d7da;
            border-radius: 8px;
            color: #721c24;
            margin-top: 20px;
        }

        .click-dot {
            position: absolute;
            width: 12px;
            height: 12px;
            background: #ff4444;
            border-radius: 50%;
            box-shadow: 0 0 4px rgba(255,0,0,0.5);
        }
        
        .img-wrapper {
            position: relative;  
            display: inline-block;  
            margin: 0 auto;  
        }

        /* Qwen3-VL Optimization */
        .action-summary {
            margin-bottom: 10px;
            padding: 8px;
            background-color: #e3f2fd;
            border-radius: 4px;
            border-left: 4px solid #2196f3;
        }
        
        .action-detail {
            margin-left: 10px;
            font-size: 0.95em;
            color: #555;
        }
        
        .action-text {
            margin-bottom: 10px;
            padding: 8px;
            background-color: #fff3e0;
            border-radius: 4px;
            border-left: 4px solid #ff9800;
            font-style: italic;
        }
        
        .json-block {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .action-type {
            font-weight: bold;
            color: #1565c0;
            text-transform: uppercase;
        }
        
        .memory-badge {
            display: inline-block;
            background-color: #9c27b0;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-left: 8px;
            vertical-align: middle;
        }
        
        .memory-call-block {
            margin-top: 10px;
            padding: 10px;
            background-color: #f3e5f5;
            border-radius: 4px;
            border-left: 4px solid #9c27b0;
        }
        
        .memory-header {
            font-weight: bold;
            color: #9c27b0;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .raw-info-toggle {
            margin-top: 10px;
            padding: 8px 12px;
            background-color: #e3f2fd;
            border-radius: 4px;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.3s;
            text-align: center;
            font-weight: 500;
        }
        
        .raw-info-toggle:hover {
            background-color: #bbdefb;
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            animation: fadeIn 0.3s;
        }
        
        .modal.visible {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background-color: #fff;
            border-radius: 8px;
            width: 90%;
            max-width: 1400px;
            max-height: 90vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s;
        }
        
        @keyframes slideIn {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .modal-header {
            background: linear-gradient(135deg, #1976d2, #1565c0);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .modal-title {
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        .modal-close {
            background: none;
            border: none;
            color: white;
            font-size: 2rem;
            cursor: pointer;
            padding: 0;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: background-color 0.3s;
        }
        
        .modal-close:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        
        .modal-body {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
        }
        
        .raw-section {
            margin-bottom: 25px;
        }
        
        .raw-section-title {
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 10px;
            font-size: 1.1em;
            padding-bottom: 5px;
            border-bottom: 2px solid #e3f2fd;
        }
        
        .raw-section-content {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .raw-section-content::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .raw-section-content::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        .raw-section-content::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        
        .raw-section-content::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
    """

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Android World Visualization</title>
    <meta charset="UTF-8">
    <style>
        {css_content}
    </style>
</head>
<body>
    <header>
        <h1>Android World Visualization</h1>
        <div class="nav-links">
            <a href="{back_link}">← Back to Summary</a>
            <span>Current Task: {task_name}{memory_badge}</span>
        </div>
    </header>

    <div class="task-overview">
        <div class="goal">Goal: {metadata["goal"]}</div>
        <div class="status {status_text.lower()}">Status: {status_text}</div>
    </div>

    <div class="steps-container">
        <table>
            <thead>
                <tr>
                    <th style="width: 30%">Screenshot</th>
                    <th style="width: 17%">Think</th>
                    <th style="width: 17%">Action</th>
                    <th style="width: 17%">Memory</th>
                    <th style="width: 17%">Conclusion</th>
                </tr>
            </thead>
            <tbody>
""")

        for step in steps_data:
            f.write("<tr>\n")

            # Screenshot Column
            f.write("<td>\n")
            if step["image"]:
                img_path = f"screenshots/{step['image']}"

                # Add click dot if coordinates exist
                dot_html = ""
                if step["x"] != "" and step["y"] != "":
                    dot_html = f"""
                    <div class="click-dot"
                        data-original-x="{step["x"]}"
                        data-original-y="{step["y"]}">
                    </div>
                    """

                f.write(f"""
                    <div class="image-container">
                        <div class="step-header">step {step["prefix"]}</div>
                        <div class="img-wrapper">
                            <img
                                src="{img_path}"
                                alt="screenshot"
                                class="step-image"
                                data-original-width="{step["image_width"]}"
                                data-original-height="{step["image_height"]}"
                            >
                            {dot_html}
                        </div>
                    </div>
""")
            else:
                f.write('<div class="no-image">this step no image</div>')
            f.write("</td>\n")

            # Think Column
            f.write("<td>\n")
            f.write('<div class="step-header">Think</div>')
            f.write(f'<div class="text-content">{step["thinking_content"]}</div>')
            f.write("</td>\n")

            # Action Column
            f.write("<td>\n")
            f.write('<div class="step-header">Action</div>')

            # Optimize display for Qwen3-VL JSON output
            action_content = step["tool_call_content"]
            action_text = step.get("action_text", "")

            display_html = ""

            # Add action text description if available
            if action_text:
                display_html += f'<div class="action-text">{action_text}</div>'

            # Try to parse as JSON to create a better display
            json_display = ""
            try:
                clean_content = action_content
                if clean_content.startswith("```json"):
                    clean_content = (
                        clean_content.replace("```json", "").replace("```", "").strip()
                    )
                elif clean_content.startswith("```"):
                    clean_content = clean_content.replace("```", "").strip()

                if clean_content.startswith("{") and clean_content.endswith("}"):
                    data = json.loads(clean_content)

                    summary_html = ""
                    if isinstance(data, dict):
                        # Handle mobile_use tool
                        if data.get("name") == "mobile_use":
                            args = data.get("arguments", {})
                            action_type = args.get("action", "unknown")
                            summary_html += f'<div class="action-summary"><span class="action-type">{action_type}</span></div>'

                            if "coordinate" in args:
                                coords = args["coordinate"]
                                summary_html += f'<div class="action-detail"><strong>Coordinate:</strong> {coords}</div>'

                            if "text" in args:
                                text = args["text"]
                                summary_html += f'<div class="action-detail"><strong>Text:</strong> "{text}"</div>'

                            if "direction" in args:
                                direction = args["direction"]
                                summary_html += f'<div class="action-detail"><strong>Direction:</strong> {direction}</div>'

                        # Handle generic action format
                        elif "action" in data:
                            action_type = data.get("action")
                            summary_html += f'<div class="action-summary"><span class="action-type">{action_type}</span></div>'

                            if "coordinate" in data:
                                coords = data["coordinate"]
                                summary_html += f'<div class="action-detail"><strong>Coordinate:</strong> {coords}</div>'

                    if summary_html:
                        # Pretty print the JSON
                        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
                        json_display = (
                            f'{summary_html}<div class="json-block">{pretty_json}</div>'
                        )
            except Exception as e:
                # Fallback to original content if parsing fails
                pass

            if json_display:
                display_html += json_display
            elif action_content:
                display_html += f'<div class="json-block">{action_content}</div>'

            f.write(f'<div class="text-content">{display_html}</div>')
            f.write("</td>\n")

            # Memory Column
            f.write("<td>\n")
            f.write('<div class="step-header">Memory</div>')

            memory_call_content = step.get("memory_call_content", "")
            if memory_call_content:
                # Try to parse and display memory call nicely
                memory_display = ""
                try:
                    clean_memory = memory_call_content
                    if clean_memory.startswith("```json"):
                        clean_memory = (
                            clean_memory.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )
                    elif clean_memory.startswith("```"):
                        clean_memory = clean_memory.replace("```", "").strip()

                    if clean_memory.startswith("{"):
                        mem_data = json.loads(clean_memory)
                        mem_args = mem_data.get("arguments", {})
                        operation = mem_args.get("operation", "unknown")
                        memory_id = mem_args.get("memory_id", "")
                        description = mem_args.get("description", "")
                        content = mem_args.get("content", "")
                        memory_ids = mem_args.get("memory_ids", [])

                        memory_display += f'<div class="memory-call-block">'
                        memory_display += (
                            f'<div class="memory-header">🧠 {operation.upper()}</div>'
                        )

                        if memory_id:
                            memory_display += f'<div style="margin-top:5px;"><strong>ID:</strong> {memory_id}</div>'
                        if description:
                            memory_display += (
                                f"<div><strong>Desc:</strong> {description}</div>"
                            )
                        if content:
                            content_preview = (
                                content[:100] + "..." if len(content) > 100 else content
                            )
                            memory_display += f"<div><strong>Content:</strong> {content_preview}</div>"
                        if memory_ids:
                            memory_display += f"<div><strong>Check IDs:</strong> {', '.join(memory_ids)}</div>"

                        memory_display += "</div>"
                except:
                    memory_display = (
                        f'<div class="text-content">{memory_call_content}</div>'
                    )

                f.write(memory_display)
            else:
                f.write(
                    '<div class="text-content" style="color:#999;font-style:italic;">No memory operation</div>'
                )

            f.write("</td>\n")

            # Conclusion Column
            f.write("<td>\n")
            f.write('<div class="step-header">Conclusion</div>')
            f.write(f'<div class="text-content">{step["conclusion_content"]}</div>')

            # Add raw info toggle button at the bottom of conclusion column
            raw_response = step.get("raw_response", "")
            user_prompt = step.get("user_prompt", "")
            system_prompt = step.get("system_prompt", "")

            if raw_response or user_prompt:
                step_id = step["prefix"]
                f.write(f"""
                <div class="raw-info-toggle" onclick="openRawInfoModal({step_id})">
                    🔍 查看原始信息
                </div>
                """)

            f.write("</td>\n")

            f.write("</tr>\n")

        f.write(
            """
            </tbody>
        </table>
    </div>

    <div class="footer">
        <p>Android World visualization | """
            + task_name
            + """</p>
    </div>
    
    <!-- Modal for displaying raw info -->
    <div id="rawInfoModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">📋 原始信息详情 - Step <span id="modalStepNumber"></span></div>
                <button class="modal-close" onclick="closeRawInfoModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
                <!-- Content will be injected here -->
            </div>
        </div>
    </div>
    
    <script>
    // Store raw info data for each step
    const rawInfoData = {};
    </script>
    """
        )

        # Add raw info data to JavaScript for each step
        for step in steps_data:
            step_id = step["prefix"]
            raw_response = step.get("raw_response", "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            user_prompt = step.get("user_prompt", "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            system_prompt = step.get("system_prompt", "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
            
            f.write(f"""
    <script>
    rawInfoData[{step_id}] = {{
        userPrompt: `{user_prompt if user_prompt else "Not available"}`,
        modelResponse: `{raw_response if raw_response else "Not available"}`,
        systemPrompt: `{system_prompt if system_prompt else "Not available"}`
    }};
    </script>
    """)

        f.write("""
    <script>
    // Open modal with raw info
    function openRawInfoModal(stepId) {
        const modal = document.getElementById('rawInfoModal');
        const modalBody = document.getElementById('modalBody');
        const stepNumber = document.getElementById('modalStepNumber');
        
        const data = rawInfoData[stepId];
        if (!data) return;
        
        stepNumber.textContent = stepId;
        
        modalBody.innerHTML = `
            <div class="raw-section">
                <div class="raw-section-title">📤 User Prompt:</div>
                <div class="raw-section-content">${data.userPrompt}</div>
            </div>
            <div class="raw-section">
                <div class="raw-section-title">🤖 Model Response:</div>
                <div class="raw-section-content">${data.modelResponse}</div>
            </div>
            <div class="raw-section">
                <div class="raw-section-title">⚙️ System Prompt:</div>
                <div class="raw-section-content">${data.systemPrompt}</div>
            </div>
        `;
        
        modal.classList.add('visible');
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }
    
    // Close modal
    function closeRawInfoModal() {
        const modal = document.getElementById('rawInfoModal');
        modal.classList.remove('visible');
        document.body.style.overflow = ''; // Restore scrolling
    }
    
    // Close modal when clicking outside
    document.getElementById('rawInfoModal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeRawInfoModal();
        }
    });
    
    // Close modal with ESC key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeRawInfoModal();
        }
    });
    
    // Click dot positioning
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.image-container').forEach(container => {
            const wrapper = container.querySelector('.img-wrapper');
            const img = wrapper.querySelector('.step-image');
            const dot = wrapper.querySelector('.click-dot');
            if (!wrapper || !img || !dot) return;

            const getImageSize = () => ({
                originalWidth: img.naturalWidth,
                originalHeight: img.naturalHeight
            });

            img.onload = () => {
                const { originalWidth, originalHeight } = getImageSize();
                if (originalWidth === 0 || originalHeight === 0) {
                    console.error('fail to get image size');
                    return;
                }

                const displayWidth = img.clientWidth;
                const displayHeight = img.clientHeight;

                const imgOffsetLeft = img.offsetLeft;  
                const imgOffsetTop = img.offsetTop;    

                const originalX = parseFloat(dot.dataset.originalX);
                const originalY = parseFloat(dot.dataset.originalY);

                const widthRatio = displayWidth / originalWidth;
                const heightRatio = displayHeight / originalHeight;
                const scale = Math.min(widthRatio, heightRatio);

                dot.style.left = `${imgOffsetLeft + originalX * scale}px`;  
                dot.style.top = `${imgOffsetTop + originalY * scale}px`;    
            };

            img.onerror = () => {
                console.error('fail to show image');
            };
        });
    });
    </script>
</body>
</html>
"""
        )

    print(f"Created enhanced HTML report: {html_file}")
    return True


def create_html_reports(
    output_dir="extracted_data", experiment_name="experiment", summary_output_dir=None
):
    """Create HTML reports for all tasks and a summary dashboard"""
    print("\nCreating HTML reports with screenshots...")

    if summary_output_dir is None:
        summary_output_dir = output_dir

    # Find all task directories with screenshots
    task_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        screenshots_path = os.path.join(item_path, "screenshots")
        if os.path.isdir(item_path) and os.path.exists(screenshots_path):
            task_dirs.append(item)

    if not task_dirs:
        print("No tasks with screenshots found")
        return

    print(f"Creating reports for {len(task_dirs)} tasks")

    task_summaries = []

    for task_name in sorted(task_dirs):
        create_html_report(task_name, output_dir, experiment_name, summary_output_dir)

        # Collect summary info
        metadata = load_task_metadata(task_name, output_dir)
        if metadata:
            # Get steps count - try step_idx first, then step_number
            episode_data = metadata.get("episode_data", {})
            steps_list = episode_data.get("step_idx", [])
            if not steps_list:
                steps_list = episode_data.get("step_number", [])

            task_summaries.append(
                {
                    "name": task_name,
                    "goal": metadata.get("goal", ""),
                    "is_successful": bool(metadata.get("is_successful", 0)),
                    "run_time": metadata.get("run_time", 0),
                    "steps": len(steps_list),
                }
            )

    # Create Main Summary HTML (e.g., memgui-aw-25112601.html)
    summary_file = os.path.join(summary_output_dir, f"{experiment_name}.html")

    # Calculate link prefix
    link_prefix = ""
    if summary_output_dir != output_dir:
        rel_path = os.path.relpath(output_dir, summary_output_dir)
        link_prefix = f"{rel_path}/"

    # Calculate overall stats
    total_tasks = len(task_summaries)
    successful_tasks = sum(1 for t in task_summaries if t["is_successful"])
    success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

    # Calculate Memory Task Stats
    memory_success_rate = 0
    memory_total = 0
    memory_successful = 0

    memory_tasks = get_memory_tasks_set()
    if memory_tasks:
        print(f"Found {len(memory_tasks)} memory tasks in metadata for statistics")
        # Match tasks by extracting base name (removing _0, _1 suffix)
        memory_task_summaries = [
            t
            for t in task_summaries
            if extract_base_task_name(t["name"]) in memory_tasks
        ]
        memory_total = len(memory_task_summaries)
        memory_successful = sum(1 for t in memory_task_summaries if t["is_successful"])
        memory_success_rate = (
            (memory_successful / memory_total * 100) if memory_total > 0 else 0
        )
        print(
            f"Matched {memory_total} memory task instances: {[t['name'] for t in memory_task_summaries]}"
        )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Summary: {experiment_name}</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f8fa;
        }}
        
        header {{
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        h1 {{ margin: 0; }}
        
        .stats-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-around;
            text-align: center;
        }}
        
        .stat-item {{
            flex: 1;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        
        th {{
            background-color: #2c3e50;
            color: white;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .status-badge {{
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: bold;
        }}
        
        .status-success {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}
        
        .status-failure {{
            background-color: #ffebee;
            color: #c62828;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .memory-badge {{
            display: inline-block;
            background-color: #9c27b0;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-left: 8px;
            vertical-align: middle;
        }}
        
        .filter-controls {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .filter-btn {{
            padding: 8px 16px;
            border: 2px solid #9c27b0;
            background-color: white;
            color: #9c27b0;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }}
        
        .filter-btn:hover {{
            background-color: #f3e5f5;
        }}
        
        .filter-btn.active {{
            background-color: #9c27b0;
            color: white;
        }}
        
        .filter-label {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        tr.non-memory.filtered {{
            display: none;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Experiment Summary: {experiment_name}</h1>
    </header>
    
    <div class="stats-card">
        <div class="stat-item">
            <div class="stat-value">{total_tasks}</div>
            <div class="stat-label">Total Tasks</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{successful_tasks}</div>
            <div class="stat-label">Successful</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{success_rate:.1f}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{memory_success_rate:.1f}%</div>
            <div class="stat-label">Memory Task SR ({memory_successful}/{memory_total})</div>
        </div>
    </div>
    
    <div class="filter-controls">
        <span class="filter-label">筛选任务：</span>
        <button class="filter-btn" id="memoryFilterBtn" onclick="toggleMemoryFilter()">
            <span id="filterBtnText">只看 Memory 任务</span>
        </button>
        <span class="filter-label" id="filterStatus">显示全部 {total_tasks} 个任务</span>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Task Name</th>
                <th>Status</th>
                <th>Instruction / Goal</th>
                <th>Steps</th>
                <th>Time (s)</th>
            </tr>
        </thead>
        <tbody>
""")

        for task in task_summaries:
            status_class = (
                "status-success" if task["is_successful"] else "status-failure"
            )
            status_text = "Success" if task["is_successful"] else "Failure"

            # Check if it's a memory task (compare base name without suffix)
            memory_badge = ""
            base_task_name = extract_base_task_name(task["name"])
            is_memory = base_task_name in memory_tasks
            row_class = "memory-task" if is_memory else "non-memory"
            if is_memory:
                memory_badge = '<span class="memory-badge">Memory</span>'

            f.write(f"""
            <tr class="{row_class}">
                <td><a href="{link_prefix}{task["name"]}/report.html">{task["name"]}</a>{memory_badge}</td>
                <td><span class="status-badge {status_class}">{status_text}</span></td>
                <td>{task["goal"]}</td>
                <td>{task["steps"]}</td>
                <td>{task["run_time"]:.2f}</td>
            </tr>
""")

        f.write(f"""
        </tbody>
    </table>
    
    <script>
        let isFiltered = false;
        const memoryCount = {memory_total};
        const totalCount = {total_tasks};
        
        function toggleMemoryFilter() {{
            const btn = document.getElementById('memoryFilterBtn');
            const btnText = document.getElementById('filterBtnText');
            const filterStatus = document.getElementById('filterStatus');
            const rows = document.querySelectorAll('tr.non-memory');
            
            isFiltered = !isFiltered;
            
            if (isFiltered) {{
                // 筛选模式：只显示memory任务
                rows.forEach(row => row.classList.add('filtered'));
                btn.classList.add('active');
                btnText.textContent = '显示全部任务';
                filterStatus.textContent = `显示 ${{memoryCount}} 个 Memory 任务`;
            }} else {{
                // 显示全部任务
                rows.forEach(row => row.classList.remove('filtered'));
                btn.classList.remove('active');
                btnText.textContent = '只看 Memory 任务';
                filterStatus.textContent = `显示全部 ${{totalCount}} 个任务`;
            }}
        }}
    </script>
</body>
</html>
""")

    print(f"Created summary dashboard: {summary_file}")


def parse_pkl_gz_files(directory):
    """Parse all pkl.gz files in a directory"""
    print(f"\nParsing pkl.gz files in '{directory}'...")

    task_data = []
    task_files = [f for f in os.listdir(directory) if f.endswith(".pkl.gz")]

    if not task_files:
        print("No pkl.gz files found")
        return pd.DataFrame()

    print(f"Found {len(task_files)} pkl.gz files")

    # Get checkpoint directory name for column prefix
    checkpoint_name = os.path.basename(directory.rstrip("/"))

    for i, file_name in enumerate(task_files):
        if i % 10 == 0:
            print(f"Processing: {i + 1}/{len(task_files)} files...")

        file_path = os.path.join(directory, file_name)
        task_name = file_name.split("_")[0]  # Extract task name from filename

        try:
            data = _unzip_and_read_pickle(file_path)
            if data:
                # Process each task's data
                for episode in data:
                    episode_data = {
                        f"{checkpoint_name}_task_name": task_name,
                        f"{checkpoint_name}_is_successful": episode.get(
                            "is_successful", 0
                        ),
                        f"{checkpoint_name}_run_time": episode.get("run_time", 0),
                        f"{checkpoint_name}_goal": episode.get("goal", ""),
                        f"{checkpoint_name}_note": "",  # Empty note column
                    }
                    task_data.append(episode_data)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print(f"Successfully parsed {len(task_data)} task records")

    # Convert to DataFrame and sort by task_name
    if task_data:
        df = pd.DataFrame(task_data)
        # Sort by task_name column (using the prefixed column name)
        task_name_col = f"{checkpoint_name}_task_name"
        df = df.sort_values(by=task_name_col, ascending=True)
        return df
    else:
        return pd.DataFrame()


def analyze_results(results_df):
    """Analyze result data and print summary information"""
    if results_df.empty:
        print("No result data found")
        return

    print(
        f"\nResult data contains {len(results_df)} rows with columns: {', '.join(results_df.columns)}"
    )

    # Find column names with prefixes
    task_name_col = None
    is_successful_col = None
    run_time_col = None

    for col in results_df.columns:
        if col.endswith("_task_name"):
            task_name_col = col
        elif col.endswith("_is_successful"):
            is_successful_col = col
        elif col.endswith("_run_time"):
            run_time_col = col

    # Basic statistics
    total_tasks = len(results_df)
    if is_successful_col:
        successful_tasks = results_df[is_successful_col].sum()
        success_rate = successful_tasks / total_tasks * 100 if total_tasks > 0 else 0
        print(f"Total tasks: {total_tasks}")
        print(f"Successful tasks: {successful_tasks}")
        print(f"Success rate: {success_rate:.2f}%")

    # Execution time statistics
    if run_time_col:
        avg_time = results_df[run_time_col].mean()
        max_time = results_df[run_time_col].max()
        min_time = results_df[run_time_col].min()
        print(f"Average execution time: {avg_time:.2f} seconds")
        print(f"Maximum execution time: {max_time:.2f} seconds")
        print(f"Minimum execution time: {min_time:.2f} seconds")

    # Task type distribution
    if task_name_col:
        task_counts = Counter(results_df[task_name_col])
        print("\nTask type distribution:")
        for task, count in task_counts.most_common():
            print(f"  {task}: {count} times")

        # Success rate by task
        if is_successful_col:
            print("\nSuccess rate by task:")
            task_success = (
                results_df.groupby(task_name_col)[is_successful_col].mean() * 100
            )
            for task, rate in task_success.sort_values(ascending=False).items():
                print(f"  {task}: {rate:.2f}%")


def _find_run_results_json(checkpoint_dir):
    """Find the JSON file written by run.py's ResultSaver, if available."""
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip("/"))
    preferred = os.path.join(checkpoint_dir, f"{checkpoint_name}_results.json")
    if os.path.exists(preferred):
        return preferred

    if not os.path.exists(checkpoint_dir):
        return None

    candidates = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.endswith("_results.json")
    ]
    return sorted(candidates)[0] if candidates else None


def _parse_instance_from_checkpoint_name(base_name):
    """Return task template and instance id from names like TaskName_2."""
    match = re.match(r"^(?P<task>.+)_(?P<instance>\d+)$", base_name)
    if match:
        return match.group("task"), int(match.group("instance"))
    return base_name, None


def _success_value(value):
    """Convert success-like values to 0.0/1.0 compatible with run.py."""
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    try:
        return 1.0 if float(value) > 0.5 else 0.0
    except Exception:
        return 1.0 if bool(value) else 0.0


def _json_safe(value):
    """Convert pandas/numpy values into JSON-safe Python primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _load_run_json_records(checkpoint_dir):
    """Load task records and stored summaries from run.py's JSON result file."""
    json_path = _find_run_results_json(checkpoint_dir)
    if not json_path:
        return pd.DataFrame(), {}, None

    try:
        data = _load_json_file(json_path)
    except Exception as e:
        print(f"Failed to load run.py JSON result {json_path}: {e}")
        return pd.DataFrame(), {}, None

    records = data.get("tasks", [])
    if not isinstance(records, list):
        records = []

    df = pd.DataFrame(records)
    if not df.empty:
        if TASK_TEMPLATE_COLUMN not in df.columns and "task_name" in df.columns:
            df[TASK_TEMPLATE_COLUMN] = df["task_name"].apply(extract_base_task_name)
        if "task_name" not in df.columns and TASK_TEMPLATE_COLUMN in df.columns:
            df["task_name"] = df[TASK_TEMPLATE_COLUMN]
        if "is_successful" in df.columns:
            df["is_successful"] = df["is_successful"].apply(_success_value)
        if "tags" in df.columns:
            df["tags"] = df["tags"].apply(normalize_tags)

    stored_summary = {
        "summary": data.get("summary", {}),
        "pass_at_n": data.get("pass_at_n", {}),
        "by_difficulty": data.get("by_difficulty", {}),
        "by_tag": data.get("by_tag", {}),
    }
    return df, stored_summary, json_path


def parse_checkpoint_records_for_summary(checkpoint_dir):
    """Parse checkpoint pkl.gz files into unprefixed episode records."""
    task_files = sorted(
        name for name in os.listdir(checkpoint_dir) if name.endswith(".pkl.gz")
    )
    records = []

    for i, file_name in enumerate(task_files):
        if i % 25 == 0:
            print(f"Summary parsing checkpoints: {i + 1}/{len(task_files)} files...")

        file_path = os.path.join(checkpoint_dir, file_name)
        base_name = file_name.replace(".pkl.gz", "")
        task_from_file, instance_from_file = _parse_instance_from_checkpoint_name(
            base_name
        )

        try:
            data = _unzip_and_read_pickle(file_path)
            if not data:
                continue
            for episode in data:
                task_template = episode.get(TASK_TEMPLATE_COLUMN) or task_from_file
                instance_id = episode.get("instance_id", instance_from_file)
                records.append(
                    {
                        TASK_TEMPLATE_COLUMN: task_template,
                        "task_name": task_template,
                        "task_instance_name": base_name,
                        "instance_id": instance_id,
                        "is_successful": _success_value(episode.get("is_successful", 0)),
                        "run_time": episode.get("run_time", 0),
                        "episode_length": episode.get("episode_length", None),
                        "goal": episode.get("goal", ""),
                        "agent_name": episode.get("agent_name", ""),
                        "exception_info": episode.get("exception_info"),
                        "difficulty": episode.get("difficulty"),
                        "tags": episode.get("tags"),
                    }
                )
        except Exception as e:
            print(f"Error parsing summary data from {file_name}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = enrich_summary_records_with_metadata(df)
    return df


def enrich_summary_records_with_metadata(df):
    """Fill difficulty/tags/task metadata for summary calculations."""
    metadata_df = get_metadata_df()
    if metadata_df is None or df.empty or TASK_TEMPLATE_COLUMN not in df.columns:
        return df

    try:
        metadata_cols = [
            "task_name",
            "task_template",
            "difficulty",
            "tags",
            "optimal_steps",
            "memory-task",
        ]
        metadata_cols = [c for c in metadata_cols if c in metadata_df.columns]
        metadata_subset = metadata_df[metadata_cols].copy()

        merged = df.copy()
        if "task_name" not in merged.columns:
            merged["task_name"] = merged[TASK_TEMPLATE_COLUMN]

        merged = merged.merge(
            metadata_subset,
            left_on=TASK_TEMPLATE_COLUMN,
            right_on="task_name",
            how="left",
            suffixes=("", "_metadata"),
        )

        for column in [
            "difficulty",
            "optimal_steps",
            "tags",
            TASK_PROMPT_COLUMN,
            "memory-task",
        ]:
            metadata_column = f"{column}_metadata"
            if metadata_column not in merged.columns:
                continue
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[metadata_column])
            else:
                merged[column] = merged[metadata_column]
            merged = merged.drop(columns=[metadata_column])

        if "task_name_metadata" in merged.columns:
            merged = merged.drop(columns=["task_name_metadata"])
        if "task_template_metadata" in merged.columns:
            merged = merged.drop(columns=["task_template_metadata"])
        if "tags" in merged.columns:
            merged["tags"] = merged["tags"].apply(normalize_tags)
        return merged
    except Exception as e:
        print(f"Error enriching summary records with metadata: {e}")
        return df


def infer_n_task_combinations(df, stored_pass_at_n=None):
    """Infer n_task_combinations from stored pass@k or observed instances."""
    if stored_pass_at_n:
        ks = []
        for key in stored_pass_at_n:
            match = re.match(r"pass@(\d+)$", key)
            if match:
                ks.append(int(match.group(1)))
        if ks:
            return max(ks)

    if df.empty or TASK_TEMPLATE_COLUMN not in df.columns:
        return 1
    return int(df.groupby(TASK_TEMPLATE_COLUMN).size().max())


def compute_run_summary_tables(df, n_task_combinations=None, stored_summary=None):
    """Compute the same tables printed by run.py's realtime summary."""
    stored_summary = stored_summary or {}
    if df.empty:
        return {
            "summary": {},
            "pass_at_n": {},
            "attempt_pass_at_1": [],
            "by_difficulty": {},
            "by_tag": {},
            "per_task": [],
        }

    df = df.copy()
    if "is_successful" not in df.columns:
        df["is_successful"] = 0.0
    df["is_successful"] = df["is_successful"].apply(_success_value)
    if "instance_id" in df.columns:
        df["instance_id"] = pd.to_numeric(df["instance_id"], errors="coerce")

    total_tasks = len(df)
    num_successful = int((df["is_successful"] > 0.5).sum())
    num_failed = int(total_tasks - num_successful)
    success_rate = num_successful / total_tasks if total_tasks else 0
    summary = {
        "total_tasks": total_tasks,
        "num_successful": num_successful,
        "num_failed": num_failed,
        "success_rate": round(success_rate, 4),
    }

    if stored_summary.get("summary"):
        summary = stored_summary["summary"]

    if n_task_combinations is None:
        n_task_combinations = infer_n_task_combinations(
            df, stored_summary.get("pass_at_n")
        )

    pass_at_n = {}
    if n_task_combinations > 1 and TASK_TEMPLATE_COLUMN in df.columns:
        grouped = df.groupby(TASK_TEMPLATE_COLUMN, sort=True)
        num_task_templates = len(grouped)

        for n in range(1, n_task_combinations + 1):
            pass_count = 0
            for _, group in grouped:
                if "instance_id" in group.columns:
                    instances = group.sort_values("instance_id")
                else:
                    instances = group
                if (instances.head(n)["is_successful"] > 0.5).any():
                    pass_count += 1

            pass_rate = pass_count / num_task_templates if num_task_templates else 0
            pass_at_n[f"pass@{n}"] = {
                "pass_count": int(pass_count),
                "total_tasks": int(num_task_templates),
                "pass_rate": round(pass_rate, 4),
            }

        if "instance_id" in df.columns:
            for n in range(1, n_task_combinations + 1):
                within_n_success = 0
                for _, group in grouped:
                    instances = group.sort_values("instance_id")
                    within_n_success += int(
                        (instances.head(n)["is_successful"] > 0.5).sum()
                    )

                total_attempts = num_task_templates * n
                within_n_rate = (
                    within_n_success / total_attempts if total_attempts else 0
                )
                pass_at_n[f"within_{n}_success"] = {
                    "total_successes": int(within_n_success),
                    "total_attempts": int(total_attempts),
                    "success_rate": round(within_n_rate, 4),
                }

    if stored_summary.get("pass_at_n"):
        pass_at_n = stored_summary["pass_at_n"]

    attempt_pass_at_1 = []
    if "instance_id" in df.columns:
        attempt_rates = []
        for attempt_id, group in df.dropna(subset=["instance_id"]).groupby(
            "instance_id", sort=True
        ):
            total = len(group)
            successful = int((group["is_successful"] > 0.5).sum())
            pass_rate = successful / total if total else 0
            attempt_rates.append(pass_rate)
            if float(attempt_id).is_integer():
                attempt_name = int(attempt_id)
            else:
                attempt_name = attempt_id
            attempt_pass_at_1.append(
                {
                    "attempt_id": attempt_name,
                    "pass_count": successful,
                    "total_tasks": int(total),
                    "pass_rate": round(pass_rate, 4),
                }
            )

        if attempt_rates:
            attempt_pass_at_1.extend(
                [
                    {
                        "attempt_id": "mean",
                        "pass_count": "",
                        "total_tasks": "",
                        "pass_rate": round(sum(attempt_rates) / len(attempt_rates), 4),
                    },
                    {
                        "attempt_id": "min",
                        "pass_count": "",
                        "total_tasks": "",
                        "pass_rate": round(min(attempt_rates), 4),
                    },
                    {
                        "attempt_id": "max",
                        "pass_count": "",
                        "total_tasks": "",
                        "pass_rate": round(max(attempt_rates), 4),
                    },
                ]
            )

    by_difficulty = {}
    if "difficulty" in df.columns:
        difficulty_values = ["easy", "medium", "hard"]
        extra_values = [
            value
            for value in sorted(df["difficulty"].dropna().astype(str).unique())
            if value not in difficulty_values
        ]
        for difficulty in difficulty_values + extra_values:
            diff_df = df[df["difficulty"].astype(str) == difficulty]
            if diff_df.empty:
                continue
            total = len(diff_df)
            successful = int((diff_df["is_successful"] > 0.5).sum())
            failed = int(total - successful)
            by_difficulty[str(difficulty)] = {
                "total": int(total),
                "successful": successful,
                "failed": failed,
                "success_rate": round(successful / total if total else 0, 4),
            }

    if stored_summary.get("by_difficulty"):
        by_difficulty = stored_summary["by_difficulty"]

    by_tag = {}
    if "tags" in df.columns:
        tag_rows = []
        for _, row in df.iterrows():
            for tag in normalize_tags(row.get("tags")):
                row_dict = row.to_dict()
                row_dict["tag"] = tag
                tag_rows.append(row_dict)
        if tag_rows:
            tag_df = pd.DataFrame(tag_rows)
            for tag in sorted(tag_df["tag"].dropna().astype(str).unique()):
                current = tag_df[tag_df["tag"].astype(str) == tag]
                total = len(current)
                successful = int((current["is_successful"] > 0.5).sum())
                failed = int(total - successful)
                by_tag[str(tag)] = {
                    "total": int(total),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": round(successful / total if total else 0, 4),
                }

    if stored_summary.get("by_tag"):
        by_tag = stored_summary["by_tag"]

    per_task = []
    if TASK_TEMPLATE_COLUMN in df.columns:
        grouped = df.groupby(TASK_TEMPLATE_COLUMN, sort=True)
        for idx, (task_name, group) in enumerate(grouped, start=1):
            total = len(group)
            successful = int((group["is_successful"] > 0.5).sum())
            rate = successful / total if total else 0
            row = {
                "no": idx,
                "task_name": task_name,
                "successful": successful,
                "trials": int(total),
                "success_rate": round(rate, 4),
            }
            if "difficulty" in group.columns:
                difficulty = group["difficulty"].dropna()
                row["difficulty"] = difficulty.iloc[0] if not difficulty.empty else None
            if "tags" in group.columns:
                tags = []
                for value in group["tags"].dropna():
                    tags.extend(normalize_tags(value))
                row["tags"] = sorted(set(tags))
            per_task.append(row)

    return {
        "summary": summary,
        "pass_at_n": pass_at_n,
        "attempt_pass_at_1": attempt_pass_at_1,
        "by_difficulty": by_difficulty,
        "by_tag": by_tag,
        "per_task": per_task,
    }


def _dict_table_to_rows(table, name_field):
    rows = []
    for name, values in table.items():
        row = {name_field: name}
        if isinstance(values, dict):
            row.update(values)
        else:
            row["value"] = values
        rows.append(row)
    return rows


def _pass_at_n_to_rows(pass_at_n):
    rows = []
    for metric, values in pass_at_n.items():
        row = {"metric": metric}
        match = re.search(r"(\d+)", metric)
        if match:
            row["k"] = int(match.group(1))
        if isinstance(values, dict):
            row.update(values)
        rows.append(row)
    return rows


def save_run_summary_outputs(
    checkpoint_dir,
    output_dir,
    n_task_combinations=None,
    prefer_run_json=True,
):
    """Save run.py-style Pass@k, difficulty, tag, and per-task summaries."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip("/"))

    df = pd.DataFrame()
    stored_summary = {}
    source = "pkl.gz"
    source_json = None

    if prefer_run_json:
        df, stored_summary, source_json = _load_run_json_records(checkpoint_dir)
        if not df.empty:
            source = "run_json"

    if df.empty:
        df = parse_checkpoint_records_for_summary(checkpoint_dir)
    else:
        df = enrich_summary_records_with_metadata(df)

    if df.empty:
        print(f"No records available for run summary: {checkpoint_dir}")
        return None

    tables = compute_run_summary_tables(
        df,
        n_task_combinations=n_task_combinations,
        stored_summary=stored_summary,
    )

    payload = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": checkpoint_dir,
        "source": source,
        "source_json": source_json,
        "generated_at": datetime.now().isoformat(),
        **tables,
    }
    payload = _json_safe(payload)

    json_path = os.path.join(output_dir, f"{checkpoint_name}_run_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    pass_csv = os.path.join(output_dir, f"{checkpoint_name}_pass_at_k.csv")
    attempt_pass_csv = os.path.join(
        output_dir, f"{checkpoint_name}_attempt_pass_at_1.csv"
    )
    difficulty_csv = os.path.join(output_dir, f"{checkpoint_name}_by_difficulty.csv")
    tag_csv = os.path.join(output_dir, f"{checkpoint_name}_by_tag.csv")
    per_task_csv = os.path.join(output_dir, f"{checkpoint_name}_per_task.csv")

    pd.DataFrame(_pass_at_n_to_rows(tables["pass_at_n"])).to_csv(pass_csv, index=False)
    pd.DataFrame(tables["attempt_pass_at_1"]).to_csv(
        attempt_pass_csv, index=False
    )
    pd.DataFrame(_dict_table_to_rows(tables["by_difficulty"], "difficulty")).to_csv(
        difficulty_csv, index=False
    )
    pd.DataFrame(_dict_table_to_rows(tables["by_tag"], "tag")).to_csv(
        tag_csv, index=False
    )
    pd.DataFrame(tables["per_task"]).to_csv(per_task_csv, index=False)

    print(f"\nRun.py-style summary saved for {checkpoint_name}:")
    print(f"  JSON:          {json_path}")
    print(f"  Pass@k CSV:    {pass_csv}")
    print(f"  Attempt Pass@1 CSV:{attempt_pass_csv}")
    print(f"  Difficulty CSV:{difficulty_csv}")
    print(f"  Tag CSV:       {tag_csv}")
    print(f"  Per-task CSV:  {per_task_csv}")

    if tables["pass_at_n"]:
        print("  Pass@k:")
        for row in _pass_at_n_to_rows(tables["pass_at_n"]):
            if "pass_count" in row:
                print(
                    f"    {row['metric']}: {row['pass_count']}/{row['total_tasks']} "
                    f"({row['pass_rate']:.4f})"
                )

    return {
        "json": json_path,
        "pass_at_k_csv": pass_csv,
        "attempt_pass_at_1_csv": attempt_pass_csv,
        "by_difficulty_csv": difficulty_csv,
        "by_tag_csv": tag_csv,
        "per_task_csv": per_task_csv,
        "tables": tables,
    }


def visualize_results(results_df, output_dir=None):
    """Visualize result data"""
    if results_df.empty:
        print("No result data found for visualization")
        return
    import matplotlib.pyplot as plt

    # Create output directory
    if output_dir is None:
        output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Success rate pie chart
    if "success" in results_df.columns:
        plt.figure(figsize=(8, 6))
        success_count = results_df["success"].sum()
        fail_count = len(results_df) - success_count
        plt.pie(
            [success_count, fail_count],
            labels=["Success", "Failure"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
        )
        plt.title("Task Success Rate")
        plt.savefig(os.path.join(output_dir, "success_rate_pie.png"))
        plt.close()
        print(
            f"Success rate pie chart saved to: {os.path.join(output_dir, 'success_rate_pie.png')}"
        )

    # Execution time bar chart
    if "execution_time" in results_df.columns and "task_name" in results_df.columns:
        plt.figure(figsize=(15, 10))
        task_times = (
            results_df.groupby("task_name")["execution_time"]
            .mean()
            .sort_values(ascending=False)
        )
        task_times.plot(kind="bar")
        plt.title("Average Execution Time by Task")
        plt.ylabel("Execution Time (seconds)")
        plt.xlabel("Task Name")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_execution_times.png"))
        plt.close()
        print(
            f"Task execution time chart saved to: {os.path.join(output_dir, 'task_execution_times.png')}"
        )

    # Task success rate bar chart
    if "success" in results_df.columns and "task_name" in results_df.columns:
        plt.figure(figsize=(15, 10))
        task_success = results_df.groupby("task_name")["success"].mean() * 100
        task_success.sort_values(ascending=False).plot(kind="bar")
        plt.title("Success Rate by Task")
        plt.ylabel("Success Rate (%)")
        plt.xlabel("Task Name")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_success_rates.png"))
        plt.close()
        print(
            f"Task success rate chart saved to: {os.path.join(output_dir, 'task_success_rates.png')}"
        )

    # Steps bar chart
    if "steps" in results_df.columns and "task_name" in results_df.columns:
        plt.figure(figsize=(15, 10))
        task_steps = (
            results_df.groupby("task_name")["steps"].mean().sort_values(ascending=False)
        )
        task_steps.plot(kind="bar")
        plt.title("Average Steps by Task")
        plt.ylabel("Steps")
        plt.xlabel("Task Name")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_steps.png"))
        plt.close()
        print(
            f"Task steps chart saved to: {os.path.join(output_dir, 'task_steps.png')}"
        )


def extract_task_goals(results_df):
    """Extract and display goals for each task"""
    # Find column names with prefixes
    task_name_col = None
    goal_col = None

    for col in results_df.columns:
        if col.endswith("_task_name"):
            task_name_col = col
        elif col.endswith("_goal"):
            goal_col = col

    if goal_col and task_name_col:
        print("\nTask goals examples:")
        for task_name in sorted(results_df[task_name_col].unique()):
            goals = results_df[results_df[task_name_col] == task_name][
                goal_col
            ].unique()
            if goals.size > 0:
                print(f"\nTask: {task_name}")
                print(f"Goal: {goals[0]}")


def analyze_checkpoint_data(checkpoint_dir):
    """Use the project's built-in checkpointer to parse checkpoint directory"""
    print(f"Parsing checkpoint directory: {checkpoint_dir}")

    # Use the project's IncrementalCheckpointer to load data
    loader = checkpointer.IncrementalCheckpointer(checkpoint_dir)
    episodes = loader.load()

    if not episodes:
        print("No task data found")
        return None

    print(f"Successfully loaded {len(episodes)} task records")

    # Convert to DataFrame for analysis
    task_data = []
    for episode in episodes:
        episode_data = {
            "task_name": episode.get("task_name", "unknown"),
            "success": episode.get("success", False),
            "execution_time": episode.get("execution_time", 0),
            "steps": len(episode.get("steps", [])),
            "goal": episode.get("goal", ""),
            "reward": episode.get("reward", 0),
        }
        task_data.append(episode_data)

    return pd.DataFrame(task_data)


def is_directory_already_parsed(checkpoint_dir, vis_output_dir):
    """Check if a directory has already been parsed by looking for key output files"""
    # 检查是否存在可视化输出目录
    if not os.path.exists(vis_output_dir):
        return False
    
    # 检查是否有任务子目录（解析后的数据）
    task_dirs = [d for d in os.listdir(vis_output_dir) 
                 if os.path.isdir(os.path.join(vis_output_dir, d))]
    
    if not task_dirs:
        return False
    
    # 检查源目录中的pkl.gz文件数量
    if not os.path.exists(checkpoint_dir):
        return False
    
    pkl_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl.gz")]
    
    # 如果解析后的任务目录数量与pkl文件数量匹配，认为已经解析过
    if len(task_dirs) >= len(pkl_files) and len(pkl_files) > 0:
        print(f"✓ 目录已解析过: 发现 {len(task_dirs)} 个任务目录，源目录有 {len(pkl_files)} 个pkl.gz文件")
        return True
    
    return False


def process_multiple_checkpoints(
    checkpoint_dirs,
    output_combined_csv="combined_results.csv",
    summary_only=False,
    n_task_combinations=None,
    prefer_run_json=True,
    summary_output_dir=None,
):
    """Process multiple checkpoint directories and combine results"""
    all_dataframes = []
    individual_csvs = []

    for checkpoint_dir in checkpoint_dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing checkpoint directory: {checkpoint_dir}")
        print(f"{'=' * 60}")

        # Check if directory exists
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Directory '{checkpoint_dir}' does not exist!")
            parent_dir = os.path.dirname(checkpoint_dir)
            if os.path.exists(parent_dir):
                print(f"Parent directory '{parent_dir}' exists, listing contents:")
                list_directory_contents(parent_dir)

                # Find possible run directories
                run_dirs = [d for d in os.listdir(parent_dir) if d.startswith("run_")]
                if run_dirs:
                    print("\nFound the following possible run directories:")
                    for d in sorted(run_dirs):
                        print(f"  {d}")
                    latest_run = sorted(run_dirs)[-1]
                    checkpoint_dir = os.path.join(parent_dir, latest_run)
                    print(f"\nUsing latest run directory: {checkpoint_dir}")
            else:
                print(f"Skipping {checkpoint_dir} - directory not found")
                continue

        # Determine output directory for visualization
        checkpoint_name = os.path.basename(checkpoint_dir.rstrip("/"))
        vis_base_dir = summary_output_dir or os.path.join("results", "00_vis")
        vis_output_dir = os.path.join(vis_base_dir, f"{checkpoint_name}-vis")
        os.makedirs(vis_base_dir, exist_ok=True)

        summary_outputs = save_run_summary_outputs(
            checkpoint_dir=checkpoint_dir,
            output_dir=vis_base_dir,
            n_task_combinations=n_task_combinations,
            prefer_run_json=prefer_run_json,
        )

        if summary_only:
            if summary_outputs:
                per_task_csv = summary_outputs["per_task_csv"]
                if os.path.exists(per_task_csv):
                    individual_csvs.append(per_task_csv)
                    per_task_df = pd.read_csv(per_task_csv)
                    if not per_task_df.empty:
                        prefixed_df = per_task_df.rename(
                            columns={
                                "task_name": f"{checkpoint_name}_task_name",
                                "successful": f"{checkpoint_name}_successful",
                                "trials": f"{checkpoint_name}_trials",
                                "success_rate": f"{checkpoint_name}_success_rate",
                            }
                        )
                        all_dataframes.append(prefixed_df)
            continue

        # Check if already parsed
        if is_directory_already_parsed(checkpoint_dir, vis_output_dir):
            print(f"⏭️  跳过已解析的目录: {checkpoint_dir}")
            print(f"   输出目录: {vis_output_dir}")
            
            # Still need to load the data for combining
            results_df = parse_pkl_gz_files(checkpoint_dir)
            if not results_df.empty:
                # Enrich with metadata
                task_name_col = f"{checkpoint_name}_task_name"
                results_df = enrich_df_with_metadata(results_df, task_name_col)
                
                # Check if individual CSV exists
                individual_csv_path = os.path.join(vis_base_dir, f"{checkpoint_name}.csv")
                if os.path.exists(individual_csv_path):
                    print(f"   已存在CSV文件: {individual_csv_path}")
                    individual_csvs.append(individual_csv_path)
                
                all_dataframes.append(results_df)
            
            continue

        # List directory contents
        list_directory_contents(checkpoint_dir)

        # Create directories if they don't exist
        os.makedirs(vis_output_dir, exist_ok=True)

        print(f"\nVisualization output directory: {vis_output_dir}")

        # Extract all pkl.gz files to readable JSON format and extract screenshots
        # We extract directly to the visualization directory
        extract_all_pkl_gz_files(checkpoint_dir, vis_output_dir)
        print(f"\nExtracted data saved to '{vis_output_dir}' directory")

        # Create HTML reports with screenshots
        # Pass vis_base_dir as summary_output_dir to save the main HTML in results/00_vis/
        create_html_reports(
            vis_output_dir,
            experiment_name=checkpoint_name,
            summary_output_dir=vis_base_dir,
        )
        print(f"\nHTML reports created in '{vis_output_dir}' directory")

        # Parse pkl.gz files for analysis
        results_df = parse_pkl_gz_files(checkpoint_dir)

        if not results_df.empty:
            # Analyze and print results
            analyze_results(results_df)

            # Extract task goals
            extract_task_goals(results_df)

            # Visualize results
            visualize_results(results_df, vis_output_dir)

            # Enrich with metadata before saving individual CSV
            task_name_col = f"{checkpoint_name}_task_name"
            results_df = enrich_df_with_metadata(results_df, task_name_col)

            # Save individual CSV
            individual_csv_name = f"{checkpoint_name}.csv"
            individual_csv_path = os.path.join(vis_base_dir, individual_csv_name)
            results_df.to_csv(individual_csv_path, index=False)
            print(f"Individual CSV saved to: {individual_csv_path}")
            individual_csvs.append(individual_csv_path)

            # Store for combining
            all_dataframes.append(results_df)
        else:
            print(f"\nNo result data found for {checkpoint_dir}")

    # Combine all results
    if all_dataframes:
        print(f"\n{'=' * 60}")
        print("Combining results from all checkpoint directories")
        print(f"{'=' * 60}")

        combined_df = combine_checkpoint_results(all_dataframes)

        # Save combined results
        combined_df.to_csv(output_combined_csv, index=False)
        print(f"\nCombined results saved to: {output_combined_csv}")

        # Print summary
        print(f"\nSummary:")
        print(f"  - Processed {len(checkpoint_dirs)} checkpoint directories")
        print(f"  - Generated {len(individual_csvs)} individual CSV files")
        print(
            f"  - Combined CSV contains {len(combined_df)} rows and {len(combined_df.columns)} columns"
        )
        print(f"  - Individual CSV files: {individual_csvs}")
        print(f"  - Combined CSV file: {output_combined_csv}")
    else:
        print("\nNo data to combine!")


def combine_checkpoint_results(dataframes):
    """Combine multiple checkpoint DataFrames ensuring same tasks are in same rows"""
    if not dataframes:
        return pd.DataFrame()

    # Get all unique task names from all dataframes
    all_task_names = set()
    checkpoint_prefixes = []

    for df in dataframes:
        # Find task_name column for this dataframe
        task_name_col = None
        for col in df.columns:
            if col.endswith("_task_name"):
                task_name_col = col
                # Extract checkpoint prefix
                prefix = col.replace("_task_name", "")
                checkpoint_prefixes.append(prefix)
                break

        if task_name_col:
            all_task_names.update(df[task_name_col].unique())

    # Sort task names for consistent ordering
    sorted_task_names = sorted(all_task_names)

    print(f"Found {len(sorted_task_names)} unique task names across all checkpoints")
    print(f"Checkpoint prefixes: {checkpoint_prefixes}")

    # Create a base DataFrame with all task names
    combined_data = []

    for task_name in sorted_task_names:
        row_data = {"task_name": task_name}

        # For each checkpoint, find data for this task
        for i, df in enumerate(dataframes):
            prefix = checkpoint_prefixes[i]
            task_name_col = f"{prefix}_task_name"

            # Find the row for this task in current dataframe
            task_rows = df[df[task_name_col] == task_name]

            if not task_rows.empty:
                # Use the first occurrence if multiple exist
                task_row = task_rows.iloc[0]
                # Add all columns from this checkpoint
                for col in df.columns:
                    if col.endswith("_task_name"):
                        continue  # Skip task_name columns except the main one
                    row_data[col] = task_row[col]
            else:
                # Add empty values for missing tasks
                for col in df.columns:
                    if col.endswith("_task_name"):
                        continue
                    elif col.endswith("_is_successful"):
                        row_data[col] = 0
                    elif col.endswith("_run_time"):
                        row_data[col] = 0
                    elif col.endswith("_goal"):
                        row_data[col] = ""
                    elif col.endswith("_note"):
                        row_data[col] = ""
                    else:
                        row_data[col] = ""

        combined_data.append(row_data)

    # Create DataFrame and sort by task_name
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values(by="task_name", ascending=True)

    # Reorder columns: task_name first, then grouped by checkpoint
    columns_order = ["task_name"]
    for prefix in checkpoint_prefixes:
        prefix_cols = [
            col for col in combined_df.columns if col.startswith(prefix + "_")
        ]
        columns_order.extend(sorted(prefix_cols))

    # Add any remaining columns
    remaining_cols = [col for col in combined_df.columns if col not in columns_order]
    columns_order.extend(remaining_cols)

    combined_df = combined_df[columns_order]

    # Load metadata and merge
    metadata_df = get_metadata_df()
    if metadata_df is not None:
        try:
            print(
                f"Merging metadata from docs/androidworld-filter-memory - task_metadata_filled.csv"
            )
            # Select required columns
            cols_to_merge = [
                "task_name",
                "task_template",
                "difficulty",
                "tags",
                "optimal_steps",
                "memory-task",
            ]
            # Ensure columns exist
            cols_to_merge = [c for c in cols_to_merge if c in metadata_df.columns]

            metadata_subset = metadata_df[cols_to_merge].copy()

            # Drop existing metadata columns from combined_df to avoid duplicates (_x, _y)
            cols_to_drop = [
                c
                for c in cols_to_merge
                if c in combined_df.columns and c != "task_name"
            ]
            if cols_to_drop:
                combined_df = combined_df.drop(columns=cols_to_drop)

            # Merge
            combined_df = pd.merge(
                combined_df, metadata_subset, on="task_name", how="left"
            )

            # Reorder columns to put metadata after task_name
            cols = list(combined_df.columns)
            # task_name is first.
            # We want metadata columns next.
            metadata_cols = [c for c in cols_to_merge if c != "task_name"]

            # Only include metadata columns that are actually in combined_df
            metadata_cols = [c for c in metadata_cols if c in combined_df.columns]

            other_cols = [
                c for c in cols if c not in metadata_cols and c != "task_name"
            ]

            new_order = ["task_name"] + metadata_cols + other_cols
            combined_df = combined_df[new_order]

        except Exception as e:
            print(f"Error merging metadata: {e}")

    return combined_df


if __name__ == "__main__":
    # Generate default output filename with timestamp
    timestamp = datetime.now().strftime("%y%m%d%H")
    default_output = f"aw_combined_results_{timestamp}.csv"
    
    parser = argparse.ArgumentParser(
        description="Parse checkpoint directories and generate reports."
    )
    parser.add_argument(
        "dirs", nargs="*", help="List of checkpoint directories to process"
    )
    parser.add_argument(
        "--dir", help="Comma-separated list of checkpoint directories to process"
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output file for combined CSV results (default: {default_output})",
    )
    parser.add_argument(
        "--summary-output-dir",
        default=None,
        help=(
            "Directory for run.py-style summary outputs. Defaults to results/00_vis."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Only save run.py-style summary CSV/JSON files "
            "(Pass@k, by difficulty, by tag, per task)."
        ),
    )
    parser.add_argument(
        "--n-task-combinations",
        type=int,
        default=None,
        help="Override n_task_combinations used for Pass@k. By default it is inferred.",
    )
    parser.add_argument(
        "--force-pkl-summary",
        action="store_true",
        help=(
            "Ignore run.py's *_results.json when generating summary outputs "
            "and recompute from .pkl.gz checkpoints."
        ),
    )

    args = parser.parse_args()

    checkpoint_dirs = []

    # Handle positional arguments
    if args.dirs:
        checkpoint_dirs.extend(args.dirs)

    # Handle --dir argument
    if args.dir:
        # Split by comma and strip whitespace
        dirs_from_arg = [d.strip() for d in args.dir.split(",") if d.strip()]
        checkpoint_dirs.extend(dirs_from_arg)

    # Remove duplicates while preserving order
    seen = set()
    checkpoint_dirs = [x for x in checkpoint_dirs if not (x in seen or seen.add(x))]

    if not checkpoint_dirs:
        # Default fallback
        checkpoint_dirs = ["results/memgui-aw-25112706-debug"]
        print(
            f"No directories specified via command line. Using default: {checkpoint_dirs}"
        )

    # Output file for combined results
    output_combined_csv = args.output

    # If the output path doesn't have a directory component, save it to results/00_vis/
    if not os.path.dirname(output_combined_csv):
        vis_base_dir = args.summary_output_dir or os.path.join("results", "00_vis")
        os.makedirs(vis_base_dir, exist_ok=True)
        output_combined_csv = os.path.join(vis_base_dir, output_combined_csv)
    else:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_combined_csv), exist_ok=True)

    # Process all checkpoint directories
    process_multiple_checkpoints(
        checkpoint_dirs,
        output_combined_csv,
        summary_only=args.summary_only,
        n_task_combinations=args.n_task_combinations,
        prefer_run_json=not args.force_pkl_summary,
        summary_output_dir=args.summary_output_dir,
    )
