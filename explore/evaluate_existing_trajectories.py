import os
import sys
import json
from glob import glob
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import re # Import re for robust JSON extraction
import time # Import time for retry mechanism
import argparse # Add argparse for command-line arguments

# Add the project root to sys.path to enable importing modules like utils and MLLM_Agent
# This assumes the script is run from the project root or its parent directory is the root.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary functions from existing modules
from utils.utils import load_object_from_disk # Removed extract_json import
from utils.prompt_templates import TASK_EVALUATOR, TASK_GENERATOR, TASK_FEASIBILITY_EVALUATOR # Removed unused single-difficulty templates
from utils.device import _generate_ui_elements_description_list, UIElement
from MLLM_Agent.GUI_explorer import ask_mllm # Assuming ask_mllm is callable externally
from dataclasses import dataclass

# Statistics data structure
@dataclass
class ProcessingStatistics:
    """Statistics for trajectory processing and task generation"""
    # Trajectory processing stats
    total_trajectories: int = 0
    complete_trajectories: int = 0
    incomplete_trajectories: int = 0
    infeasible_trajectories: int = 0
    failed_evaluation_trajectories: int = 0

    # Task generation stats
    tasks_generated_low: int = 0
    tasks_generated_medium: int = 0
    tasks_generated_high: int = 0

    # Task screening stats
    tasks_passed_low: int = 0
    tasks_passed_medium: int = 0
    tasks_passed_high: int = 0
    tasks_rejected_low: int = 0
    tasks_rejected_medium: int = 0
    tasks_rejected_high: int = 0

    # Difficulty reassessment stats
    tasks_difficulty_reassessed: int = 0
    tasks_upgraded_to_medium: int = 0
    tasks_upgraded_to_high: int = 0
    tasks_downgraded_to_low: int = 0
    tasks_downgraded_to_medium: int = 0

    # Token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def get_total_tasks_generated(self) -> int:
        return self.tasks_generated_low + self.tasks_generated_medium + self.tasks_generated_high

    def get_total_tasks_passed(self) -> int:
        return self.tasks_passed_low + self.tasks_passed_medium + self.tasks_passed_high

    def get_total_tasks_rejected(self) -> int:
        return self.tasks_rejected_low + self.tasks_rejected_medium + self.tasks_rejected_high

    def get_pass_rate(self, difficulty: str = "all") -> float:
        if difficulty == "low":
            total = self.tasks_generated_low
            passed = self.tasks_passed_low
        elif difficulty == "medium":
            total = self.tasks_generated_medium
            passed = self.tasks_passed_medium
        elif difficulty == "high":
            total = self.tasks_generated_high
            passed = self.tasks_passed_high
        else:  # all
            total = self.get_total_tasks_generated()
            passed = self.get_total_tasks_passed()

        return (passed / total * 100) if total > 0 else 0.0

# Global statistics instance
processing_stats = ProcessingStatistics()

# Import page analysis functionality
try:
    from utils.page_analyzer import page_analyzer
    PAGE_ANALYSIS_AVAILABLE = True
    print("Page analyzer imported successfully")
except ImportError:
    print("Page analysis functionality not available, will use basic metadata saving")
    PAGE_ANALYSIS_AVAILABLE = False

# Add simplified page analysis functionality as backup
def simple_page_classification(screenshot: np.ndarray, ui_elements: List[UIElement], activity_name: str) -> Dict:
    """Simplified page classification method - backup solution"""
    import hashlib

    # Generate page ID
    ui_signature = f"total_{len(ui_elements)}_click_{sum(1 for ui in ui_elements if getattr(ui, 'is_clickable', False))}_input_{sum(1 for ui in ui_elements if ui.class_name and 'edit' in ui.class_name.lower())}"

    # Simplified screenshot hash calculation
    if screenshot.size > 0:
        shape_str = f"{screenshot.shape[0]}x{screenshot.shape[1]}"
        if len(screenshot.shape) == 3:
            shape_str += f"x{screenshot.shape[2]}"

        # Sample a few pixel points
        sample_points = []
        h, w = screenshot.shape[:2]
        for y in [0, h//4, h//2, 3*h//4, h-1]:
            for x in [0, w//4, w//2, 3*w//4, w-1]:
                if len(screenshot.shape) == 3:
                    sample_points.append(str(screenshot[y, x, 0]))
                else:
                    sample_points.append(str(screenshot[y, x]))

        hash_input = f"{shape_str}_{'.'.join(sample_points[:10])}"
    else:
        hash_input = "empty_screenshot"

    screenshot_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
    page_id = f"simple_{activity_name.split('.')[-1] if activity_name else 'unknown'}_{screenshot_hash}"

    # Simple page type inference
    page_type = "custom_page"
    if activity_name:
        activity_lower = activity_name.lower()
        if "main" in activity_lower or "home" in activity_lower:
            page_type = "main_page"
        elif "setting" in activity_lower:
            page_type = "settings_page"
        elif "login" in activity_lower or "auth" in activity_lower:
            page_type = "login_page"
        elif "list" in activity_lower:
            page_type = "list_page"

    return {
        "page_id": page_id,
        "page_type": page_type,
        "activity_name": activity_name or "unknown",
        "confidence": 0.6,
        "ui_signature": ui_signature,
        "screenshot_hash": screenshot_hash,
        "key_features": [f"Contains {len(ui_elements)} UI elements"],
        "ui_element_count": len(ui_elements),
        "clickable_count": sum(1 for ui in ui_elements if getattr(ui, 'is_clickable', False)),
        "input_count": sum(1 for ui in ui_elements if ui.class_name and 'edit' in ui.class_name.lower())
    }

def create_enhanced_task_metadata(app_name: str, app_pkg: str, task_difficulty: str,
                                original_task_goal: str, new_task_desc: str, completion_condition: str,
                                feasibility_result: Dict, trajectory_data: List[Dict],
                                trajectory_screenshots: List[np.ndarray]) -> Dict:
    """Create enhanced task metadata, including page navigation information"""

    # Basic metadata
    metadata = {
        "app_name": app_name,
        "package_name": app_pkg,
        "difficulty_level": task_difficulty,
        "original_task_goal": original_task_goal,
        "new_task_description": new_task_desc,
        "completion_condition": completion_condition,
        "feasibility_confidence": feasibility_result["confidence"],
        "feasibility_reasoning": feasibility_result["reasoning"],
        "trajectory_screenshots_count": len(trajectory_screenshots),
        "generation_timestamp": time.time(),
    }

    # Always try to add enhanced information (no longer depends on PAGE_ANALYSIS_AVAILABLE)
    if trajectory_data:
        try:
            print("  Generating enhanced metadata...")
            # Analyze starting page
            starting_page_info = analyze_starting_page(trajectory_data[0])
            if starting_page_info:
                metadata["starting_page"] = starting_page_info
                print(f"    Starting page: {starting_page_info.get('page_type', 'unknown')}")

            # Analyze navigation path
            navigation_path = extract_navigation_path(trajectory_data)
            if navigation_path:
                metadata["navigation_path"] = navigation_path
                print(f"    Navigation steps: {len(navigation_path)} steps")

            # Analyze involved page types
            involved_pages = analyze_involved_pages(trajectory_data)
            if involved_pages:
                metadata["involved_pages"] = involved_pages
                print(f"    Involved pages: {len(involved_pages)} pages")

            # Prerequisites analysis
            prerequisites = analyze_prerequisites(trajectory_data, starting_page_info)
            metadata["prerequisites"] = prerequisites

            print("  Enhanced metadata generation completed")

        except Exception as e:
            print(f"Error generating enhanced metadata: {e}")
            metadata["enhancement_error"] = str(e)

    return metadata

def analyze_starting_page(first_step_data: Dict) -> Optional[Dict]:
    """Analyze starting page information"""
    try:
        # Extract screenshot and UI elements
        screenshot = extract_screenshot_from_step(first_step_data)
        ui_elements = extract_ui_elements_from_step(first_step_data)
        activity_name = extract_activity_from_step(first_step_data)

        if screenshot is not None and ui_elements:
            try:
                if PAGE_ANALYSIS_AVAILABLE:
                    # Use page analyzer for classification
                    page_info = page_analyzer.classify_page_type(screenshot, ui_elements, activity_name)
                    return {
                        "page_type": page_info.page_type,
                        "activity_name": page_info.activity_name,
                        "confidence": page_info.confidence,
                        "ui_signature": page_info.ui_signature,
                        "screenshot_hash": page_info.screenshot_hash,
                        "key_features": page_info.key_features,
                        "ui_element_count": page_info.ui_element_count,
                        "clickable_count": page_info.clickable_count,
                        "input_count": page_info.input_count
                    }
                else:
                    # Use simplified analysis as backup
                    return simple_page_classification(screenshot, ui_elements, activity_name)
            except Exception as e:
                print(f"Page analysis failed, using simplified analysis: {e}")
                # Use simplified analysis as fallback
                return simple_page_classification(screenshot, ui_elements, activity_name)
    except Exception as e:
        print(f"Failed to analyze starting page: {e}")

    return None

def extract_navigation_path(trajectory_data: List[Dict]) -> List[Dict]:
    """Extract navigation path information"""
    navigation_path = []

    for i, step_data in enumerate(trajectory_data):
        # Extract action information
        action_info = extract_action_from_step(step_data)
        if action_info:
            step_info = {
                "step_index": i,
                "action_type": action_info.get("action_type", "unknown"),
                "summary": step_data.get("summary", ""),
                "timestamp": step_data.get("timestamp")
            }

            # Add more action details
            if action_info.get("action_type") in ["click", "long_press"]:
                step_info["target_index"] = action_info.get("index")
            elif action_info.get("action_type") == "scroll":
                step_info["direction"] = action_info.get("direction")
            elif action_info.get("action_type") == "input_text":
                step_info["input_text"] = action_info.get("text", "")[:50]  # Limit length

            navigation_path.append(step_info)

    return navigation_path

def analyze_involved_pages(trajectory_data: List[Dict]) -> List[Dict]:
    """Analyze page types involved in trajectory"""
    involved_pages = []
    seen_pages = set()

    for i, step_data in enumerate(trajectory_data):
        try:
            screenshot = extract_screenshot_from_step(step_data)
            ui_elements = extract_ui_elements_from_step(step_data)
            activity_name = extract_activity_from_step(step_data)

            if screenshot is not None and ui_elements:
                try:
                    if PAGE_ANALYSIS_AVAILABLE:
                        page_info = page_analyzer.classify_page_type(screenshot, ui_elements, activity_name)
                        page_signature = f"{page_info.page_type}_{page_info.activity_name}"

                        if page_signature not in seen_pages:
                            seen_pages.add(page_signature)
                            involved_pages.append({
                                "step_index": i,
                                "page_type": page_info.page_type,
                                "activity_name": page_info.activity_name,
                                "confidence": page_info.confidence
                            })
                    else:
                        # Use simplified analysis
                        page_info = simple_page_classification(screenshot, ui_elements, activity_name)
                        page_signature = f"{page_info['page_type']}_{page_info['activity_name']}"

                        if page_signature not in seen_pages:
                            seen_pages.add(page_signature)
                            involved_pages.append({
                                "step_index": i,
                                "page_type": page_info["page_type"],
                                "activity_name": page_info["activity_name"],
                                "confidence": page_info["confidence"]
                            })
                except Exception as e:
                    print(f"Failed to analyze page at step {i}, using simplified analysis: {e}")
                    # Use simplified analysis as fallback
                    page_info = simple_page_classification(screenshot, ui_elements, activity_name)
                    page_signature = f"{page_info['page_type']}_{page_info['activity_name']}"

                    if page_signature not in seen_pages:
                        seen_pages.add(page_signature)
                        involved_pages.append({
                            "step_index": i,
                            "page_type": page_info["page_type"],
                            "activity_name": page_info["activity_name"],
                            "confidence": page_info["confidence"]
                        })

        except Exception as e:
            print(f"Failed to analyze page at step {i}: {e}")

    return involved_pages

def analyze_prerequisites(trajectory_data: List[Dict], starting_page_info: Optional[Dict]) -> Dict:
    """Analyze prerequisites for task execution"""
    prerequisites = {
        "login_required": False,
        "specific_data_needed": [],
        "permissions_needed": [],
        "network_required": False,
        "starting_page_requirements": {}
    }

    # Infer prerequisites based on starting page
    if starting_page_info:
        page_type = starting_page_info.get("page_type", "")

        if page_type == "login_page":
            prerequisites["login_required"] = True
        elif page_type in ["profile_page", "settings_page"]:
            prerequisites["login_required"] = True

        prerequisites["starting_page_requirements"] = {
            "page_type": page_type,
            "activity_name": starting_page_info.get("activity_name", ""),
            "navigation_hint": f"Need to navigate to {page_type} page"
        }

    # Analyze operation patterns in trajectory
    for step_data in trajectory_data:
        summary = step_data.get("summary", "").lower()

        # Check if network is needed
        if any(keyword in summary for keyword in ["search", "download", "sync", "update", "login"]):
            prerequisites["network_required"] = True

        # Check if specific data is needed
        if any(keyword in summary for keyword in ["contact", "photo", "file", "message"]):
            data_type = next((kw for kw in ["contact", "photo", "file", "message"] if kw in summary), "data")
            if data_type not in prerequisites["specific_data_needed"]:
                prerequisites["specific_data_needed"].append(data_type)

    return prerequisites

def extract_screenshot_from_step(step_data: Dict) -> Optional[np.ndarray]:
    """Extract screenshot from step data"""
    screenshot_keys = [
        'after_screenshot_with_som', 'before_screenshot_with_som',
        'after_screenshot', 'before_screenshot', 'raw_screenshot'
    ]

    for key in screenshot_keys:
        if key in step_data and step_data[key] is not None:
            screenshot = step_data[key]
            if isinstance(screenshot, np.ndarray) and screenshot.size > 0:
                return screenshot
    return None

def extract_ui_elements_from_step(step_data: Dict) -> List[UIElement]:
    """Extract UI elements from step data"""
    if 'ui_elements' not in step_data:
        return []

    ui_elements_raw = step_data['ui_elements']
    if not ui_elements_raw:
        return []

    ui_elements = []
    for ui_data in ui_elements_raw:
        if isinstance(ui_data, dict):
            ui_elements.append(UIElement(**ui_data))
        elif isinstance(ui_data, UIElement):
            ui_elements.append(ui_data)

    return ui_elements

def extract_activity_from_step(step_data: Dict) -> Optional[str]:
    """Extract Activity name from step data"""
    activity_keys = ['activity', 'activity_name', 'current_activity']
    for key in activity_keys:
        if key in step_data and step_data[key]:
            activity = step_data[key]
            if isinstance(activity, list) and activity:
                return activity[0]
            elif isinstance(activity, str):
                return activity
    return None

def extract_action_from_step(step_data: Dict) -> Optional[Dict]:
    """Extract action information from step data"""
    if 'converted_action' in step_data:
        action = step_data['converted_action']
        if isinstance(action, dict):
            return action
        elif hasattr(action, 'action_type'):
            # Handle JSONAction objects
            action_dict = {
                "action_type": getattr(action, 'action_type', 'unknown')
            }
            # Add other possible attributes
            for attr in ['direction', 'index', 'text', 'coordinates']:
                if hasattr(action, attr):
                    action_dict[attr] = getattr(action, attr)
            return action_dict
    return None

def robust_extract_json(text: str) -> dict | list | None:
    """
    Robustly extract the outermost JSON object or array from a string that may contain additional text.
    """
    json_str = None
    # Try to match the outermost JSON object {...} or array [...]
    # Use non-greedy matching .*? to capture the shortest match to avoid spanning multiple JSON structures
    # However, for nested structures, simple non-greedy matching may not be sufficient to capture the correct closing delimiter
    # Therefore, we try to match from the first '{' or '[' to the last '}' or ']'

    # Match outermost JSON object from '{' to '}'
    match_obj = re.search(r'\{.*\}', text, re.DOTALL)
    # Match outermost JSON array from '[' to ']'
    match_arr = re.search(r'\[.*\]', text, re.DOTALL)

    # Prioritize the valid match with the earlier starting position
    if match_obj and match_arr:
        if match_obj.start() < match_arr.start():
            json_str = match_obj.group(0)
        else:
            json_str = match_arr.group(0)
    elif match_obj:
        json_str = match_obj.group(0)
    elif match_arr:
        json_str = match_arr.group(0)

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, attempted to parse string: {json_str[:500]}...") # Increase print length for better debugging
            return None
    print(f"Could not find valid JSON structure in text: {text[:500]}...") # Increase print length for better debugging
    return None


def evaluate_task_success(
    task_goal: str,
    trajectory_data: List[Dict[str, Any]],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    max_retries: int = 3, # Add retry parameter
) -> Dict[str, str]:
    """
    Use MLLM to evaluate whether a given task goal was successfully completed in the trajectory.
    Will retry if MLLM output format is incorrect.
    """
    if not trajectory_data:
        return {"evaluation": "incomplete", "reasoning": "No trajectory data available for evaluation."}

    # Extract trajectory summary
    trajectory_summary_list = [
        f"Step {i + 1}: {step.get('summary', 'No summary')}"
        for i, step in enumerate(trajectory_data)
    ]
    trajectory_summary = "\n".join(trajectory_summary_list)

    # Get final screen state and UI elements
    final_step_data = trajectory_data[-1]
    # Prefer screenshots with SOM markers, fallback to regular screenshots
    final_screenshot = final_step_data.get('after_screenshot_with_som') or final_step_data.get('after_screenshot')
    final_ui_elements_raw = final_step_data.get('ui_elements', [])
    final_logical_screen_size = final_step_data.get('logical_screen_size', (1080, 1920))

    # Convert UI elements to MLLM-readable descriptions
    ui_elements_for_desc = [UIElement(**ui) for ui in final_ui_elements_raw]
    final_ui_elements_desc = _generate_ui_elements_description_list(
        ui_elements_for_desc,
        final_logical_screen_size
    )
    final_ui_elements_desc_str = final_ui_elements_desc if final_ui_elements_desc else "No visible UI elements."


    evaluation_prompt = TASK_EVALUATOR.format(
        task_goal=task_goal,
        trajectory_summary=trajectory_summary,
        final_ui_elements=final_ui_elements_desc_str,
    )

    images_to_send = []
    if final_screenshot is not None and isinstance(final_screenshot, np.ndarray) and final_screenshot.size > 0:
        images_to_send.append(final_screenshot)

    for attempt_count in range(max_retries + 1): # max_retries is additional retry count, so total attempts is max_retries + 1
        if attempt_count == 0:
            print(f"  [Evaluation] Attempting MLLM evaluation...")
        else:
            print(f"  [Evaluation] Attempting MLLM evaluation (retry {attempt_count}/{max_retries})... ")
        evaluation_output, _ = ask_mllm(evaluation_prompt, images_to_send)

        try:
            evaluation_result = robust_extract_json(evaluation_output)
            if isinstance(evaluation_result, dict) and "evaluation" in evaluation_result and "reasoning" in evaluation_result:
                # Update token usage statistics
                processing_stats.total_prompt_tokens += usage.get("prompt_tokens", 0)
                processing_stats.total_completion_tokens += usage.get("completion_tokens", 0)
                return evaluation_result
            else:
                print(f"  MLLM returned evaluation result with incorrect format or missing key fields")
        except Exception as e:
            print(f"  Error parsing MLLM evaluation result: {e}")

        # If parsing fails or format is incorrect, and there are still retry opportunities, wait and retry
        if attempt_count < max_retries:
            time.sleep(1) # Brief wait before retry
        else:
            print(f"  MLLM evaluation still failed after {max_retries + 1} attempts.") # Corrected to total attempts
            # Update token usage statistics even for failed attempts
            processing_stats.total_prompt_tokens += usage.get("prompt_tokens", 0)
            processing_stats.total_completion_tokens += usage.get("completion_tokens", 0)
            return {"evaluation": "incomplete", "reasoning": f"MLLM evaluation still failed after {max_retries + 1} attempts, raw output: {evaluation_output}"}

    # This line should theoretically never be executed
    return {"evaluation": "incomplete", "reasoning": "Unknown evaluation error."}

def generate_difficulty_based_tasks(
    app_name: str,
    original_task_goal: str,
    package_name: str,
    activity_list: str,
    difficulty_level: str,  # "low_level", "medium_level", "high_level"
    trajectory_screenshots: List[np.ndarray],  # All screenshots in the trajectory
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    max_retries: int = 3,
) -> List[Dict[str, str]]:
    """
    Use MLLM to generate 5 new tasks based on trajectory screenshots and specified difficulty.
    """
    if not trajectory_screenshots or all(img is None or (isinstance(img, np.ndarray) and img.size == 0) for img in trajectory_screenshots):
        print(f"No valid trajectory screenshots available, cannot generate {difficulty_level} difficulty tasks.")
        return []

    # Filter out valid screenshots
    valid_screenshots = [img for img in trajectory_screenshots if img is not None and isinstance(img, np.ndarray) and img.size > 0]
    if not valid_screenshots:
        print(f"No valid trajectory screenshots available, cannot generate {difficulty_level} difficulty tasks.")
        return []

    task_generator_prompt = TASK_GENERATOR.format(
        app_name=app_name,
        original_task_goal=original_task_goal,
        package_name=package_name,
        activity_list=activity_list,
        difficulty_level=difficulty_level,
    )

    for attempt_count in range(max_retries + 1):
        if attempt_count == 0:
            print(f"  [Task Generation-{difficulty_level}] Attempting MLLM task generation...")
        else:
            print(f"  [Task Generation-{difficulty_level}] Attempting MLLM task generation (retry {attempt_count}/{max_retries})...")
        
        print(f"  [DEBUG] Preparing to send {len(valid_screenshots)} screenshots to MLLM (Task Generation-{difficulty_level})")

        generation_output, _ = ask_mllm(task_generator_prompt, valid_screenshots)

        try:
            new_tasks = robust_extract_json(generation_output)
            if isinstance(new_tasks, list) and all(
                isinstance(task, dict) and 
                "task_description" in task and 
                "completion_condition" in task and
                "difficulty_level" in task
                for task in new_tasks
            ):
                return new_tasks
            else:
                print(f"  MLLM returned {difficulty_level} tasks with incorrect format: {generation_output}")
        except Exception as e:
            print(f"  Error parsing MLLM {difficulty_level} tasks: {e}")
        
        if attempt_count < max_retries:
            time.sleep(1)
        else:
            print(f"  MLLM {difficulty_level} task generation still failed after {max_retries + 1} attempts.")
            return []

    return []


def evaluate_task_feasibility_with_trajectory(
    new_task_description: str,
    difficulty_level: str,  # Add difficulty_level parameter
    trajectory_screenshots: List[np.ndarray],  # All screenshots in the trajectory
    trajectory_summary: str,  # Trajectory summary
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Use MLLM to evaluate the feasibility and confidence of new task based on complete trajectory screenshots.
    """
    if not trajectory_screenshots or all(img is None or (isinstance(img, np.ndarray) and img.size == 0) for img in trajectory_screenshots):
        return {"can_complete": False, "confidence": 0.0, "reasoning": "No valid trajectory screenshots available, cannot evaluate task feasibility."}
    
    # Filter out valid screenshots
    valid_screenshots = [img for img in trajectory_screenshots if img is not None and isinstance(img, np.ndarray) and img.size > 0]
    if not valid_screenshots:
        return {"can_complete": False, "confidence": 0.0, "reasoning": "No valid trajectory screenshots available, cannot evaluate task feasibility."}
    
    feasibility_prompt = TASK_FEASIBILITY_EVALUATOR.format(
        new_task_description=new_task_description,
        difficulty_level=difficulty_level,  # Add difficulty_level parameter
        trajectory_summary=trajectory_summary,
    )

    for attempt_count in range(max_retries + 1):
        if attempt_count == 0:
            print(f"  [Feasibility Assessment-Trajectory] Attempting MLLM feasibility evaluation...")
        else:
            print(f"  [Feasibility Assessment-Trajectory] Attempting MLLM feasibility evaluation (retry {attempt_count}/{max_retries})...")
        
        print(f"  [DEBUG] Preparing to send {len(valid_screenshots)} screenshots to MLLM (Feasibility Assessment-Trajectory)")

        feasibility_output, _ = ask_mllm(feasibility_prompt, valid_screenshots)

        try:
            feasibility_result = robust_extract_json(feasibility_output)
            if isinstance(feasibility_result, dict) and "can_complete" in feasibility_result and "confidence" in feasibility_result and "reasoning" in feasibility_result:
                # 新的响应格式可能包含 reassessed_difficulty 和 difficulty_change_reason 字段，但它们是可选的
                return feasibility_result
            else:
                print(f"  MLLM returned feasibility assessment with incorrect format: {feasibility_output}")
        except Exception as e:
            print(f"  Error parsing MLLM feasibility assessment: {e}")

        if attempt_count < max_retries:
            time.sleep(1)
        else:
            print(f"  MLLM feasibility assessment still failed after {max_retries + 1} attempts.")
            return {"can_complete": False, "confidence": 0.0, "reasoning": f"MLLM feasibility assessment still failed after {max_retries + 1} attempts, raw output: {feasibility_output}"}

    return {"can_complete": False, "confidence": 0.0, "reasoning": "Unknown feasibility assessment error."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate existing trajectories and generate new graded tasks.")
    parser.add_argument('--app_package', type=str, required=True, help="App package name to process, e.g., net.osmand")
    args = parser.parse_args()

    target_app_package = args.app_package

    # Define exploration output directory
    exploration_output_directory = "./exploration_output"
    # Define directory to save newly generated tasks
    generated_tasks_output_directory = "./generated_tasks"
    os.makedirs(generated_tasks_output_directory, exist_ok=True)

    # Only process trajectories under target_app_package folder
    exploration_output_target_dir = os.path.join(exploration_output_directory, target_app_package)
    all_pkl_paths = glob(os.path.join(exploration_output_target_dir, "**", "*.pkl.zst"), recursive=True)

    if not all_pkl_paths:
        print(f"No .pkl.zst trajectory files found in directory '{exploration_output_target_dir}'. Please ensure exploration data for this package exists.")
        sys.exit(0)

    print(f"Found {len(all_pkl_paths)} trajectory data files for evaluation and task generation.")
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # Store app initial page screenshots for each package (first raw_screenshot from first exploration)
    app_initial_screenshots: Dict[str, np.ndarray] = {}

    # Store app_info.json information for each package
    app_infos: Dict[str, Dict[str, Any]] = {}

    # Load target app's app_info.json
    app_info_json_path = os.path.join(exploration_output_target_dir, "app_info.json")
    if os.path.exists(app_info_json_path):
        with open(app_info_json_path, 'r', encoding='utf-8') as f:
            app_infos[target_app_package] = json.load(f)
        print(f"Loaded app_info.json for app {target_app_package}.")
    else:
        print(f"app_info.json not found for app {target_app_package}, cannot proceed with task generation. Please ensure the file exists at {exploration_output_target_dir}.")
        sys.exit(1)

    # Get activity_list from app_info
    # If app_info.json doesn't have activity_list, we can try to get it from the first trajectory's step_data
    # But for simplicity and to avoid complexity, we assume app_info has it or it was generated before task_goal_generator
    # Here we simulate getting it from app_info, providing a default value if not available
    app_activity_list_str = "Not Available" # Default value
    if "activity_list" in app_infos[target_app_package]:
        app_activity_list_str = "\n".join(app_infos[target_app_package].get("activity_list", []))
    # Otherwise, try to get from first trajectory, but note it might be empty
    else:
        # Try to get activity_list from first trajectory
        first_pkl_path = all_pkl_paths[0] if all_pkl_paths else None
        if first_pkl_path:
            try:
                first_trajectory_data = load_object_from_disk(first_pkl_path)
                if first_trajectory_data and first_trajectory_data[0] and "activity" in first_trajectory_data[0]: # Assume activity exists in step_data
                    # Note: original task_goal_generator uses `activity` list then `\n`.join
                    app_activity_list_str = "\n".join(first_trajectory_data[0].get("activity", [])) # Assume activity is a list
            except Exception as e:
                print(f"Error getting activity_list from first trajectory file: {e}")

    for pkl_path in all_pkl_paths:
        app_pkg = target_app_package # Directly use target package name

        print(f"\n--- Processing trajectory file: {pkl_path} ---")
        try:
            trajectory_data = load_object_from_disk(pkl_path)

            if not trajectory_data or not isinstance(trajectory_data, list):
                print("Trajectory data is empty or format is incorrect, skipping.")
                continue

            # Update total trajectory count
            processing_stats.total_trajectories += 1

            # Get app initial page screenshot (if not recorded yet)
            if app_pkg not in app_initial_screenshots and trajectory_data[0].get('raw_screenshot') is not None and isinstance(trajectory_data[0].get('raw_screenshot'), np.ndarray) and trajectory_data[0].get('raw_screenshot').size > 0:
                app_initial_screenshots[app_pkg] = trajectory_data[0].get('raw_screenshot')
                print(f"Recorded initial page screenshot for app {app_pkg}.")

            # Extract task goal
            task_goal = trajectory_data[0].get('goal', 'N/A')
            if task_goal == 'N/A':
                print("Task goal not found, skipping this trajectory.")
                continue

            print(f"Original task goal: {task_goal}")

            # Evaluate original task success (with retry mechanism)
            evaluation_result = evaluate_task_success(task_goal, trajectory_data, total_usage, max_retries=3)

            print(f"Original task evaluation result: {evaluation_result['evaluation']}")
            print(f"Original task evaluation reason: {evaluation_result['reasoning']}")

            # Update trajectory evaluation statistics
            if evaluation_result["evaluation"] == "complete":
                processing_stats.complete_trajectories += 1
            elif evaluation_result["evaluation"] == "incomplete":
                processing_stats.incomplete_trajectories += 1
            elif evaluation_result["evaluation"] == "infeasible":
                processing_stats.infeasible_trajectories += 1
            else:
                processing_stats.failed_evaluation_trajectories += 1

            if evaluation_result["evaluation"] == "complete":
                print(f"Original task '{task_goal}' evaluated as complete, starting to generate graded tasks...")
                
                app_name = app_infos.get(app_pkg, {}).get('app_name', app_pkg) # Get app name
                
                # Extract all screenshots from trajectory for task generation and feasibility evaluation
                trajectory_screenshots = []
                for i, step_data in enumerate(trajectory_data):
                    screenshot = None
                    if 'after_screenshot_with_som' in step_data and step_data['after_screenshot_with_som'] is not None:
                        screenshot = step_data['after_screenshot_with_som']
                    elif 'before_screenshot_with_som' in step_data and step_data['before_screenshot_with_som'] is not None:
                        screenshot = step_data['before_screenshot_with_som']
                    elif 'after_screenshot' in step_data and step_data['after_screenshot'] is not None:
                        screenshot = step_data['after_screenshot']
                    elif 'before_screenshot' in step_data and step_data['before_screenshot'] is not None:
                        screenshot = step_data['before_screenshot']
                    elif 'raw_screenshot' in step_data and step_data['raw_screenshot'] is not None:
                        screenshot = step_data['raw_screenshot']

                    if screenshot is not None:
                        if not isinstance(screenshot, np.ndarray):
                            print(f"  [ERROR] Screenshot at step {i} is not a numpy array, type: {type(screenshot)}")
                            continue
                        if screenshot.size == 0:
                            print(f"  [WARNING] Screenshot at step {i} is an empty numpy array.")
                            continue
                        # Ensure screenshot.size is a scalar, not an array
                        if not isinstance(screenshot.size, (int, np.integer)):
                             print(f"  [ERROR] Screenshot.size at step {i} returns a non-scalar value, type: {type(screenshot.size)}, value: {screenshot.size}")
                             continue

                        trajectory_screenshots.append(screenshot)
                
                if not trajectory_screenshots:
                    print(f"No valid screenshots in trajectory, cannot generate tasks, skipping.")
                    continue
                
                print(f"Extracted {len(trajectory_screenshots)} valid screenshots from trajectory.")
                
                # Generate trajectory summary for feasibility evaluation
                trajectory_summary_list = [
                    f"Step {i + 1}: {step.get('summary', 'No summary')}"
                    for i, step in enumerate(trajectory_data)
                ]
                trajectory_summary = "\n".join(trajectory_summary_list)
                
                # Generate tasks for three difficulty levels
                difficulty_levels = ["low_level", "medium_level", "high_level"]
                
                for difficulty_level in difficulty_levels:
                    print(f"\n  === Generating {difficulty_level} difficulty tasks ===")
                    
                    # Generate tasks for specified difficulty
                    difficulty_tasks = generate_difficulty_based_tasks(
                        app_name=app_name,
                        original_task_goal=task_goal,
                        package_name=app_pkg,
                        activity_list=app_activity_list_str,
                        difficulty_level=difficulty_level,
                        trajectory_screenshots=trajectory_screenshots,
                        usage=total_usage,
                        max_retries=3,
                    )
                    
                    if difficulty_tasks:
                        print(f"  Generated {len(difficulty_tasks)} {difficulty_level} difficulty tasks, starting screening...")

                        # Update task generation statistics
                        if difficulty_level == "low_level":
                            processing_stats.tasks_generated_low += len(difficulty_tasks)
                        elif difficulty_level == "medium_level":
                            processing_stats.tasks_generated_medium += len(difficulty_tasks)
                        elif difficulty_level == "high_level":
                            processing_stats.tasks_generated_high += len(difficulty_tasks)
                        
                        for i, new_task_info in enumerate(difficulty_tasks):
                            new_task_desc = new_task_info["task_description"]
                            completion_condition = new_task_info["completion_condition"]
                            task_difficulty = new_task_info.get("difficulty_level", difficulty_level)
                            
                            # Use trajectory screenshots for feasibility evaluation
                            feasibility_result = evaluate_task_feasibility_with_trajectory(
                                new_task_description=new_task_desc,
                                difficulty_level=task_difficulty,
                                trajectory_screenshots=trajectory_screenshots,
                                trajectory_summary=trajectory_summary,
                                usage=total_usage,
                                max_retries=3,
                            )
                            
                            if feasibility_result["can_complete"]:
                                # 检查是否需要重新评估难度级别
                                original_difficulty = difficulty_level
                                reassessed_difficulty = feasibility_result.get("reassessed_difficulty", None)
                                difficulty_changed = False

                                if reassessed_difficulty:
                                    # 将重新评估的难度映射到标准格式
                                    difficulty_mapping = {
                                        "low": "low_level",
                                        "medium": "medium_level",
                                        "high": "high_level"
                                    }
                                    mapped_reassessed_difficulty = difficulty_mapping.get(reassessed_difficulty, difficulty_level)

                                    if mapped_reassessed_difficulty != original_difficulty:
                                        difficulty_changed = True
                                        difficulty_level = mapped_reassessed_difficulty
                                        print(f"    [Difficulty Reassessed] Task difficulty changed from {original_difficulty} to {difficulty_level}")
                                        if "difficulty_change_reason" in feasibility_result:
                                            print(f"      Reason: {feasibility_result['difficulty_change_reason']}")

                                        # 更新难度重新评估统计
                                        processing_stats.tasks_difficulty_reassessed += 1

                                        # 更新具体的难度变化统计
                                        if original_difficulty == "low_level" and difficulty_level == "medium_level":
                                            processing_stats.tasks_upgraded_to_medium += 1
                                        elif original_difficulty == "low_level" and difficulty_level == "high_level":
                                            processing_stats.tasks_upgraded_to_high += 1
                                        elif original_difficulty == "medium_level" and difficulty_level == "high_level":
                                            processing_stats.tasks_upgraded_to_high += 1
                                        elif original_difficulty == "medium_level" and difficulty_level == "low_level":
                                            processing_stats.tasks_downgraded_to_low += 1
                                        elif original_difficulty == "high_level" and difficulty_level == "low_level":
                                            processing_stats.tasks_downgraded_to_low += 1
                                        elif original_difficulty == "high_level" and difficulty_level == "medium_level":
                                            processing_stats.tasks_downgraded_to_medium += 1

                                print(f"    [Screening Passed] {difficulty_level} task '{new_task_desc}' (confidence: {feasibility_result['confidence']:.2f})")

                                # Update task screening pass statistics (using final difficulty level)
                                if difficulty_level == "low_level":
                                    processing_stats.tasks_passed_low += 1
                                elif difficulty_level == "medium_level":
                                    processing_stats.tasks_passed_medium += 1
                                elif difficulty_level == "high_level":
                                    processing_stats.tasks_passed_high += 1

                                # Create directory structure organized by (final) difficulty
                                task_id = f"{app_pkg}_{difficulty_level}_{str(hash(new_task_desc))[:8]}"
                                difficulty_output_dir = os.path.join(generated_tasks_output_directory, app_pkg, difficulty_level)
                                os.makedirs(difficulty_output_dir, exist_ok=True)

                                # 增强任务元数据保存 - 集成页面分析功能和难度重新评估信息
                                enhanced_metadata = create_enhanced_task_metadata(
                                    app_name, app_pkg, task_difficulty, task_goal, new_task_desc,
                                    completion_condition, feasibility_result, trajectory_data, trajectory_screenshots
                                )

                                # 添加难度重新评估信息到元数据
                                if difficulty_changed:
                                    enhanced_metadata["difficulty_reassessment"] = {
                                        "original_difficulty": original_difficulty,
                                        "reassessed_difficulty": difficulty_level,
                                        "change_reason": feasibility_result.get("difficulty_change_reason", "No reason provided")
                                    }

                                task_json_path = os.path.join(difficulty_output_dir, f"{task_id}.json")
                                with open(task_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
                                print(f"      Task information saved to: {task_json_path}")
                                
                                # Save first screenshot from trajectory as reference
                                if trajectory_screenshots:
                                    image_filename = os.path.join(difficulty_output_dir, f"{task_id}_reference_screen.png")
                                    try:
                                        Image.fromarray(trajectory_screenshots[0].astype(np.uint8)).save(image_filename, "PNG")
                                        print(f"      Reference screenshot saved to: {image_filename}")
                                    except Exception as img_e:
                                        print(f"      Error saving reference screenshot: {img_e}")
                            else:
                                print(f"    [Screening Failed] {difficulty_level} task '{new_task_desc}' (confidence: {feasibility_result['confidence']:.2f}, reason: {feasibility_result['reasoning']})")

                                # Update task screening rejection statistics
                                if difficulty_level == "low_level":
                                    processing_stats.tasks_rejected_low += 1
                                elif difficulty_level == "medium_level":
                                    processing_stats.tasks_rejected_medium += 1
                                elif difficulty_level == "high_level":
                                    processing_stats.tasks_rejected_high += 1
                    else:
                        print(f"  Failed to generate {difficulty_level} difficulty tasks.")

        except Exception as e:
            print(f"Error processing trajectory file {pkl_path}: {e}")

    print(f"\n=== 轨迹处理和任务生成统计报告 ===")
    print(f"总处理轨迹数: {processing_stats.total_trajectories}")
    print(f"成功完成轨迹数: {processing_stats.complete_trajectories} ({processing_stats.complete_trajectories/processing_stats.total_trajectories*100:.1f}%)" if processing_stats.total_trajectories > 0 else "成功完成轨迹数: 0")
    print(f"不完整轨迹数: {processing_stats.incomplete_trajectories} ({processing_stats.incomplete_trajectories/processing_stats.total_trajectories*100:.1f}%)" if processing_stats.total_trajectories > 0 else "不完整轨迹数: 0")
    print(f"不可行轨迹数: {processing_stats.infeasible_trajectories} ({processing_stats.infeasible_trajectories/processing_stats.total_trajectories*100:.1f}%)" if processing_stats.total_trajectories > 0 else "不可行轨迹数: 0")
    print(f"评估失败轨迹数: {processing_stats.failed_evaluation_trajectories}")

    print(f"\n=== 任务生成统计 ===")
    print(f"低难度任务生成数: {processing_stats.tasks_generated_low}")
    print(f"中难度任务生成数: {processing_stats.tasks_generated_medium}")
    print(f"高难度任务生成数: {processing_stats.tasks_generated_high}")
    print(f"总任务生成数: {processing_stats.get_total_tasks_generated()}")

    print(f"\n=== 任务筛选统计 ===")
    print(f"低难度任务通过数: {processing_stats.tasks_passed_low} / {processing_stats.tasks_generated_low} ({processing_stats.get_pass_rate('low'):.1f}%)" if processing_stats.tasks_generated_low > 0 else "低难度任务通过数: 0 / 0")
    print(f"中难度任务通过数: {processing_stats.tasks_passed_medium} / {processing_stats.tasks_generated_medium} ({processing_stats.get_pass_rate('medium'):.1f}%)" if processing_stats.tasks_generated_medium > 0 else "中难度任务通过数: 0 / 0")
    print(f"高难度任务通过数: {processing_stats.tasks_passed_high} / {processing_stats.tasks_generated_high} ({processing_stats.get_pass_rate('high'):.1f}%)" if processing_stats.tasks_generated_high > 0 else "高难度任务通过数: 0 / 0")
    print(f"总任务通过数: {processing_stats.get_total_tasks_passed()} / {processing_stats.get_total_tasks_generated()} ({processing_stats.get_pass_rate():.1f}%)" if processing_stats.get_total_tasks_generated() > 0 else "总任务通过数: 0 / 0")

    print(f"\n=== 难度重新评估统计 ===")
    print(f"难度重新评估任务数: {processing_stats.tasks_difficulty_reassessed}")
    if processing_stats.tasks_difficulty_reassessed > 0:
        print(f"  升级到中等难度: {processing_stats.tasks_upgraded_to_medium}")
        print(f"  升级到高等难度: {processing_stats.tasks_upgraded_to_high}")
        print(f"  降级到低等难度: {processing_stats.tasks_downgraded_to_low}")
        print(f"  降级到中等难度: {processing_stats.tasks_downgraded_to_medium}")
        reassessment_rate = (processing_stats.tasks_difficulty_reassessed / processing_stats.get_total_tasks_passed() * 100) if processing_stats.get_total_tasks_passed() > 0 else 0
        print(f"  重新评估率: {reassessment_rate:.1f}% (在通过筛选的任务中)")
    else:
        print("  无任务需要难度重新评估")

    print(f"\n=== Token使用统计 ===")
    print(f"总Prompt Tokens: {processing_stats.total_prompt_tokens:,}")
    print(f"总Completion Tokens: {processing_stats.total_completion_tokens:,}")
    print(f"总Token数: {processing_stats.total_prompt_tokens + processing_stats.total_completion_tokens:,}")

    print(f"\nAll trajectory evaluation and task generation completed.")
    print(f"Total LLM Token usage: {total_usage}")
