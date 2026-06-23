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

"""Multi-attempt episode runner with cross-attempt long-term memory.

This module implements the multi-attempt rollout logic:
1. For each task instance, run multiple attempts
2. After each attempt, the agent performs self-reflection
3. If the attempt fails (both self-judgment and AndroidWorld evaluator),
   generate long-term memory for the next attempt
4. Long-term memory is independent per task (not shared across tasks)
"""

import dataclasses
import datetime
import gzip
import glob
import io
import json
import os
import pickle
import time
import traceback
from typing import Any, Callable, List, Optional, Type, TypeVar

from android_world import constants
from android_world import episode_runner
from android_world.agents import base_agent
from android_world.agents.long_term_memory_utils import (
    CrossAttemptMemoryManager,
    collect_long_term_memories,
    format_long_term_memory_context,
    get_attempt_output_dir,
    save_attempt_result,
)
from android_world.env import interface
from android_world.task_evals import task_eval
import termcolor


def _gzip_pickle(data: Any) -> bytes:
    """Pickle and gzip compress an object in memory."""
    pickled_data = io.BytesIO()
    pickle.dump(data, pickled_data)
    pickled_data.seek(0)
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode="wb", compresslevel=5) as f_out:
        f_out.write(pickled_data.getvalue())
    return compressed_data.getvalue()


def save_episode_as_pkl_gz(
    episode_data: dict[str, Any],
    checkpoint_dir: str,
    task_name: str,
    instance_id: int,
) -> str:
    """Save episode data as .pkl.gz file.
    
    Args:
        episode_data: Episode data dictionary.
        checkpoint_dir: Directory to save the checkpoint.
        task_name: Name of the task.
        instance_id: Instance ID.
        
    Returns:
        Path to the saved file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f"{task_name}_{instance_id}.pkl.gz")
    with open(filename, "wb") as f:
        compressed = _gzip_pickle([episode_data])
        f.write(compressed)
    print(Colors.info(f"Saved checkpoint: {filename}"))
    return filename


# ANSI Color codes for terminal output
class Colors:
    """Terminal color utility class"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{Colors.BOLD}[X] {text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{Colors.BOLD}[OK] {text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{Colors.BOLD}[!] {text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def attempt(text: str) -> str:
        return f"{Colors.CYAN}{Colors.BOLD}[ATTEMPT] {text}{Colors.RESET}"

    @staticmethod
    def memory(text: str) -> str:
        return f"{Colors.MAGENTA}[LTM] {text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}\n{text}\n{'=' * 60}{Colors.RESET}"


@dataclasses.dataclass
class MultiAttemptResult:
    """Result from running multiple attempts on a task instance.
    
    Attributes:
        task_name: Name of the task.
        instance_id: Instance ID within the task.
        attempt_results: List of results for each attempt.
        final_success: Whether any attempt succeeded.
        best_attempt: Index of the best attempt (first successful or last).
        total_attempts: Total number of attempts made.
    """
    task_name: str
    instance_id: int
    attempt_results: List[dict[str, Any]]
    final_success: float
    best_attempt: int
    total_attempts: int


def _save_screenshots_for_attempt(
    step_data: dict[str, Any],
    attempt_dir: str,
) -> None:
    """Save screenshots from step data to attempt directory.
    
    Args:
        step_data: Step data from episode runner.
        attempt_dir: Directory for this attempt.
    """
    import cv2
    import numpy as np
    
    screenshots_dir = attempt_dir
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Get before_screenshot list from step_data
    screenshots = step_data.get("before_screenshot", [])
    if not isinstance(screenshots, list):
        screenshots = [screenshots]
    
    for i, screenshot in enumerate(screenshots):
        if screenshot is not None and isinstance(screenshot, np.ndarray):
            screenshot_path = os.path.join(screenshots_dir, f"{i}.png")
            try:
                cv2.imwrite(screenshot_path, screenshot)
            except Exception as e:
                print(Colors.warning(f"Failed to save screenshot {i}: {e}"))


def run_single_attempt(
    goal: str,
    agent: base_agent.EnvironmentInteractingAgent,
    task: task_eval.TaskEval,
    env: interface.AsyncEnv,
    parser_attempt_dir: str,
    long_term_memory_context: str = "",
    max_n_steps: int = 30,
    termination_fn: Callable[[interface.AsyncEnv], float] | None = None,
) -> tuple[episode_runner.EpisodeResult, float, dict[str, Any]]:
    """Run a single attempt of a task.
    
    Args:
        goal: Task goal instruction.
        agent: The agent to run.
        task: The task evaluation object.
        env: The environment.
        parser_attempt_dir: Output directory for parsed files (screenshots, JSON).
        long_term_memory_context: Long-term memory from previous attempts.
        max_n_steps: Maximum number of steps.
        termination_fn: Optional termination function.
        
    Returns:
        Tuple of (episode_result, eval_result, reflection_data).
    """
    os.makedirs(parser_attempt_dir, exist_ok=True)
    
    # Check if agent supports long-term memory
    if hasattr(agent, "set_long_term_memory_context"):
        agent.set_long_term_memory_context(long_term_memory_context)
        if long_term_memory_context:
            print(Colors.memory(f"Injected {len(long_term_memory_context)} chars of long-term memory"))
    
    # Run the episode
    episode_result = episode_runner.run_episode(
        goal=goal,
        agent=agent,
        max_n_steps=max_n_steps,
        start_on_home_screen=task.start_on_home_screen,
        termination_fn=termination_fn,
    )
    
    # Save screenshots to parser directory
    _save_screenshots_for_attempt(episode_result.step_data, parser_attempt_dir)
    
    # Get evaluation result from AndroidWorld
    eval_result = task.is_successful(env)
    eval_str = "SUCCESS" if eval_result > 0.5 else "FAILURE"
    print(Colors.info(f"AndroidWorld Evaluator: {eval_str}"))
    
    # Generate self-reflection if agent supports it
    reflection_data = None
    if hasattr(agent, "generate_long_term_memory"):
        reflection_data = agent.generate_long_term_memory(
            output_dir=parser_attempt_dir,
            task_description=goal,
            eval_result=int(eval_result > 0.5),
        )
    
    # Save attempt result to parser directory
    save_attempt_result(
        attempt_dir=parser_attempt_dir,
        result={
            "goal": goal,
            "task_template": task.name,
            "run_time": 0,  # Will be set by caller
            "episode_length": len(episode_result.step_data.get(constants.STEP_NUMBER, [])),
        },
        eval_result=eval_result,
    )
    
    return episode_result, eval_result, reflection_data


def run_multi_attempt_episode(
    task: task_eval.TaskEval,
    instance_id: int,
    agent: base_agent.EnvironmentInteractingAgent,
    env: interface.AsyncEnv,
    checkpoint_dir: str,
    parser_dir: str,
    n_attempts: int = 3,
    max_n_steps: int = 30,
    termination_fn: Callable[[interface.AsyncEnv], float] | None = None,
    demo_mode: bool = False,
) -> MultiAttemptResult:
    """Run multiple attempts on a single task instance.
    
    This is the core function implementing the cross-attempt learning logic:
    1. Run first attempt without long-term memory
    2. After each failed attempt, generate self-reflection
    3. If both self-judgment and evaluator indicate failure, use long-term memory
       for the next attempt
    4. Stop early if an attempt succeeds
    
    Args:
        task: The task evaluation object.
        instance_id: Instance ID within the task.
        agent: The agent to run.
        env: The environment.
        checkpoint_dir: Directory for .pkl.gz checkpoint files.
        parser_dir: Directory for parsed files (screenshots, JSON, etc.).
        n_attempts: Maximum number of attempts.
        max_n_steps: Maximum steps per attempt.
        termination_fn: Optional termination function.
        demo_mode: Whether running in demo mode.
        
    Returns:
        MultiAttemptResult containing all attempt results.
    """
    task_name = task.name
    goal = task.goal
    
    print(Colors.header(f"Multi-Attempt Episode: {task_name} (Instance {instance_id})"))
    print(Colors.info(f"Goal: {goal}"))
    print(Colors.info(f"Max attempts: {n_attempts}"))
    
    # Initialize memory manager with parser_dir for long-term memory files
    memory_manager = CrossAttemptMemoryManager(
        base_output_dir=parser_dir,
        task_name=task_name,
        instance_id=instance_id,
    )
    
    attempt_results = []
    attempt_episode_data = []  # Store episode data for each attempt
    best_attempt = 0
    final_success = 0.0
    
    for attempt in range(n_attempts):
        print(Colors.attempt(f"Starting Attempt {attempt + 1}/{n_attempts}"))
        
        # Get attempt directory for parsed files
        parser_attempt_dir = memory_manager.get_current_attempt_dir()
        os.makedirs(parser_attempt_dir, exist_ok=True)
        
        # Collect long-term memories from previous attempts
        if attempt > 0:
            memory_manager.collect_previous_memories()
            ltm_context = memory_manager.get_formatted_context()
            if ltm_context:
                print(Colors.memory(f"Using long-term memory from {len(memory_manager.memories)} previous attempt(s)"))
        else:
            ltm_context = ""
        
        # Initialize task for this attempt
        start_time = time.time()
        episode_result = None
        try:
            task.initialize_task(env)
            
            # Run the attempt
            episode_result, eval_result, reflection_data = run_single_attempt(
                goal=goal,
                agent=agent,
                task=task,
                env=env,
                parser_attempt_dir=parser_attempt_dir,
                long_term_memory_context=ltm_context,
                max_n_steps=max_n_steps,
                termination_fn=termination_fn,
            )
            
            run_time = time.time() - start_time
            
            # Determine success
            agent_successful = eval_result if episode_result.done else 0.0
            
            # Create attempt result
            attempt_result = {
                "attempt": attempt,
                "attempt_dir": parser_attempt_dir,
                "goal": goal,
                "task_template": task_name,
                "is_successful": agent_successful,
                "episode_length": len(episode_result.step_data.get(constants.STEP_NUMBER, [])),
                "run_time": run_time,
                "done": episode_result.done,
                "reflection_data": reflection_data,
                "step_data": episode_result.step_data,  # Include step data for checkpoint
            }
            attempt_results.append(attempt_result)
            attempt_episode_data.append(episode_result.step_data)
            
            # Log result
            result_str = "SUCCESS ✅" if agent_successful > 0.5 else "FAILURE ❌"
            print(Colors.info(f"Attempt {attempt + 1} Result: {result_str}"))
            
            # Check if successful
            if agent_successful > 0.5:
                final_success = agent_successful
                best_attempt = attempt
                print(Colors.success(f"Task completed successfully on attempt {attempt + 1}!"))
                
                # Tear down and break
                task.tear_down(env)
                break
            else:
                # Check if we got long-term memory for next attempt
                if reflection_data and reflection_data.get("long_term_memory"):
                    memory_manager.add_memory(reflection_data.get("long_term_memory", {}))
                    print(Colors.memory("Generated long-term memory for next attempt"))
                    
        except Exception as e:
            print(Colors.error(f"Error in attempt {attempt + 1}: {e}"))
            traceback.print_exc()
            
            run_time = time.time() - start_time
            attempt_result = {
                "attempt": attempt,
                "attempt_dir": parser_attempt_dir,
                "goal": goal,
                "task_template": task_name,
                "is_successful": 0.0,
                "episode_length": 0,
                "run_time": run_time,
                "done": False,
                "error": str(e),
                "step_data": episode_result.step_data if episode_result else {},
            }
            attempt_results.append(attempt_result)
            attempt_episode_data.append(episode_result.step_data if episode_result else {})
        
        finally:
            # Tear down task
            try:
                task.tear_down(env)
            except Exception:
                pass
        
        # Advance to next attempt
        memory_manager.advance_attempt()
        
        # Brief pause between attempts
        if attempt < n_attempts - 1:
            print(Colors.info("Pausing before next attempt..."))
            time.sleep(2.0)
    
    # Create final result
    result = MultiAttemptResult(
        task_name=task_name,
        instance_id=instance_id,
        attempt_results=attempt_results,
        final_success=final_success,
        best_attempt=best_attempt,
        total_attempts=len(attempt_results),
    )
    
    # Save multi-attempt summary to parser directory
    summary_path = os.path.join(
        parser_dir,
        f"{task_name}_{instance_id}",
        "multi_attempt_summary.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary_data = {
        "task_name": task_name,
        "instance_id": instance_id,
        "goal": goal,
        "total_attempts": result.total_attempts,
        "final_success": result.final_success,
        "best_attempt": result.best_attempt,
        "attempt_results": [
            {
                "attempt": ar["attempt"],
                "is_successful": ar["is_successful"],
                "episode_length": ar["episode_length"],
                "run_time": ar["run_time"],
                "done": ar.get("done", False),
            }
            for ar in attempt_results
        ],
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    # Save .pkl.gz checkpoint file (combining all attempts)
    # Sum up total run time
    total_run_time = sum(ar.get("run_time", 0) for ar in attempt_results)
    
    checkpoint_episode_data = {
        constants.EpisodeConstants.GOAL: goal,
        constants.EpisodeConstants.TASK_TEMPLATE: task_name,
        constants.EpisodeConstants.IS_SUCCESSFUL: final_success,
        constants.EpisodeConstants.RUN_TIME: total_run_time,
        constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
        constants.EpisodeConstants.EPISODE_LENGTH: sum(
            ar.get("episode_length", 0) for ar in attempt_results
        ),
        constants.EpisodeConstants.INSTANCE_ID: instance_id,
        constants.EpisodeConstants.EPISODE_DATA: attempt_episode_data[best_attempt] if attempt_episode_data else {},
        "multi_attempt_data": {
            "total_attempts": result.total_attempts,
            "best_attempt": result.best_attempt,
            "attempt_summaries": [
                {
                    "attempt": ar["attempt"],
                    "is_successful": ar["is_successful"],
                    "episode_length": ar["episode_length"],
                    "run_time": ar["run_time"],
                    "done": ar.get("done", False),
                }
                for ar in attempt_results
            ],
        },
    }
    
    # Save checkpoint
    save_episode_as_pkl_gz(
        episode_data=checkpoint_episode_data,
        checkpoint_dir=checkpoint_dir,
        task_name=task_name,
        instance_id=instance_id,
    )
    
    print(Colors.header(f"Multi-Attempt Summary: {task_name}"))
    print(Colors.info(f"Total attempts: {result.total_attempts}"))
    print(Colors.info(f"Final result: {'SUCCESS ✅' if result.final_success > 0.5 else 'FAILURE ❌'}"))
    if result.final_success > 0.5:
        print(Colors.success(f"Succeeded on attempt {result.best_attempt + 1}"))
    
    return result


def convert_multi_attempt_to_episode_result(
    multi_result: MultiAttemptResult,
) -> dict[str, Any]:
    """Convert MultiAttemptResult to standard episode result format.
    
    This is for compatibility with the existing checkpointing and
    result processing systems.
    
    Args:
        multi_result: The multi-attempt result.
        
    Returns:
        Standard episode result dictionary.
    """
    # Use the best attempt's data
    if multi_result.attempt_results:
        best_idx = multi_result.best_attempt
        best_result = multi_result.attempt_results[best_idx]
    else:
        best_result = {}
    
    # Sum up total run time
    total_run_time = sum(
        ar.get("run_time", 0) for ar in multi_result.attempt_results
    )
    
    return {
        constants.EpisodeConstants.GOAL: best_result.get("goal", ""),
        constants.EpisodeConstants.TASK_TEMPLATE: multi_result.task_name,
        constants.EpisodeConstants.IS_SUCCESSFUL: multi_result.final_success,
        constants.EpisodeConstants.RUN_TIME: total_run_time,
        constants.EpisodeConstants.FINISH_DTIME: datetime.datetime.now(),
        constants.EpisodeConstants.EPISODE_LENGTH: sum(
            ar.get("episode_length", 0) for ar in multi_result.attempt_results
        ),
        "multi_attempt_data": {
            "total_attempts": multi_result.total_attempts,
            "best_attempt": multi_result.best_attempt,
            "attempt_results": multi_result.attempt_results,
        },
    }

