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

"""Utilities for managing cross-attempt long-term memory.

This module provides functions for:
1. Collecting long-term memory from previous attempts
2. Formatting long-term memory for injection into agent prompts
3. Managing memory state across attempts within the same task
"""

import json
import os
from typing import Any, Dict, List, Optional


def collect_long_term_memories(
    attempt_dirs: List[str],
) -> List[Dict[str, Any]]:
    """Collect long-term memories from previous attempts.
    
    Args:
        attempt_dirs: List of directories for previous attempts.
        
    Returns:
        List of long-term memory dictionaries from failed attempts.
    """
    memories = []
    
    for attempt_dir in attempt_dirs:
        ltm_path = os.path.join(attempt_dir, "long_term_memory.json")
        
        if os.path.exists(ltm_path):
            try:
                with open(ltm_path, "r", encoding="utf-8") as f:
                    ltm_data = json.load(f)
                    
                # Add attempt info
                ltm_data["_source_dir"] = attempt_dir
                memories.append(ltm_data)
                print(f"[LTM] Loaded long-term memory from {attempt_dir}")
                
            except Exception as e:
                print(f"[LTM] Error loading long-term memory from {attempt_dir}: {e}")
                
    return memories


def format_long_term_memory_context(
    memories: List[Dict[str, Any]],
    max_memories: int = 5,
) -> str:
    """Format long-term memories for injection into agent prompt.
    
    Args:
        memories: List of long-term memory dictionaries.
        max_memories: Maximum number of memories to include.
        
    Returns:
        Formatted string to inject into agent prompt.
    """
    if not memories:
        return ""
    
    # Take the most recent memories
    recent_memories = memories[-max_memories:]
    
    formatted_parts = []
    
    for i, mem in enumerate(recent_memories, 1):
        attempt_num = i
        parts = [f"\n--- Previous Attempt {attempt_num} Analysis ---"]
        
        # Key mistake
        key_mistake = mem.get("key_mistake")
        if key_mistake:
            parts.append(f"Key Mistake: {key_mistake}")
        
        # What to avoid
        what_to_avoid = mem.get("what_to_avoid", [])
        if what_to_avoid:
            if isinstance(what_to_avoid, list):
                avoid_str = ", ".join(what_to_avoid)
            else:
                avoid_str = str(what_to_avoid)
            parts.append(f"Avoid: {avoid_str}")
        
        # Suggested approach
        suggested_approach = mem.get("suggested_approach", [])
        if suggested_approach:
            if isinstance(suggested_approach, list):
                approach_str = "; ".join(suggested_approach)
            else:
                approach_str = str(suggested_approach)
            parts.append(f"Suggested Approach: {approach_str}")
        
        # Important insights
        important_insights = mem.get("important_insights", [])
        if important_insights:
            if isinstance(important_insights, list):
                insights_str = "; ".join(important_insights)
            else:
                insights_str = str(important_insights)
            parts.append(f"Important Insights: {insights_str}")
        
        # Hint summary
        hint_summary = mem.get("hint_summary")
        if hint_summary:
            parts.append(f"Summary Hint: {hint_summary}")
        
        formatted_parts.append("\n".join(parts))
    
    if formatted_parts:
        header = "**IMPORTANT: Lessons from Previous Failed Attempts**\n"
        header += "Use these insights to avoid repeating the same mistakes:\n"
        return header + "\n".join(formatted_parts) + "\n---\n"
    
    return ""


def get_attempt_output_dir(
    base_output_dir: str,
    task_name: str,
    instance_id: int,
    attempt: int,
) -> str:
    """Get the output directory for a specific attempt.
    
    Args:
        base_output_dir: Base output directory.
        task_name: Name of the task.
        instance_id: Instance ID within the task.
        attempt: Attempt number (0-indexed).
        
    Returns:
        Path to the attempt output directory.
    """
    return os.path.join(
        base_output_dir,
        f"{task_name}_{instance_id}",
        f"attempt_{attempt}",
    )


def get_all_previous_attempt_dirs(
    base_output_dir: str,
    task_name: str,
    instance_id: int,
    current_attempt: int,
) -> List[str]:
    """Get directories for all previous attempts.
    
    Args:
        base_output_dir: Base output directory.
        task_name: Name of the task.
        instance_id: Instance ID within the task.
        current_attempt: Current attempt number (0-indexed).
        
    Returns:
        List of paths to previous attempt directories.
    """
    previous_dirs = []
    
    for attempt in range(current_attempt):
        attempt_dir = get_attempt_output_dir(
            base_output_dir, task_name, instance_id, attempt
        )
        if os.path.exists(attempt_dir):
            previous_dirs.append(attempt_dir)
            
    return previous_dirs


def save_attempt_result(
    attempt_dir: str,
    result: Dict[str, Any],
    eval_result: float,
) -> None:
    """Save attempt result to file.
    
    Args:
        attempt_dir: Directory for this attempt.
        result: Result dictionary from episode.
        eval_result: Evaluation result (0 or 1).
    """
    os.makedirs(attempt_dir, exist_ok=True)
    
    # Save evaluation result
    eval_data = {
        "is_successful": eval_result,
        "goal": result.get("goal", ""),
        "task_template": result.get("task_template", ""),
        "run_time": result.get("run_time", 0),
        "episode_length": result.get("episode_length", 0),
    }
    
    eval_path = os.path.join(attempt_dir, "eval_result.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)


class CrossAttemptMemoryManager:
    """Manager for cross-attempt long-term memory within a single task.
    
    This class manages the collection and formatting of long-term memories
    from previous attempts of the same task instance.
    """
    
    def __init__(
        self,
        base_output_dir: str,
        task_name: str,
        instance_id: int,
    ):
        """Initialize the memory manager.
        
        Args:
            base_output_dir: Base output directory.
            task_name: Name of the task.
            instance_id: Instance ID within the task.
        """
        self.base_output_dir = base_output_dir
        self.task_name = task_name
        self.instance_id = instance_id
        self.current_attempt = 0
        self.memories: List[Dict[str, Any]] = []
        
    def get_current_attempt_dir(self) -> str:
        """Get the output directory for the current attempt."""
        return get_attempt_output_dir(
            self.base_output_dir,
            self.task_name,
            self.instance_id,
            self.current_attempt,
        )
    
    def collect_previous_memories(self) -> None:
        """Collect long-term memories from previous attempts."""
        previous_dirs = get_all_previous_attempt_dirs(
            self.base_output_dir,
            self.task_name,
            self.instance_id,
            self.current_attempt,
        )
        self.memories = collect_long_term_memories(previous_dirs)
        
    def get_formatted_context(self) -> str:
        """Get formatted long-term memory context for the agent."""
        return format_long_term_memory_context(self.memories)
    
    def advance_attempt(self) -> None:
        """Advance to the next attempt."""
        self.current_attempt += 1
        
    def add_memory(self, memory: Dict[str, Any]) -> None:
        """Add a new memory from the current attempt.
        
        Args:
            memory: Long-term memory dictionary.
        """
        self.memories.append(memory)
        
    def reset_for_new_task(
        self,
        task_name: str,
        instance_id: int,
    ) -> None:
        """Reset the manager for a new task.
        
        Args:
            task_name: Name of the new task.
            instance_id: Instance ID of the new task.
        """
        self.task_name = task_name
        self.instance_id = instance_id
        self.current_attempt = 0
        self.memories = []

