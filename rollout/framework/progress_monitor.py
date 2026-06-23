"""
Real-time progress monitoring and metrics calculation module.
Provides functions to calculate and display Pass@K metrics.
Also handles saving metrics to files for persistence and analysis.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


# File names for metrics storage
REALTIME_METRICS_FILE = "realtime_metrics.json"
METRICS_HISTORY_FILE = "realtime_metrics_history.jsonl"
FINAL_SUMMARY_FILE = "final_metrics_summary.md"

# Global start time for time estimation
_start_time: Optional[float] = None


def set_start_time():
    """Set the start time for time estimation."""
    global _start_time
    _start_time = time.time()


def get_elapsed_time() -> float:
    """Get elapsed time in seconds since start."""
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def collect_task_results(
    task_scope: List[Any],
    agent_scope: List[str],
    output_dir: str,
    max_attempts: int,
) -> Dict[str, Any]:
    """
    Collect execution and evaluation results for all tasks.

    Args:
        task_scope: List of task namedtuples with task_identifier
        agent_scope: List of agent names
        output_dir: Base output directory
        max_attempts: Maximum number of attempts per task

    Returns:
        Dictionary containing collected metrics data
    """
    total_tasks = len(task_scope)
    executed_count = 0
    evaluated_count = 0

    # Pass@K tracking
    pass_at_k = {k: set() for k in range(1, max_attempts + 1)}
    # Pass@K tracking for fully evaluated tasks only (to avoid >100% rates)
    pass_at_k_evaluated = {k: set() for k in range(1, max_attempts + 1)}

    # Track attempt results
    task_attempt_results = {}
    
    # Track per-app metrics
    app_metrics = {}  # {app_name: {"tasks": set(), "pass_at_k": {...}, "executed": 0, "evaluated": 0}}
    
    # Track step counts
    all_attempts_steps = []  # All attempts step counts
    success_attempts_steps = []  # Only successful attempts step counts

    for agent_name in agent_scope:
        for task in task_scope:
            task_id = task.task_identifier
            # Get app_name from task (if available)
            app_name = getattr(task, 'app_name', 'Unknown')
            
            # Initialize app_metrics entry
            if app_name not in app_metrics:
                app_metrics[app_name] = {
                    "tasks": set(),
                    "pass_at_k": {k: set() for k in range(1, max_attempts + 1)},
                    "pass_at_k_evaluated": {k: set() for k in range(1, max_attempts + 1)},  # Only for fully evaluated tasks
                    "executed": 0,
                    "evaluated": 0,
                    "total_tasks": 0,
                }
            app_metrics[app_name]["tasks"].add(task_id)
            app_metrics[app_name]["total_tasks"] = len(app_metrics[app_name]["tasks"])
            
            executed_attempts = 0  # 已执行的 attempts 数量
            evaluated_attempts = 0  # 已评估的 attempts 数量
            first_success_attempt = None
            attempt_results = {}

            for attempt in range(1, max_attempts + 1):
                attempt_dir = os.path.join(
                    output_dir, task_id, agent_name, f"attempt_{attempt}"
                )
                log_path = os.path.join(attempt_dir, "log.json")
                eval_path = os.path.join(attempt_dir, "evaluation_summary.json")

                # Read step count from log.json
                if os.path.exists(log_path):
                    executed_attempts += 1
                    try:
                        with open(log_path, "r") as f:
                            log_data = json.load(f)
                            if log_data and isinstance(log_data, list):
                                summary = log_data[-1]  # Last element is summary
                                total_steps = summary.get("total_steps", 0)
                                all_attempts_steps.append(total_steps)
                    except (json.JSONDecodeError, IOError, KeyError, IndexError):
                        pass

                # Read evaluation result
                attempt_success = False
                if os.path.exists(eval_path):
                    evaluated_attempts += 1
                    try:
                        with open(eval_path, "r") as f:
                            result = json.load(f).get("final_result", 0)
                            attempt_success = result == 1
                            attempt_results[attempt] = attempt_success
                            if attempt_success and first_success_attempt is None:
                                first_success_attempt = attempt
                                
                            # If success, add step count to success list
                            if attempt_success and os.path.exists(log_path):
                                try:
                                    with open(log_path, "r") as lf:
                                        log_data = json.load(lf)
                                        if log_data and isinstance(log_data, list):
                                            summary = log_data[-1]
                                            total_steps = summary.get("total_steps", 0)
                                            success_attempts_steps.append(total_steps)
                                except (json.JSONDecodeError, IOError, KeyError, IndexError):
                                    pass
                    except (json.JSONDecodeError, IOError):
                        attempt_results[attempt] = False

            task_attempt_results[task_id] = attempt_results

            # 判断任务是否"完成"执行：
            # - rollout 模式：所有 max_attempts 都执行完（_EARLY_STOP_ON_SUCCESS: false）
            # - benchmark 模式：成功后早停（_EARLY_STOP_ON_SUCCESS: true）
            # 任务完成的条件：执行了所有 attempts，或者有成功且后续没有继续执行
            task_execution_completed = (
                executed_attempts == max_attempts or  # 执行了所有 attempts
                (first_success_attempt is not None and executed_attempts == first_success_attempt)  # 成功后早停
            )
            
            if task_execution_completed:
                executed_count += 1
                app_metrics[app_name]["executed"] += 1
            
            # 判断任务是否"完成"评估：
            # - 所有已执行的 attempts 都已评估完成
            # - 并且任务执行已完成
            is_fully_evaluated = (
                executed_attempts > 0 and
                evaluated_attempts == executed_attempts and  # 所有已执行的 attempts 都已评估
                task_execution_completed  # 任务执行已完成
            )
            if is_fully_evaluated:
                evaluated_count += 1
                app_metrics[app_name]["evaluated"] += 1

            # Pass@K calculation (overall and per-app)
            if first_success_attempt is not None:
                for k in range(first_success_attempt, max_attempts + 1):
                    pass_at_k[k].add(task_id)
                    app_metrics[app_name]["pass_at_k"][k].add(task_id)
                    # Only add to pass_at_k_evaluated if this task is fully evaluated
                    if is_fully_evaluated:
                        pass_at_k_evaluated[k].add(task_id)
                        app_metrics[app_name]["pass_at_k_evaluated"][k].add(task_id)

    return {
        "total_tasks": total_tasks,
        "executed_count": executed_count,
        "evaluated_count": evaluated_count,
        "pass_at_k": pass_at_k,
        "pass_at_k_evaluated": pass_at_k_evaluated,  # Only for fully evaluated tasks
        "task_attempt_results": task_attempt_results,
        "max_attempts": max_attempts,
        "app_metrics": app_metrics,
        "all_attempts_steps": all_attempts_steps,
        "success_attempts_steps": success_attempts_steps,
    }


def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all metrics from collected data.

    Args:
        data: Dictionary from collect_task_results()

    Returns:
        Dictionary containing all calculated metrics
    """
    total_tasks = data["total_tasks"]
    evaluated_count = data["evaluated_count"]
    max_attempts = data["max_attempts"]

    # Pass@K rates (overall - based on all tasks)
    pass_at_k_rates = {}
    for k in range(1, max_attempts + 1):
        pass_at_k_rates[k] = (
            len(data["pass_at_k"][k]) / total_tasks * 100 if total_tasks > 0 else 0
        )

    # Pass@K rates (evaluated only - based on evaluated tasks)
    # Use pass_at_k_evaluated which only counts fully evaluated tasks to avoid >100% rates
    pass_at_k_rates_evaluated = {}
    for k in range(1, max_attempts + 1):
        pass_at_k_rates_evaluated[k] = (
            len(data["pass_at_k_evaluated"][k]) / evaluated_count * 100
            if evaluated_count > 0
            else 0
        )

    # Success distribution: count how many tasks succeeded exactly N times
    success_distribution = {i: 0 for i in range(max_attempts + 1)}
    for task_id, results in data["task_attempt_results"].items():
        success_count = sum(1 for v in results.values() if v)
        if success_count <= max_attempts:
            success_distribution[success_count] += 1
    
    # Calculate per-app metrics
    app_metrics_calculated = {}
    for app_name, app_data in data["app_metrics"].items():
        app_total = app_data["total_tasks"]
        app_evaluated = app_data["evaluated"]
        
        # Per-app Pass@K rates
        app_pass_at_k_rates = {}
        app_pass_at_k_rates_evaluated = {}
        for k in range(1, max_attempts + 1):
            app_pass_at_k_rates[k] = (
                len(app_data["pass_at_k"][k]) / app_total * 100 
                if app_total > 0 else 0
            )
            # Use pass_at_k_evaluated to avoid >100% rates
            app_pass_at_k_rates_evaluated[k] = (
                len(app_data["pass_at_k_evaluated"][k]) / app_evaluated * 100
                if app_evaluated > 0 else 0
            )
        
        app_metrics_calculated[app_name] = {
            "total_tasks": app_total,
            "executed": app_data["executed"],
            "evaluated": app_evaluated,
            "pass_at_k_rates": app_pass_at_k_rates,
            "pass_at_k_rates_evaluated": app_pass_at_k_rates_evaluated,
            "pass_at_k_counts": {k: len(app_data["pass_at_k"][k]) for k in range(1, max_attempts + 1)},
        }
    
    # Calculate step statistics
    all_steps = data["all_attempts_steps"]
    success_steps = data["success_attempts_steps"]
    
    step_stats = {
        "all_attempts": {
            "total_steps": sum(all_steps),
            "count": len(all_steps),
            "avg_steps": sum(all_steps) / len(all_steps) if all_steps else 0,
            "min_steps": min(all_steps) if all_steps else 0,
            "max_steps": max(all_steps) if all_steps else 0,
        },
        "success_attempts": {
            "total_steps": sum(success_steps),
            "count": len(success_steps),
            "avg_steps": sum(success_steps) / len(success_steps) if success_steps else 0,
            "min_steps": min(success_steps) if success_steps else 0,
            "max_steps": max(success_steps) if success_steps else 0,
        },
    }

    return {
        "pass_at_k_rates": pass_at_k_rates,
        "pass_at_k_rates_evaluated": pass_at_k_rates_evaluated,
        "success_distribution": success_distribution,
        "app_metrics": app_metrics_calculated,
        "step_stats": step_stats,
    }


def build_metrics_snapshot(
    data: Dict[str, Any], metrics: Dict[str, Any], trigger: str
) -> Dict[str, Any]:
    """
    Build a complete metrics snapshot for saving.

    Args:
        data: Raw collected data
        metrics: Calculated metrics
        trigger: What triggered this snapshot

    Returns:
        Complete metrics snapshot dictionary
    """
    max_attempts = data["max_attempts"]

    # Convert sets to counts for JSON serialization
    pass_at_k_counts = {
        k: len(data["pass_at_k"][k]) for k in range(1, max_attempts + 1)
    }

    # Calculate time info
    elapsed = get_elapsed_time()
    total_tasks = data["total_tasks"]
    evaluated_count = data["evaluated_count"]
    
    eta_seconds = None
    if evaluated_count > 0 and evaluated_count < total_tasks:
        avg_time_per_task = elapsed / evaluated_count
        remaining_tasks = total_tasks - evaluated_count
        eta_seconds = avg_time_per_task * remaining_tasks

    return {
        "timestamp": datetime.now().isoformat(),
        "trigger": trigger,
        "time": {
            "elapsed_seconds": elapsed,
            "elapsed_formatted": format_duration(elapsed),
            "eta_seconds": eta_seconds,
            "eta_formatted": format_duration(eta_seconds) if eta_seconds else None,
        },
        "progress": {
            "total_tasks": total_tasks,
            "executed": data["executed_count"],
            "evaluated": evaluated_count,
            "execution_rate": (
                data["executed_count"] / total_tasks * 100
                if total_tasks > 0
                else 0
            ),
            "evaluation_rate": (
                evaluated_count / total_tasks * 100
                if total_tasks > 0
                else 0
            ),
        },
        "pass_at_k": {
            "counts": pass_at_k_counts,
            "rates_overall": metrics["pass_at_k_rates"],
            "rates_evaluated": metrics["pass_at_k_rates_evaluated"],
        },
        "success_distribution": metrics["success_distribution"],
        "app_metrics": metrics["app_metrics"],
        "step_statistics": metrics["step_stats"],
        "max_attempts": max_attempts,
    }


def save_realtime_metrics(output_dir: str, snapshot: Dict[str, Any]) -> None:
    """
    Save current metrics snapshot to realtime_metrics.json.

    Args:
        output_dir: Base output directory
        snapshot: Metrics snapshot to save
    """
    metrics_path = os.path.join(output_dir, REALTIME_METRICS_FILE)
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Failed to save realtime metrics: {e}")


def append_metrics_history(output_dir: str, snapshot: Dict[str, Any]) -> None:
    """
    Append metrics snapshot to history file (JSONL format).

    Args:
        output_dir: Base output directory
        snapshot: Metrics snapshot to append
    """
    history_path = os.path.join(output_dir, METRICS_HISTORY_FILE)
    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    except IOError as e:
        print(f"Warning: Failed to append metrics history: {e}")


def generate_final_summary(output_dir: str, snapshot: Dict[str, Any]) -> None:
    """
    Generate final metrics summary in Markdown format.

    Args:
        output_dir: Base output directory
        snapshot: Final metrics snapshot
    """
    summary_path = os.path.join(output_dir, FINAL_SUMMARY_FILE)

    progress = snapshot["progress"]
    pass_at_k = snapshot["pass_at_k"]
    success_dist = snapshot["success_distribution"]
    max_attempts = snapshot["max_attempts"]
    app_metrics = snapshot.get("app_metrics", {})
    step_stats = snapshot.get("step_statistics", {})

    md_content = f"""# Evaluation Metrics Summary

Generated: {snapshot["timestamp"]}

## Progress Overview

| Metric | Value |
|--------|-------|
| Total Tasks | {progress["total_tasks"]} |
| Executed | {progress["executed"]} ({progress["execution_rate"]:.1f}%) |
| Evaluated | {progress["evaluated"]} ({progress["evaluation_rate"]:.1f}%) |

## Pass@K Results (Overall - based on all tasks)

| K | Success Count | Success Rate (out of {progress["total_tasks"]} tasks) |
|---|---------------|--------------|
"""

    for k in range(1, max_attempts + 1):
        md_content += f"| {k} | {pass_at_k['counts'][k]} | {pass_at_k['rates_overall'][k]:.1f}% |\n"

    md_content += f"""
## Pass@K Results (Evaluated Only - based on {progress["evaluated"]} evaluated tasks)

| K | Success Count | Success Rate |
|---|---------------|--------------|
"""

    for k in range(1, max_attempts + 1):
        md_content += f"| {k} | {pass_at_k['counts'][k]} | {pass_at_k['rates_evaluated'][k]:.1f}% |\n"

    md_content += """
## Success Distribution

| Success Count | Tasks |
|---------------|-------|
"""

    for i in range(max_attempts + 1):
        md_content += f"| {i} | {success_dist.get(i, 0)} |\n"

    # Add per-app metrics
    if app_metrics:
        md_content += "\n## Per-App Metrics\n\n"
        for app_name in sorted(app_metrics.keys()):
            app_data = app_metrics[app_name]
            md_content += f"### {app_name}\n\n"
            md_content += f"| Metric | Value |\n"
            md_content += f"|--------|-------|\n"
            md_content += f"| Total Tasks | {app_data['total_tasks']} |\n"
            md_content += f"| Executed | {app_data['executed']} |\n"
            md_content += f"| Evaluated | {app_data['evaluated']} |\n"
            
            md_content += f"\n**Pass@K (Overall):**\n\n"
            md_content += f"| K | Success Count | Success Rate |\n"
            md_content += f"|---|---------------|-------------|\n"
            for k in range(1, max_attempts + 1):
                count = app_data['pass_at_k_counts'][k]
                rate = app_data['pass_at_k_rates'][k]
                md_content += f"| {k} | {count} | {rate:.1f}% |\n"
            
            if app_data['evaluated'] > 0:
                md_content += f"\n**Pass@K (Evaluated only):**\n\n"
                md_content += f"| K | Success Count | Success Rate |\n"
                md_content += f"|---|---------------|-------------|\n"
                for k in range(1, max_attempts + 1):
                    count = app_data['pass_at_k_counts'][k]
                    rate = app_data['pass_at_k_rates_evaluated'][k]
                    md_content += f"| {k} | {count} | {rate:.1f}% |\n"
            
            md_content += "\n"
    
    # Add step statistics
    if step_stats:
        md_content += "## Agent Step Statistics\n\n"
        
        # All attempts
        all_stats = step_stats.get("all_attempts", {})
        md_content += "### All Attempts\n\n"
        md_content += "| Metric | Value |\n"
        md_content += "|--------|-------|\n"
        md_content += f"| Total Attempts | {all_stats.get('count', 0)} |\n"
        md_content += f"| Total Steps | {all_stats.get('total_steps', 0)} |\n"
        md_content += f"| Average Steps | {all_stats.get('avg_steps', 0):.2f} |\n"
        md_content += f"| Min Steps | {all_stats.get('min_steps', 0)} |\n"
        md_content += f"| Max Steps | {all_stats.get('max_steps', 0)} |\n\n"
        
        # Success attempts
        success_stats = step_stats.get("success_attempts", {})
        md_content += "### Success Attempts Only\n\n"
        md_content += "| Metric | Value |\n"
        md_content += "|--------|-------|\n"
        md_content += f"| Total Success Attempts | {success_stats.get('count', 0)} |\n"
        md_content += f"| Total Steps | {success_stats.get('total_steps', 0)} |\n"
        md_content += f"| Average Steps | {success_stats.get('avg_steps', 0):.2f} |\n"
        md_content += f"| Min Steps | {success_stats.get('min_steps', 0)} |\n"
        md_content += f"| Max Steps | {success_stats.get('max_steps', 0)} |\n\n"

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    except IOError as e:
        print(f"Warning: Failed to save final summary: {e}")


def print_realtime_progress(
    task_scope: List[Any],
    agent_scope: List[str],
    output_dir: str,
    max_attempts: int,
    trigger: str = "",
    save_to_file: bool = True,
) -> Dict[str, Any]:
    """
    Print real-time progress with comprehensive metrics and save to files.

    Displays and saves:
    - Task progress (executed/evaluated counts)
    - Pass@K for all tasks

    Files saved:
    - realtime_metrics.json: Current metrics snapshot (overwritten each time)
    - realtime_metrics_history.jsonl: History of all snapshots (appended)
    - final_metrics_summary.md: Markdown summary (overwritten each time)

    Args:
        task_scope: List of task namedtuples
        agent_scope: List of agent names
        output_dir: Base output directory
        max_attempts: Maximum number of attempts per task
        trigger: Description of what triggered this progress print
        save_to_file: Whether to save metrics to files

    Returns:
        Metrics snapshot dictionary
    """
    # Collect results
    data = collect_task_results(task_scope, agent_scope, output_dir, max_attempts)

    # Calculate metrics
    metrics = calculate_metrics(data)

    # Build snapshot
    snapshot = build_metrics_snapshot(data, metrics, trigger)

    # Save to files
    if save_to_file:
        save_realtime_metrics(output_dir, snapshot)
        append_metrics_history(output_dir, snapshot)
        generate_final_summary(output_dir, snapshot)

    # Extract for printing
    total_tasks = data["total_tasks"]
    executed_count = data["executed_count"]
    evaluated_count = data["evaluated_count"]
    pass_at_k = data["pass_at_k"]
    pass_at_k_evaluated = data["pass_at_k_evaluated"]  # Only for fully evaluated tasks

    # Print output
    print(f"\n{'=' * 80}")
    print(f"[*] REALTIME PROGRESS [{trigger}]")
    print(f"{'=' * 80}")

    # Time estimation
    elapsed = get_elapsed_time()
    elapsed_str = format_duration(elapsed)
    
    # Calculate ETA based on evaluated tasks (most accurate indicator)
    if evaluated_count > 0 and evaluated_count < total_tasks:
        avg_time_per_task = elapsed / evaluated_count
        remaining_tasks = total_tasks - evaluated_count
        eta_seconds = avg_time_per_task * remaining_tasks
        eta_str = format_duration(eta_seconds)
        total_estimated = elapsed + eta_seconds
        total_str = format_duration(total_estimated)
        print(f"[>] Time: Elapsed {elapsed_str} | ETA {eta_str} | Total Est. {total_str}")
    elif evaluated_count >= total_tasks:
        print(f"[>] Time: Elapsed {elapsed_str} | Completed!")
    else:
        print(f"[>] Time: Elapsed {elapsed_str} | ETA calculating...")

    # Progress info
    exec_pct = executed_count / total_tasks * 100 if total_tasks > 0 else 0
    eval_pct = evaluated_count / total_tasks * 100 if total_tasks > 0 else 0
    print(f"\n[>] Task Progress: Total {total_tasks}")
    print(
        f"    Executed: {executed_count}/{total_tasks} ({exec_pct:.1f}%) | Evaluated: {evaluated_count}/{total_tasks} ({eval_pct:.1f}%)"
    )

    # Pass@K - Overall (based on all tasks)
    print("\n[>] Pass@K (Overall - based on all tasks):")
    if total_tasks > 0:
        pass_str = " | ".join(
            [
                f"@{k}: {len(pass_at_k[k])}/{total_tasks}({len(pass_at_k[k]) / total_tasks * 100:.1f}%)"
                for k in range(1, max_attempts + 1)
            ]
        )
        print(f"    {pass_str}")
    else:
        print("    N/A")

    # Pass@K - Evaluated tasks only (use pass_at_k_evaluated to avoid >100% rates)
    print("\n[>] Pass@K (Evaluated only - based on evaluated tasks):")
    if evaluated_count > 0:
        pass_str_eval = " | ".join(
            [
                f"@{k}: {len(pass_at_k_evaluated[k])}/{evaluated_count}({len(pass_at_k_evaluated[k]) / evaluated_count * 100:.1f}%)"
                for k in range(1, max_attempts + 1)
            ]
        )
        print(f"    {pass_str_eval}")
    else:
        print("    N/A (no tasks evaluated yet)")

    # Success distribution
    print("\n[>] Success Distribution:")
    dist_str = " | ".join(
        [
            f"{i}x: {metrics['success_distribution'].get(i, 0)}"
            for i in range(max_attempts + 1)
        ]
    )
    print(f"    {dist_str}")
    
    # Step statistics
    step_stats = metrics.get("step_stats", {})
    if step_stats:
        all_stats = step_stats.get("all_attempts", {})
        success_stats = step_stats.get("success_attempts", {})
        
        print("\n[>] Agent Step Statistics:")
        print(f"    All Attempts: Total={all_stats.get('total_steps', 0)} steps | "
              f"Count={all_stats.get('count', 0)} | "
              f"Avg={all_stats.get('avg_steps', 0):.1f} steps/attempt")
        print(f"    Success Only: Total={success_stats.get('total_steps', 0)} steps | "
              f"Count={success_stats.get('count', 0)} | "
              f"Avg={success_stats.get('avg_steps', 0):.1f} steps/attempt")
    
    # Per-app metrics summary with Pass@1 and Pass@K
    app_metrics_data = metrics.get("app_metrics", {})
    if app_metrics_data:
        # Build summary with all pass@k rates
        app_summary = []
        for app_name, app_data in app_metrics_data.items():
            if app_data['evaluated'] > 0:
                pass_at_k_rates = app_data['pass_at_k_rates_evaluated']
                app_summary.append({
                    'name': app_name,
                    'pass_at_k': pass_at_k_rates,
                    'evaluated': app_data['evaluated'],
                    'total': app_data['total_tasks'],
                })

        # Sort by pass@1 rate descending, then by pass@max descending
        app_summary.sort(key=lambda x: (x['pass_at_k'].get(1, 0), x['pass_at_k'].get(max_attempts, 0)), reverse=True)

        # Build header: show Pass@1 and Pass@max (e.g., Pass@4)
        print(f"\n[>] Per-App Success Rate (Evaluated only, sorted by Pass@1):")
        header = f"    {'App':30s}  {'Pass@1':>8s}  {'Pass@' + str(max_attempts):>8s}  {'Evaluated':>12s}"
        print(header)
        print(f"    {'─' * 30}  {'─' * 8}  {'─' * 8}  {'─' * 12}")

        for app in app_summary:
            p1 = app['pass_at_k'].get(1, 0)
            pk = app['pass_at_k'].get(max_attempts, 0)
            print(f"    {app['name']:30s}  {p1:7.1f}%  {pk:7.1f}%  {app['evaluated']:>4d}/{app['total']:<4d}")

    print(f"{'=' * 80}\n")

    return snapshot

