#!/usr/bin/env python3
"""
MobileForge Rollout数据处理器 - 主入口脚本

这是一个完整的MobileForge rollout数据处理工具，包含以下功能：
1. 自动检测数据格式版本，支持新旧两种数据格式
2. 新版本：直接使用detailed_model_logs.json中的完整base64图像数据
3. 旧版本：修复图像错位问题的数据处理（图像占位符替换）
4. 支持并行处理和断点续传
5. 生成MobileForge GRPO训练格式数据（修复PyArrow兼容性）
6. 模块化设计，易于维护和扩展

数据格式版本说明：
- 新版本（推荐）：run.py运行后的detailed_model_logs.json直接包含完整的base64图像数据
- 占位符图像格式：detailed_model_logs.json包含图像占位符，需要从PNG文件重建图像数据

修复的核心问题：
- 自动适配最新的UITARS数据保存格式
- UI-TARS图像映射错位问题（占位符图像格式）
- 多轮对话中图像历史不匹配问题（占位符图像格式）
- PyArrow格式兼容性问题（content字段格式统一）
- 确保训练数据中的图像与实际执行时的截图一致

作者: MobileForge team
版本: v2.2 (自动版本检测 + 新版本数据格式支持)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from collections import defaultdict

# 添加core模块到路径
sys.path.append(str(Path(__file__).parent))

from core.processor import MobileForgeDataProcessor
from core.data_saver import MobileForgeDataSaver
from core.parallel_processor import MobileForgeParallelProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  可选的训练数据筛选函数
#  与 data_analyzer/filters.py 和 MobileForge training 中的
#  筛选逻辑保持一致，但全部为可选项
# ════════════════════════════════════════════════════════════════


def filter_loop_attempts(
    samples: List[Dict[str, Any]], threshold: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    死循环 Attempt 剔除：如果某个 attempt 中连续相同 action >= threshold，
    则剔除该 attempt 的所有步骤。

    Args:
        samples: 轨迹级样本列表（每个 sample 对应一个 attempt）
        threshold: 连续相同 action 的阈值

    Returns:
        (筛选后的样本列表, 被剔除的 attempt 数)
    """
    filtered = []
    removed = 0
    for s in samples:
        max_consecutive = s.get("metadata", {}).get("max_consecutive_same_actions", 0)
        if max_consecutive >= threshold:
            removed += 1
            logger.debug(
                f"  Loop removed: {s['task_id']}/{s.get('attempt', '?')} "
                f"(consecutive same actions: {max_consecutive})"
            )
        else:
            filtered.append(s)
    return filtered, removed


def filter_best_trajectory(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    最优轨迹选择：每个任务只保留一条最优的 attempt。
    优先级：1) overall_success=true  2) positive 步骤占比最高

    Args:
        samples: 轨迹级样本列表

    Returns:
        (筛选后的样本列表, 被剔除的 attempt 数)
    """
    # 按 task_id 分组
    tasks: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        tasks[s["task_id"]].append(s)

    filtered = []
    removed_total = 0

    for tid, attempts in tasks.items():
        if len(attempts) <= 1:
            filtered.extend(attempts)
            continue

        def score(s):
            success = 1 if s.get("evaluation_result") == 1 else 0
            step_details = s.get("metadata", {}).get("step_details", [])
            positive_count = sum(
                1 for sd in step_details if sd.get("impact") == "positive"
            )
            total = len(step_details) or 1
            return (success, positive_count / total)

        best = max(attempts, key=score)
        filtered.append(best)
        removed = len(attempts) - 1
        removed_total += removed
        logger.debug(
            f"  Best trajectory: {tid} - kept {best.get('attempt', '?')}, removed {removed}"
        )

    return filtered, removed_total


def filter_infeasible_tasks(
    samples: List[Dict[str, Any]], k: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Infeasible 任务剔除：如果同一任务中 infeasible 投票 >= k 次，
    则剔除该任务的所有 attempt。

    Args:
        samples: 轨迹级样本列表
        k: infeasible 投票阈值

    Returns:
        (筛选后的样本列表, 被剔除的 attempt 数)
    """
    tasks: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        tasks[s["task_id"]].append(s)

    filtered = []
    removed_total = 0

    for tid, attempts in tasks.items():
        infeasible_count = sum(
            1 for a in attempts if a.get("metadata", {}).get("task_feasible") is False
        )
        if infeasible_count >= k:
            removed_total += len(attempts)
            logger.debug(
                f"  Infeasible removed: {tid} (infeasible votes: {infeasible_count}/{len(attempts)})"
            )
        else:
            filtered.extend(attempts)

    return filtered, removed_total


def filter_by_sr_range(
    samples: List[Dict[str, Any]], sr_min: float, sr_max: float
) -> Tuple[List[Dict[str, Any]], int]:
    """
    SR 范围筛选：保留 avg_sr ∈ [sr_min, sr_max] 的任务，
    剔除 SR 超出范围的整个任务。

    注意：SR 在此处重新计算（而非直接使用元数据中的 avg_sr），
    以确保在前序筛选移除某些 attempt 后 SR 是准确的。

    Args:
        samples: 轨迹级样本列表
        sr_min: SR 下界
        sr_max: SR 上界

    Returns:
        (筛选后的样本列表, 被剔除的 attempt 数)
    """
    tasks: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        tasks[s["task_id"]].append(s)

    filtered = []
    removed_total = 0

    for tid, attempts in tasks.items():
        total = len(attempts)
        success_count = sum(1 for a in attempts if a.get("evaluation_result") == 1)
        sr = success_count / total if total > 0 else 0.0

        if sr_min <= sr <= sr_max:
            filtered.extend(attempts)
        else:
            removed_total += len(attempts)
            logger.debug(f"  SR removed: {tid} (SR={sr:.2f}, range=[{sr_min}, {sr_max}])")

    return filtered, removed_total


def filter_failed_attempts(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    删除所有 overall_success=false 的 attempt。

    Args:
        samples: 轨迹级样本列表

    Returns:
        (筛选后的样本列表, 被剔除的 attempt 数)
    """
    filtered = [s for s in samples if s.get("evaluation_result") == 1]
    return filtered, len(samples) - len(filtered)


def _count_steps(samples: List[Dict[str, Any]]) -> int:
    """统计样本列表中的总 step 数"""
    total = 0
    for s in samples:
        step_details = s.get("metadata", {}).get("step_details", [])
        total += len(step_details)
    return total


def apply_trajectory_filters(
    processed_data: Dict[str, Any],
    filter_loop_threshold: int = 0,
    filter_best_traj: bool = False,
    filter_infeasible_k: int = 0,
    filter_sr_min: float = -1.0,
    filter_sr_max: float = -1.0,
) -> Dict[str, Any]:
    """
    在轨迹级别对处理后的数据应用可选筛选。

    筛选顺序（与 MobileForge training 的 dataset.py 一致）：
    1. 死循环 Attempt 剔除
    2. 最优轨迹选择
    3. Infeasible 任务剔除
    4. SR 范围筛选

    Args:
        processed_data: process_all_tasks() 的返回结果
        filter_loop_threshold: 死循环剔除阈值（<= 0 不启用）
        filter_best_traj: 是否只保留最优轨迹
        filter_infeasible_k: infeasible 投票阈值（<= 0 不启用）
        filter_sr_min: SR 下界（< 0 不启用）
        filter_sr_max: SR 上界（< 0 不启用）

    Returns:
        筛选后的 processed_data（原地修改 all_samples 和正负样本分类）
    """
    samples = processed_data["all_samples"]
    original_count = len(samples)
    original_steps = _count_steps(samples)
    filter_stats = {}

    logger.info(
        f"Starting trajectory-level filtering: {original_count} attempts, {original_steps} steps"
    )

    # 1. Loop removal
    if filter_loop_threshold > 0:
        before_steps = _count_steps(samples)
        samples, removed = filter_loop_attempts(samples, filter_loop_threshold)
        after_steps = _count_steps(samples)
        filter_stats["loop_removed"] = removed
        filter_stats["loop_steps_removed"] = before_steps - after_steps
        logger.info(
            f"  [1] Loop removal (k>={filter_loop_threshold}): "
            f"removed {removed} attempts ({before_steps - after_steps} steps), "
            f"remaining {len(samples)} attempts ({after_steps} steps)"
        )

    # 2. Best trajectory selection
    if filter_best_traj:
        before_steps = _count_steps(samples)
        samples, removed = filter_best_trajectory(samples)
        after_steps = _count_steps(samples)
        filter_stats["best_traj_removed"] = removed
        filter_stats["best_traj_steps_removed"] = before_steps - after_steps
        logger.info(
            f"  [2] Best trajectory: "
            f"removed {removed} attempts ({before_steps - after_steps} steps), "
            f"remaining {len(samples)} attempts ({after_steps} steps)"
        )

    # 3. Infeasible task removal
    if filter_infeasible_k > 0:
        before_steps = _count_steps(samples)
        samples, removed = filter_infeasible_tasks(samples, filter_infeasible_k)
        after_steps = _count_steps(samples)
        filter_stats["infeasible_removed"] = removed
        filter_stats["infeasible_steps_removed"] = before_steps - after_steps
        logger.info(
            f"  [3] Infeasible removal (k>={filter_infeasible_k}): "
            f"removed {removed} attempts ({before_steps - after_steps} steps), "
            f"remaining {len(samples)} attempts ({after_steps} steps)"
        )

    # 4. SR range filtering
    if filter_sr_min >= 0 and filter_sr_max >= 0:
        before_steps = _count_steps(samples)
        samples, removed = filter_by_sr_range(samples, filter_sr_min, filter_sr_max)
        after_steps = _count_steps(samples)
        filter_stats["sr_removed"] = removed
        filter_stats["sr_steps_removed"] = before_steps - after_steps
        logger.info(
            f"  [4] SR range [{filter_sr_min}, {filter_sr_max}]: "
            f"removed {removed} attempts ({before_steps - after_steps} steps), "
            f"remaining {len(samples)} attempts ({after_steps} steps)"
        )

    remaining_steps = _count_steps(samples)
    filter_stats["original_attempts"] = original_count
    filter_stats["original_steps"] = original_steps
    filter_stats["remaining_attempts"] = len(samples)
    filter_stats["remaining_steps"] = remaining_steps
    filter_stats["total_removed"] = original_count - len(samples)
    filter_stats["total_steps_removed"] = original_steps - remaining_steps

    logger.info(
        f"Trajectory filtering done: {original_count} attempts ({original_steps} steps) -> "
        f"{len(samples)} attempts ({remaining_steps} steps) "
        f"(removed {original_count - len(samples)} attempts, {original_steps - remaining_steps} steps)"
    )

    # Re-classify positive/negative samples
    positive_samples = []
    negative_samples = []
    for s in samples:
        if s.get("success") and s.get("evaluation_result") == 1:
            positive_samples.append(s)
        elif s.get("evaluation_result") == 0:
            negative_samples.append(s)

    processed_data["all_samples"] = samples
    processed_data["positive_samples"] = positive_samples
    processed_data["negative_samples"] = negative_samples
    processed_data["filter_stats"] = filter_stats

    return processed_data


def process_single_thread(
    rollout_dir: Union[str, List[str]],
    output_dir: str,
    max_tasks: Optional[int] = None,
    format_type: str = "grpo",
    positive_only: bool = False,
    filter_loop_threshold: int = 0,
    filter_best_traj: bool = False,
    filter_infeasible_k: int = 0,
    filter_sr_min: float = -1.0,
    filter_sr_max: float = -1.0,
    max_steps: int = 0,
    remove_evaluation_hints: bool = False,
) -> Dict[str, Any]:
    """
    单线程处理数据

    Args:
        rollout_dir: rollout结果目录或目录列表
        output_dir: 输出目录
        max_tasks: 最大处理任务数（均衡选择）
        format_type: 输出格式
        positive_only: 是否仅保留 impact=positive 的步骤
        filter_loop_threshold: 死循环剔除阈值（<= 0 不启用）
        filter_best_traj: 是否只保留最优轨迹
        filter_infeasible_k: infeasible 投票阈值（<= 0 不启用）
        filter_sr_min: SR 下界（< 0 不启用）
        filter_sr_max: SR 上界（< 0 不启用）
        max_steps: 最终数据集最大步骤数（<= 0 不限制），超出时按 app+action_type 均衡采样
        remove_evaluation_hints: 是否从 user prompt 中删除 EVALUATION HINTS 块

    Returns:
        处理结果字典
    """
    logger.info("Starting single-thread processing...")

    # 创建处理器（支持多目录）
    processor = MobileForgeDataProcessor(rollout_dir, output_dir)

    # 处理数据
    processed_data = processor.process_all_tasks(max_tasks=max_tasks)

    # ── 可选的轨迹级筛选 ──
    need_filter = (
        (filter_loop_threshold > 0)
        or filter_best_traj
        or (filter_infeasible_k > 0)
        or (filter_sr_min >= 0 and filter_sr_max >= 0)
    )
    if need_filter:
        processed_data = apply_trajectory_filters(
            processed_data,
            filter_loop_threshold=filter_loop_threshold,
            filter_best_traj=filter_best_traj,
            filter_infeasible_k=filter_infeasible_k,
            filter_sr_min=filter_sr_min,
            filter_sr_max=filter_sr_max,
        )

    # 创建数据保存器并保存结果
    data_saver = MobileForgeDataSaver(output_dir)
    saved_files = data_saver.save_training_data(
        processed_data,
        format_type=format_type,
        positive_only=positive_only,
        max_steps=max_steps,
        remove_evaluation_hints=remove_evaluation_hints,
    )

    # 获取步骤级统计信息用于保存到 session_summary
    step_level_stats = saved_files.get("step_level_stats", None)
    data_saver.save_session_summary(processed_data, step_level_stats=step_level_stats)

    # 保存错误分析JSON
    error_analysis_file = data_saver.save_error_analysis(processed_data)
    if error_analysis_file:
        saved_files["error_analysis"] = error_analysis_file

    # 记录会话信息
    processed_data["session_info"] = data_saver.get_session_info()
    processed_data["saved_files"] = saved_files

    logger.info("Single-thread processing complete")
    return processed_data


def print_step_level_stats(step_stats: Dict[str, Any]) -> None:
    """Print step-level statistics"""
    if not step_stats or "error" in step_stats:
        print(f"\n--- Step-Level Stats ---")
        print(f"Unable to compute step-level statistics")
        return

    total = step_stats.get("total_samples", 0)
    all_stats = step_stats.get("all_samples", {})

    print(f"\n" + "=" * 70)
    print(f"Step-Level Stats (total {total} step samples)")
    print("=" * 70)

    # bad_step stats (based on impact != 'positive')
    bad_step = all_stats.get("bad_step", {})
    print(f"\n[bad_step] (criteria: impact != 'positive')")
    print(
        f"  positive (bad_step=false): {bad_step.get('false', 0):>5} ({bad_step.get('false_pct', 0):>6.2f}%)"
    )
    print(
        f"  negative (bad_step=true):  {bad_step.get('true', 0):>5} ({bad_step.get('true_pct', 0):>6.2f}%)"
    )

    # impact stats
    impact = all_stats.get("impact", {})
    print(f"\n[impact]")
    print(
        f"  positive: {impact.get('positive', 0):>5} ({impact.get('positive_pct', 0):>6.2f}%)"
    )
    print(
        f"  negative: {impact.get('negative', 0):>5} ({impact.get('negative_pct', 0):>6.2f}%)"
    )
    print(
        f"  neutral:  {impact.get('neutral', 0):>5} ({impact.get('neutral_pct', 0):>6.2f}%)"
    )
    print(
        f"  unknown:  {impact.get('unknown', 0):>5} ({impact.get('unknown_pct', 0):>6.2f}%)"
    )

    # overall_success stats
    overall_success = all_stats.get("overall_success", {})
    print(f"\n[overall_success]")
    print(
        f"  true:  {overall_success.get('true', 0):>5} ({overall_success.get('true_pct', 0):>6.2f}%)"
    )
    print(
        f"  false: {overall_success.get('false', 0):>5} ({overall_success.get('false_pct', 0):>6.2f}%)"
    )

    # Filtered stats (excluding easy and hard tasks)
    filter_info = step_stats.get("filter_info", {})
    filtered_stats = step_stats.get("filtered_samples", {})

    if filtered_stats and "error" not in filtered_stats:
        print(f"\n" + "-" * 70)
        print(
            f"Filtered Stats (excluded {filter_info.get('easy_tasks_excluded', 0)} easy tasks + {filter_info.get('hard_tasks_excluded', 0)} hard tasks)"
        )
        print(f"Remaining samples: {filter_info.get('remaining_samples', 0)}")
        print("-" * 70)

        filtered_total = filtered_stats.get("total_samples", 0)
        if filtered_total > 0:
            # bad_step
            f_bad_step = filtered_stats.get("bad_step", {})
            print(f"\n[bad_step] (filtered)")
            print(
                f"  positive: {f_bad_step.get('false', 0):>5} ({f_bad_step.get('false_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {f_bad_step.get('true', 0):>5} ({f_bad_step.get('true_pct', 0):>6.2f}%)"
            )

            # impact
            f_impact = filtered_stats.get("impact", {})
            print(f"\n[impact] (filtered)")
            print(
                f"  positive: {f_impact.get('positive', 0):>5} ({f_impact.get('positive_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {f_impact.get('negative', 0):>5} ({f_impact.get('negative_pct', 0):>6.2f}%)"
            )
            print(
                f"  neutral:  {f_impact.get('neutral', 0):>5} ({f_impact.get('neutral_pct', 0):>6.2f}%)"
            )
            print(
                f"  unknown:  {f_impact.get('unknown', 0):>5} ({f_impact.get('unknown_pct', 0):>6.2f}%)"
            )

    # Stats after excluding error tasks
    clean_info = step_stats.get("clean_info", {})
    clean_stats = step_stats.get("clean_samples", {})

    if clean_stats and "error" not in clean_stats:
        print(f"\n" + "=" * 70)
        print(
            f"Clean Stats (excluded {clean_info.get('error_tasks_excluded', 0)} tasks with error_trajectories)"
        )
        print(f"Remaining tasks: {clean_info.get('clean_tasks_count', 0)}")
        print(f"Remaining samples: {clean_info.get('remaining_samples', 0)}")
        print(
            f"(easy tasks: {clean_info.get('clean_easy_tasks', 0)}, hard tasks: {clean_info.get('clean_hard_tasks', 0)})"
        )
        print("=" * 70)

        clean_total = clean_stats.get("total_samples", 0)
        if clean_total > 0:
            # bad_step
            c_bad_step = clean_stats.get("bad_step", {})
            print(f"\n[bad_step] (clean)")
            print(
                f"  positive: {c_bad_step.get('false', 0):>5} ({c_bad_step.get('false_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {c_bad_step.get('true', 0):>5} ({c_bad_step.get('true_pct', 0):>6.2f}%)"
            )

            # impact
            c_impact = clean_stats.get("impact", {})
            print(f"\n[impact] (clean)")
            print(
                f"  positive: {c_impact.get('positive', 0):>5} ({c_impact.get('positive_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {c_impact.get('negative', 0):>5} ({c_impact.get('negative_pct', 0):>6.2f}%)"
            )
            print(
                f"  neutral:  {c_impact.get('neutral', 0):>5} ({c_impact.get('neutral_pct', 0):>6.2f}%)"
            )
            print(
                f"  unknown:  {c_impact.get('unknown', 0):>5} ({c_impact.get('unknown_pct', 0):>6.2f}%)"
            )

            # overall_success
            c_overall = clean_stats.get("overall_success", {})
            print(f"\n[overall_success] (clean)")
            print(
                f"  true:  {c_overall.get('true', 0):>5} ({c_overall.get('true_pct', 0):>6.2f}%)"
            )
            print(
                f"  false: {c_overall.get('false', 0):>5} ({c_overall.get('false_pct', 0):>6.2f}%)"
            )

    # Stats after excluding error tasks + keeping only medium difficulty
    medium_clean_info = step_stats.get("medium_clean_info", {})
    medium_clean_stats = step_stats.get("medium_clean_samples", {})

    if medium_clean_stats and "error" not in medium_clean_stats:
        print(f"\n" + "=" * 70)
        print(
            f"Medium Difficulty Stats (excluded {medium_clean_info.get('error_tasks_excluded', 0)} error + "
            f"{medium_clean_info.get('easy_tasks_excluded', 0)} easy + "
            f"{medium_clean_info.get('hard_tasks_excluded', 0)} hard tasks)"
        )
        print(f"Remaining tasks: {medium_clean_info.get('medium_clean_tasks_count', 0)}")
        print(f"Remaining samples: {medium_clean_info.get('remaining_samples', 0)}")
        print("=" * 70)

        mc_total = medium_clean_stats.get("total_samples", 0)
        if mc_total > 0:
            # bad_step
            mc_bad_step = medium_clean_stats.get("bad_step", {})
            print(f"\n[bad_step] (medium difficulty)")
            print(
                f"  positive: {mc_bad_step.get('false', 0):>5} ({mc_bad_step.get('false_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {mc_bad_step.get('true', 0):>5} ({mc_bad_step.get('true_pct', 0):>6.2f}%)"
            )

            # impact
            mc_impact = medium_clean_stats.get("impact", {})
            print(f"\n[impact] (medium difficulty)")
            print(
                f"  positive: {mc_impact.get('positive', 0):>5} ({mc_impact.get('positive_pct', 0):>6.2f}%)"
            )
            print(
                f"  negative: {mc_impact.get('negative', 0):>5} ({mc_impact.get('negative_pct', 0):>6.2f}%)"
            )
            print(
                f"  neutral:  {mc_impact.get('neutral', 0):>5} ({mc_impact.get('neutral_pct', 0):>6.2f}%)"
            )
            print(
                f"  unknown:  {mc_impact.get('unknown', 0):>5} ({mc_impact.get('unknown_pct', 0):>6.2f}%)"
            )

            # overall_success
            mc_overall = medium_clean_stats.get("overall_success", {})
            print(f"\n[overall_success] (medium difficulty)")
            print(
                f"  true:  {mc_overall.get('true', 0):>5} ({mc_overall.get('true_pct', 0):>6.2f}%)"
            )
            print(
                f"  false: {mc_overall.get('false', 0):>5} ({mc_overall.get('false_pct', 0):>6.2f}%)"
            )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MobileForge Rollout Data Processor v2.2 - with optional data filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing (single directory)
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout

  # Multi-directory merge
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout1 /path/to/rollout2 /path/to/rollout3

  # Parallel processing
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout --parallel --max_workers 8

  # Limit task count (balanced selection, prefer fully evaluated tasks)
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout --max_tasks 600

  # Enable filtering (positive-only + loop removal)
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout \\
      --positive_only --filter_loop_threshold 7

  # Limit final step count (balanced by app + action_type)
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout \\
      --positive_only --filter_loop_threshold 7 --max_steps 5000

  # Remove EVALUATION HINTS (for training without hints)
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout \\
      --positive_only --filter_loop_threshold 7 --remove_evaluation_hints

  # All filters combined
  python mobileforge_data_processor.py --rollout_dir /path/to/rollout \\
      --positive_only --filter_loop_threshold 7 \\
      --filter_best_trajectory --filter_infeasible_k 3 \\
      --filter_sr_min 0.0 --filter_sr_max 0.75 --max_steps 10000 \\
      --remove_evaluation_hints

Output structure (all files in timestamped subdirectory):
  processed_data/
  └── session_YYYYMMDD_HHMMSS/
      ├── mobileforge_grpo_*.json        # GRPO training data (pos+neg samples)
      ├── mobileforge_grpo_stats_*.json  # Statistics
      ├── session_summary.json        # Session summary
      ├── README.md
      └── temp_results/               # Parallel intermediate results (if any)

Multi-directory support:
  - Process multiple rollout result directories simultaneously
  - Automatically merge task data from all directories
  - Generate unified training data files

max_tasks balanced selection strategy:
  - Distribute quota evenly across different apps
  - Within each app, prefer fully evaluated tasks (all attempts have final_decision.json)
  - If insufficient, supplement with partially evaluated tasks to maintain app balance

Filter execution order:
  1. Loop attempt removal (--filter_loop_threshold)
  2. Best trajectory selection (--filter_best_trajectory)
  3. Infeasible task removal (--filter_infeasible_k)
  4. SR range filtering (--filter_sr_min / --filter_sr_max)
  5. Positive-only steps (--positive_only)
  6. Remove EVALUATION HINTS blocks (--remove_evaluation_hints)
  7. Max steps limit (--max_steps): balanced by app + action_type
        """,
    )

    # ── Basic arguments ──
    parser.add_argument(
        "--rollout_dir",
        required=True,
        nargs="+",
        help="Rollout result directory (supports multiple paths)",
    )
    parser.add_argument("--output_dir", default="processed_data", help="Output directory")
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Max tasks to process (balanced selection, default: no limit)",
    )
    parser.add_argument(
        "--format", choices=["grpo", "r1v"], default="grpo", help="Output format"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False, help="Enable parallel processing"
    )
    parser.add_argument("--max_workers", type=int, default=400, help="Parallel worker count")
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save interval for parallel processing"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        default=False,
        help="Do not resume from checkpoint (start fresh)",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test mode (process first 3 tasks only)"
    )

    # ── Optional filter arguments (all off by default) ──
    filter_group = parser.add_argument_group(
        "Optional Filters",
        "All filters are off by default. Enable as needed, any combination is supported.",
    )
    filter_group.add_argument(
        "--positive_only",
        action="store_true",
        default=False,
        help="Keep only impact=positive steps (default: off)",
    )
    filter_group.add_argument(
        "--filter_loop_threshold",
        type=int,
        default=0,
        help="Loop removal threshold: remove attempts with >= N consecutive same actions (0=off, recommended: 7)",
    )
    filter_group.add_argument(
        "--filter_best_trajectory",
        action="store_true",
        default=False,
        help="Keep only the best trajectory per task (default: off)",
    )
    filter_group.add_argument(
        "--filter_infeasible_k",
        type=int,
        default=0,
        help="Infeasible task removal: remove entire task if infeasible votes >= K (0=off)",
    )
    filter_group.add_argument(
        "--filter_sr_min",
        type=float,
        default=-1.0,
        help="SR range lower bound: keep tasks with avg_sr >= sr_min (<0=off)",
    )
    filter_group.add_argument(
        "--filter_sr_max",
        type=float,
        default=-1.0,
        help="SR range upper bound: keep tasks with avg_sr <= sr_max (<0=off)",
    )
    filter_group.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Max steps in final dataset (0=no limit). Balanced by app + action_type when exceeded",
    )
    filter_group.add_argument(
        "--remove_evaluation_hints",
        action="store_true",
        default=False,
        help="Remove EVALUATION HINTS FROM PREVIOUS ATTEMPTS blocks from user prompts (default: off)",
    )

    args = parser.parse_args()
    print(f"Input dirs: {args.rollout_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Format: {args.format}")
    print(f"Parallel: {'yes' if args.parallel else 'no'}")
    if args.parallel:
        print(f"Max workers: {args.max_workers}")
        print(f"Save interval: {args.save_interval}")
        print(f"Resume: {'no' if args.no_resume else 'yes'}")
    if args.max_tasks:
        print(f"Max tasks: {args.max_tasks} (balanced selection)")
    if args.test:
        print("Test mode: yes")

    # Print filter config
    any_filter = (
        args.positive_only
        or args.filter_loop_threshold > 0
        or args.filter_best_trajectory
        or args.filter_infeasible_k > 0
        or (args.filter_sr_min >= 0 and args.filter_sr_max >= 0)
        or args.max_steps > 0
        or args.remove_evaluation_hints
    )
    if any_filter:
        print(f"\n--- Filter Config ---")
        print(f"  positive_only:             {args.positive_only}")
        print(
            f"  filter_loop_threshold:     {args.filter_loop_threshold} {'(on)' if args.filter_loop_threshold > 0 else '(off)'}"
        )
        print(f"  filter_best_trajectory:    {args.filter_best_trajectory}")
        print(
            f"  filter_infeasible_k:       {args.filter_infeasible_k} {'(on)' if args.filter_infeasible_k > 0 else '(off)'}"
        )
        sr_enabled = args.filter_sr_min >= 0 and args.filter_sr_max >= 0
        print(
            f"  filter_sr_range:           [{args.filter_sr_min}, {args.filter_sr_max}] {'(on)' if sr_enabled else '(off)'}"
        )
        print(
            f"  max_steps:                 {args.max_steps} {'(on, balanced by app+action_type)' if args.max_steps > 0 else '(off)'}"
        )
        print(f"  remove_evaluation_hints:   {args.remove_evaluation_hints}")
    else:
        print(f"\nFilters: none enabled")

    print("-" * 70)

    # Validate input (supports multiple paths)
    rollout_dirs = args.rollout_dir
    if isinstance(rollout_dirs, str):
        rollout_dirs = [rollout_dirs]

    # Validate all input paths
    for rollout_dir in rollout_dirs:
        if not os.path.exists(rollout_dir):
            print(f"Error: rollout directory not found: {rollout_dir}")
            return 1

    print(f"Found {len(rollout_dirs)} input directories")
    for i, rollout_dir in enumerate(rollout_dirs, 1):
        print(f"  {i}. {rollout_dir}")

    try:
        start_time = time.time()

        if args.test:
            args.max_tasks = 3

        # 使用单线程处理（已修复并兼容MobileForge格式）
        processed_data = process_single_thread(
            rollout_dir=rollout_dirs,
            output_dir=args.output_dir,
            max_tasks=args.max_tasks,
            format_type=args.format,
            positive_only=args.positive_only,
            filter_loop_threshold=args.filter_loop_threshold,
            filter_best_traj=args.filter_best_trajectory,
            filter_infeasible_k=args.filter_infeasible_k,
            filter_sr_min=args.filter_sr_min,
            filter_sr_max=args.filter_sr_max,
            max_steps=args.max_steps,
            remove_evaluation_hints=args.remove_evaluation_hints,
        )

        processing_time = time.time() - start_time

        # Print summary
        print("\n" + "=" * 70)
        print("Processing Summary")
        print("=" * 70)
        stats = processed_data["statistics"]
        print(f"Input directories: {stats.get('input_directories', 1)}")
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"Processed tasks: {stats['processed_tasks']}")
        print(f"Successful trajectories: {stats['successful_trajectories']}")
        print(f"Failed trajectories: {stats['failed_trajectories']}")
        print(f"Error trajectories: {stats['error_trajectories']}")
        print(f"Image mapping fixes: {stats.get('image_mapping_fixes', 0)}")
        print(f"Placeholder replacements: {stats.get('placeholder_replacements', 0)}")

        # Filter stats
        if "filter_stats" in processed_data:
            fs = processed_data["filter_stats"]
            print(f"\n--- Filter Stats ---")
            print(
                f"Before: {fs.get('original_attempts', 0)} attempts, {fs.get('original_steps', 0)} steps"
            )
            if "loop_removed" in fs:
                print(
                    f"  Loop removed:      {fs['loop_removed']} attempts, {fs.get('loop_steps_removed', 0)} steps"
                )
            if "best_traj_removed" in fs:
                print(
                    f"  Best traj removed: {fs['best_traj_removed']} attempts, {fs.get('best_traj_steps_removed', 0)} steps"
                )
            if "infeasible_removed" in fs:
                print(
                    f"  Infeasible:        {fs['infeasible_removed']} attempts, {fs.get('infeasible_steps_removed', 0)} steps"
                )
            if "sr_removed" in fs:
                print(
                    f"  SR range removed:  {fs['sr_removed']} attempts, {fs.get('sr_steps_removed', 0)} steps"
                )
            print(
                f"After:  {fs.get('remaining_attempts', 0)} attempts, {fs.get('remaining_steps', 0)} steps"
            )
            print(
                f"Total removed: {fs.get('total_removed', 0)} attempts, {fs.get('total_steps_removed', 0)} steps"
            )

        # Balanced selection stats
        if "balanced_selection_stats" in processed_data.get("saved_files", {}):
            bs = processed_data["saved_files"]["balanced_selection_stats"]
            print(f"\n--- Balanced Selection Stats (max_steps={bs.get('max_steps', 0)}) ---")
            print(f"Before: {bs.get('before_count', 0)} steps")
            print(f"After:  {bs.get('after_count', 0)} steps")
            print(f"Apps: {bs.get('num_apps', 0)}")
            print(f"Action types: {bs.get('num_action_types', 0)}")
            # Per-app distribution
            app_dist = bs.get("app_distribution", {})
            if app_dist:
                print(f"  Per-app distribution:")
                for app_name in sorted(app_dist.keys()):
                    info = app_dist[app_name]
                    print(
                        f"    {app_name}: {info['selected']}/{info['available']} steps"
                    )
            # Per-action_type distribution
            at_dist = bs.get("action_type_distribution", {})
            if at_dist:
                print(f"  Per-action_type distribution:")
                for at_name in sorted(at_dist.keys()):
                    info = at_dist[at_name]
                    print(
                        f"    {at_name}: {info['selected']}/{info['available']} steps"
                    )

        if args.positive_only:
            print(f"\nNote: positive_only=True, GRPO output only contains impact=positive steps")

        # Hint stats
        hint_stats = processed_data.get("saved_files", {}).get("hint_stats")
        if hint_stats:
            total_scanned = hint_stats.get("total_steps_scanned", 0)
            with_hint_total = hint_stats.get("steps_with_hint_total", 0)
            with_hint_kept = hint_stats.get("steps_with_hint_kept", 0)
            hints_removed = hint_stats.get("hints_removed", 0)
            pct_total = (with_hint_total / total_scanned * 100) if total_scanned else 0
            print(f"\n--- Evaluation Hints Stats ---")
            print(
                f"Steps with hint in raw data: {with_hint_total}/{total_scanned} ({pct_total:.1f}%)"
            )
            print(f"Steps with hint in final output: {with_hint_kept}")
            if args.remove_evaluation_hints:
                print(f"Hints removed from steps:        {hints_removed}")
                print(
                    f"Note: remove_evaluation_hints=True, EVALUATION HINTS blocks removed from user prompts"
                )
            else:
                print(
                    f"Note: remove_evaluation_hints=False, hint blocks retained in user prompts"
                )

        print(f"\n--- Task Difficulty Distribution ---")
        print(f"Easy tasks (all 4 correct):  {stats.get('easy_tasks', 0)}")
        print(f"Pass@1 tasks:                {stats.get('pass1', 0)}")
        print(f"Pass@2 tasks:                {stats.get('pass2', 0)}")
        print(f"Pass@3 tasks:                {stats.get('pass3', 0)}")
        print(f"Hard tasks (all 4 failed):   {stats.get('hard_tasks', 0)}")

        print(f"\n--- Trajectory-Level Samples ---")
        print(f"Positive trajectories: {len(processed_data['positive_samples'])}")
        print(f"Negative trajectories: {len(processed_data['negative_samples'])}")
        print(f"Total trajectories:    {len(processed_data['all_samples'])}")

        # 打印步骤级统计信息
        if (
            "saved_files" in processed_data
            and "step_level_stats" in processed_data["saved_files"]
        ):
            step_stats = processed_data["saved_files"]["step_level_stats"]
            print_step_level_stats(step_stats)

        print(f"\nProcessing time: {processing_time:.2f}s")
        if stats["processed_tasks"] > 0:
            print(f"Avg per task: {processing_time / stats['processed_tasks']:.2f}s")

        # Session info
        if "session_info" in processed_data:
            session_info = processed_data["session_info"]
            print(f"\nSession info:")
            print(
                f"  Timestamp: {session_info.get('session_timestamp', session_info.get('timestamp'))}"
            )
            print(f"  Output dir: {session_info.get('session_dir')}")

        print(f"\nResults saved to: {args.output_dir}")
        print("Done! Image mapping issues fixed.")

        print(f"\nUsage tips:")
        if "session_info" in processed_data:
            session_dir = processed_data["session_info"].get("session_dir")
            session_timestamp = processed_data["session_info"].get("timestamp")

            print(f"All training files saved in timestamped subdirectory:")
            print(f"  Session dir: {session_dir}")
            print(
                f"  Files: mobileforge_grpo_{session_timestamp}.json (merged pos/neg samples, sorted by task/attempt/step)"
            )
            print(f"         mobileforge_grpo_stats_{session_timestamp}.json")
            print(f"         session_summary.json")
            print(
                f"         error_analysis_{session_timestamp}.json (error task/trajectory details)"
            )
            print(f"         README.md")

            if "saved_files" in processed_data:
                saved_files = processed_data["saved_files"]
                grpo_file = saved_files.get("grpo_data", "")
                error_file = saved_files.get("error_analysis", "")

                if grpo_file:
                    print(f"\nMobileForge GRPO training command:")
                    print(f"cd /path/to/MobileForge")
                    print(f"torchrun --nproc_per_node=8 src/open_r1/grpo_gui.py \\")
                    print(f"  --dataset_name {Path(grpo_file).absolute()} \\")
                    print(f"  --model_name_or_path your_model_path \\")
                    print(f"  --output_dir ./grpo_output \\")
                    print(f"  # ... other args")

                if error_file and stats.get("error_trajectories", 0) > 0:
                    print(f"\nError analysis file generated:")
                    print(f"  {error_file}")
                    print(f"  - {stats.get('error_trajectories', 0)} error trajectories")
                    print(f"  - {len(stats.get('error_task_ids', []))} tasks involved")
        else:
            print(f"\nUsage tips:")
            print(f"All training files saved in timestamped subdirectory")
            print(f"Check the latest session_* folder in {args.output_dir}")

        return 0

    except Exception as e:
        print(f"Error: processing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
