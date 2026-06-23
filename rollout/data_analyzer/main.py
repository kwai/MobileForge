"""
MobileForge 数据分析器 CLI 入口

用法示例:
  # 基线分析 (不加任何筛选)
  python -m data_analyzer --rollout_dir /path/to/session

  # 加筛选条件
  python -m data_analyzer --rollout_dir /path/to/session \
      --remove_infeasible 2 --sr_range 0 0.75 --remove_loops 3 \
      --success_only --best_trajectory --positive_only --remove_errors

  # 多目录 + 限制任务数
  python -m data_analyzer --rollout_dir /dir1 /dir2 --max_tasks 50

  # 按 App 筛选
  python -m data_analyzer --rollout_dir /path/to/session --app_filter Clock

  # 指定输出目录
  python -m data_analyzer --rollout_dir /path/to/session -o my_reports
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from .loader import RolloutDataLoader
from .metrics import MetricsComputer
from .filters import DataFilter
from .report import HTMLReportGenerator

logger = logging.getLogger(__name__)

BANNER = """
======================================================================
MobileForge Rollout 数据分析器 v2.0 (交互式 HTML 仪表板)
======================================================================"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MobileForge 数据分析器 — 生成可交互 HTML 仪表板",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rollout_dir", nargs="+", required=True,
                    help="rollout 结果目录 (可多个)")
    # 默认输出到 data_analyzer/analysis_reports 下
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_reports")
    p.add_argument("-o", "--output_dir", default=default_output,
                    help="输出目录 (默认 data_analyzer/analysis_reports)")
    p.add_argument("--max_tasks", type=int, default=None,
                    help="最多加载的任务数")
    p.add_argument("--app_filter", type=str, default=None,
                    help="仅分析指定 App 的任务")

    g = p.add_argument_group("筛选策略 (预计算)")
    g.add_argument("--best_trajectory", action="store_true",
                   help="筛选1: 每任务仅保留最优轨迹")
    g.add_argument("--remove_infeasible", type=int, default=0, metavar="K",
                   help="筛选2: infeasible 投票 ≥ K 则剔除任务 (0=不启用)")
    g.add_argument("--sr_range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                   help="筛选3: 保留 avg_sr ∈ [MIN, MAX]")
    g.add_argument("--positive_only", action="store_true",
                   help="筛选4: 仅保留 positive steps")
    g.add_argument("--success_only", action="store_true",
                   help="筛选5: 仅保留成功 attempts")
    g.add_argument("--remove_loops", type=int, default=0, metavar="K",
                   help="筛选6: 连续 ≥ K 次相同 action → 剔除 attempt (0=不启用)")
    g.add_argument("--remove_errors", action="store_true",
                   help="筛选7: 删除评估异常 attempts")
    g.add_argument("--step_range", nargs=2, type=int, default=None, metavar=("MIN", "MAX"),
                   help="筛选8: 仅保留步骤数 ∈ [MIN, MAX] 的 attempt")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)

    print(BANNER)
    print(f"输入目录: {args.rollout_dir}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 70)

    # ── 1. 加载数据 ──
    loader = RolloutDataLoader(args.rollout_dir, max_tasks=args.max_tasks)
    tasks = loader.load_all()
    if not tasks:
        print("❌ 未加载到任何任务数据")
        return 1

    # App 筛选
    if args.app_filter:
        tasks = {k: v for k, v in tasks.items() if v.get("app") == args.app_filter}
        print(f"App 筛选: {args.app_filter}, 剩余 {len(tasks)} 个任务")
        if not tasks:
            print("❌ 指定 App 下无任务")
            return 1

    # ── 2. 基线指标 ──
    print("\n计算基线指标...")
    baseline = MetricsComputer.compute(tasks, label="基线 (无筛选)")
    per_app = MetricsComputer.compute_per_app(tasks)

    # ── 3. 预计算: 各筛选策略单独效果 ──
    print("分析各筛选策略影响...")
    filter_sections = _compute_filter_sections(tasks, args)

    # ── 4. 生成 HTML 仪表板 ──
    print("\n生成交互式 HTML 仪表板...")
    cli_str = " ".join(sys.argv) if argv is None else " ".join(argv)
    reporter = HTMLReportGenerator(args.output_dir)
    html_path = reporter.generate(
        tasks=tasks,
        baseline=baseline,
        per_app=per_app,
        filter_sections=filter_sections,
        cli_args=cli_str,
    )

    # 同时保存 JSON 数据
    reporter.save_json({
        "baseline": baseline,
        "per_app": per_app,
        "filter_sections": [
            {"name": f["name"], "desc": f["desc"],
             "metrics": f["metrics"], "examples": f.get("examples", [])[:10]}
            for f in filter_sections
        ],
    })

    print(f"\n{'=' * 70}")
    print(f"✅ 仪表板: {html_path}")
    print(f"   报告目录: {reporter.report_dir}")
    print(f"{'=' * 70}")
    print(f"\n💡 提示: 在浏览器中打开 {html_path} 查看交互式分析报告\n")
    return 0


def _compute_filter_sections(tasks: Dict, args) -> List[Dict]:
    """预计算各筛选策略的独立效果"""
    sections = []

    # 筛选 7: 删除评估异常 (总是计算)
    f, ex = DataFilter.remove_errors(tasks)
    sections.append({
        "name": "筛选7: 删除评估异常",
        "desc": "删除 final_result ∉ {0,1} 的异常 attempt",
        "metrics": MetricsComputer.compute(f, "remove_errors"),
        "examples": ex,
    })

    # 筛选 1: 最优轨迹
    f, ex = DataFilter.best_trajectory(tasks)
    sections.append({
        "name": "筛选1: 最优轨迹选择",
        "desc": "每个任务仅保留一条最优轨迹 (优先成功, 其次 positive 占比)",
        "metrics": MetricsComputer.compute(f, "best_trajectory"),
        "examples": ex,
    })

    # 筛选 2: Infeasible
    for k in [1, 2, 3]:
        f, ex = DataFilter.remove_infeasible(tasks, threshold=k)
        sections.append({
            "name": f"筛选2: Infeasible 剔除 (k={k})",
            "desc": f"infeasible 投票 ≥ {k} → 剔除整个任务",
            "metrics": MetricsComputer.compute(f, f"infeasible_k{k}"),
            "examples": ex,
        })

    # 筛选 3: SR 范围
    for lo, hi in [(0, 0.5), (0.25, 0.75), (0, 1)]:
        f, ex = DataFilter.filter_by_sr(tasks, lo, hi)
        sections.append({
            "name": f"筛选3: SR ∈ [{lo}, {hi}]",
            "desc": f"保留 avg_sr ∈ [{lo}, {hi}] 的任务",
            "metrics": MetricsComputer.compute(f, f"sr_{lo}_{hi}"),
            "examples": ex,
        })

    # 筛选 4: 仅 positive steps
    f, ex = DataFilter.positive_steps_only(tasks)
    sections.append({
        "name": "筛选4: 仅 Positive Steps",
        "desc": "删除所有 impact ≠ positive 的步骤",
        "metrics": MetricsComputer.compute(f, "positive_only"),
        "examples": ex,
    })

    # 筛选 5: 仅成功 attempts
    f, ex = DataFilter.success_only(tasks)
    sections.append({
        "name": "筛选5: 仅成功 Attempts",
        "desc": "删除 overall_success=false 的 attempt",
        "metrics": MetricsComputer.compute(f, "success_only"),
        "examples": ex,
    })

    # 筛选 6: 死循环
    for k in [3, 5, 7]:
        f, ex = DataFilter.remove_loops(tasks, k=k)
        sections.append({
            "name": f"筛选6: 死循环剔除 (k={k})",
            "desc": f"连续 ≥ {k} 次相同 action → 剔除该 attempt",
            "metrics": MetricsComputer.compute(f, f"loops_k{k}"),
            "examples": ex,
        })

    # 筛选 8: 步骤数范围
    for lo, hi in [(1, 10), (1, 15), (3, 20)]:
        f, ex = DataFilter.filter_by_step_count(tasks, lo, hi)
        sections.append({
            "name": f"筛选8: 步骤数 ∈ [{lo}, {hi}]",
            "desc": f"仅保留步骤数 ∈ [{lo}, {hi}] 的 attempt",
            "metrics": MetricsComputer.compute(f, f"steps_{lo}_{hi}"),
            "examples": ex,
        })

    # ── 组合筛选 (根据命令行参数) ──
    if _has_cli_filters(args):
        combined = _apply_combined_filters(tasks, args)
        sections.append({
            "name": "CLI 组合筛选",
            "desc": _describe_cli_filters(args),
            "metrics": MetricsComputer.compute(combined, "combined"),
            "examples": [],
        })

    return sections


def _has_cli_filters(args) -> bool:
    return any([
        args.best_trajectory, args.remove_infeasible > 0, args.sr_range,
        args.positive_only, args.success_only, args.remove_loops > 0,
        args.remove_errors, args.step_range,
    ])


def _apply_combined_filters(tasks: Dict, args) -> Dict:
    f = tasks
    if args.remove_errors:
        f, _ = DataFilter.remove_errors(f)
    if args.remove_infeasible > 0:
        f, _ = DataFilter.remove_infeasible(f, args.remove_infeasible)
    if args.sr_range:
        f, _ = DataFilter.filter_by_sr(f, args.sr_range[0], args.sr_range[1])
    if args.remove_loops > 0:
        f, _ = DataFilter.remove_loops(f, args.remove_loops)
    if args.success_only:
        f, _ = DataFilter.success_only(f)
    if args.step_range:
        f, _ = DataFilter.filter_by_step_count(f, args.step_range[0], args.step_range[1])
    if args.best_trajectory:
        f, _ = DataFilter.best_trajectory(f)
    if args.positive_only:
        f, _ = DataFilter.positive_steps_only(f)
    return f


def _describe_cli_filters(args) -> str:
    parts = []
    if args.remove_errors:
        parts.append("删除异常")
    if args.remove_infeasible > 0:
        parts.append(f"infeasible≥{args.remove_infeasible}")
    if args.sr_range:
        parts.append(f"SR∈[{args.sr_range[0]},{args.sr_range[1]}]")
    if args.remove_loops > 0:
        parts.append(f"死循环k≥{args.remove_loops}")
    if args.success_only:
        parts.append("仅成功")
    if args.step_range:
        parts.append(f"步骤∈[{args.step_range[0]},{args.step_range[1]}]")
    if args.best_trajectory:
        parts.append("最优轨迹")
    if args.positive_only:
        parts.append("仅positive")
    return "组合: " + " + ".join(parts)
