#!/usr/bin/env python3
"""
create_scaling_splits.py
========================
为 Scaling Law ablation 实验生成**严格层级包含**的数据集切分。

核心保证：
  N=100 ⊂ N=200 ⊂ N=500 ⊂ N=1000 ⊂ ...
  即小规模 split 的任务集合是大规模 split 的严格子集。

切分逻辑（默认 pre_filter 模式）：
  1. 加载原始数据，获取全部原始 task_id
  2. 用固定 seed 洗牌，建立全局任务顺序
  3. 对每个 N，取前 N 个原始任务的所有 steps（含负样本，**不做任何过滤**）
  4. 保存为独立 JSON

  → 过滤逻辑（SR / infeasible / positive_only 等）在**训练时**由脚本参数控制
  → 每个规模的过滤参数相同，scaling ablation 只控制任务数量这一个变量

使用方法：
  python tools/create_scaling_splits.py \\
      --data_path data/mobileforge_grpo_20260307_081156.json \\
      --output_dir data/scaling_splits \\
      --task_counts 100 200 500 1000 \\
      --seed 42

  # 如需在切分前做粗粒度清洗（死循环剔除等），可选加：
  python tools/create_scaling_splits.py \\
      --data_path data/mobileforge_grpo_20260307_081156.json \\
      --output_dir data/scaling_splits_cleaned \\
      --task_counts 100 200 500 1000 \\
      --seed 42 \\
      --pre_clean_loop      # 仅剔除明显异常的死循环 attempt

生成结果（output_dir/）：
  tasks_100.json       ← 100 个原始任务的所有 steps（含负样本，不含过滤）
  tasks_200.json       ← 200 个原始任务（100 ⊂ 200，严格包含）
  tasks_500.json       ← 500 个原始任务
  tasks_1000.json      ← 1000 个原始任务
  manifest.json        ← 每个 split 的统计信息（原始任务数/样本数/action分布等）
  task_order.txt       ← 所有原始任务 ID 的排列顺序（seed 固定后可完整复现）

训练命令示例：
  for N in 100 200 500 1000; do
    bash examples/qwen3_vl_8b_mobileforge_grpo.sh \\
        --data_path data/scaling_splits/tasks_${N}.json \\
        --val_data_path data/val.json \\
        --experiment_name scaling_law_${N}tasks \\
        --filter_sr_min 0.1 \\
        --filter_sr_max 1.0 \\
        --filter_infeasible_k 1 \\
        --positive_only true
  done
"""

import argparse
import glob as _glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────
# 内联必要的数据过滤函数（逻辑与 verl/utils/dataset.py 完全一致）
# 避免 import torch 等重量级依赖
# ──────────────────────────────────────────────────────────────

def _load_json_files(data_path: str) -> List[dict]:
    all_samples: List[dict] = []
    if "," in data_path and not os.path.exists(data_path):
        paths = [p.strip() for p in data_path.split(",") if p.strip()]
    else:
        paths = [data_path]
    for path in paths:
        if os.path.isfile(path):
            print(f"[MobileForge] Loading file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_samples.extend(data)
        elif os.path.isdir(path):
            grpo_files = sorted(_glob.glob(os.path.join(path, "mobileforge_grpo_*.json")))
            if not grpo_files:
                exclude = {"session_summary.json"}
                all_json = sorted(_glob.glob(os.path.join(path, "*.json")))
                grpo_files = [
                    f for f in all_json
                    if os.path.basename(f) not in exclude
                    and "stats" not in os.path.basename(f)
                ]
            for fp in grpo_files:
                print(f"[MobileForge] Loading file: {fp}")
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
        else:
            raise FileNotFoundError(f"[MobileForge] Data path not found: {path}")
    return all_samples


def _build_task_index(all_samples: List[dict]) -> dict:
    tasks: dict = {}
    for s in all_samples:
        meta = s.get("metadata", {})
        tid = s.get("task_id", meta.get("task_id", ""))
        aid = meta.get("attempt_id", s.get("attempt_id", ""))
        if tid not in tasks:
            tasks[tid] = {"attempts": {}, "attempt_stats": {}}
        if aid not in tasks[tid]["attempts"]:
            final_result = meta.get("final_result", 0)
            attempt_success = int(final_result or 0) == 1
            tasks[tid]["attempts"][aid] = {
                "steps": [],
                "overall_success": attempt_success,
                "final_result": final_result,
                "task_feasible": meta.get("task_feasible", None),
                "max_consecutive_same_actions": meta.get("max_consecutive_same_actions", 0),
                "avg_sr": meta.get("avg_sr", 0.0),
            }
        tasks[tid]["attempts"][aid]["steps"].append(s)
    # 计算任务级 avg_sr
    for tid, tdata in tasks.items():
        atts = tdata["attempts"]
        total = len(atts)
        ok = sum(1 for a in atts.values() if a.get("overall_success"))
        tdata["avg_sr"] = ok / total if total > 0 else 0.0
    return tasks


def _flatten_tasks(tasks: dict) -> List[dict]:
    result = []
    for tdata in tasks.values():
        for adata in tdata["attempts"].values():
            result.extend(adata["steps"])
    return result


def _filter_loop_attempts(tasks: dict, k: int) -> Tuple[dict, int]:
    removed = 0
    for tid in list(tasks.keys()):
        tdata = tasks[tid]
        to_del = []
        for aid, adata in tdata["attempts"].items():
            if adata.get("max_consecutive_same_actions", 0) >= k:
                to_del.append(aid)
                removed += 1
        for aid in to_del:
            del tdata["attempts"][aid]
        if not tdata["attempts"]:
            del tasks[tid]
    return tasks, removed


def _filter_best_trajectory(tasks: dict) -> Tuple[dict, int]:
    removed = 0
    for tid in list(tasks.keys()):
        tdata = tasks[tid]
        atts = tdata["attempts"]
        if len(atts) <= 1:
            continue
        scored = []
        for aid, adata in atts.items():
            steps = adata["steps"]
            pos = sum(
                1 for s in steps
                if s.get("metadata", {}).get("impact") == "positive" or s.get("is_positive", False)
            )
            ratio = pos / len(steps) if steps else 0
            scored.append((aid, adata["overall_success"], ratio))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_aid = scored[0][0]
        for aid in [s[0] for s in scored[1:]]:
            del atts[aid]
            removed += 1
    tasks = {k: v for k, v in tasks.items() if v["attempts"]}
    return tasks, removed


def _filter_infeasible(tasks: dict, k: int) -> Tuple[dict, int]:
    removed = 0
    to_del = []
    for tid, tdata in tasks.items():
        infeasible_votes = sum(
            1 for a in tdata["attempts"].values()
            if a.get("task_feasible") is False
        )
        if infeasible_votes >= k:
            to_del.append(tid)
    for tid in to_del:
        removed += len(tasks[tid]["attempts"])
        del tasks[tid]
    return tasks, removed


def _filter_by_sr(tasks: dict, sr_min: float, sr_max: float) -> Tuple[dict, int]:
    removed = 0
    to_del = []
    for tid, tdata in tasks.items():
        sr = tdata.get("avg_sr", 0.0)
        if sr < sr_min or sr > sr_max:
            to_del.append(tid)
    for tid in to_del:
        removed += len(tasks[tid]["attempts"])
        del tasks[tid]
    return tasks, removed


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def _compute_action_distribution(samples: List[dict]) -> Dict[str, int]:
    """统计 action 类型分布（基于 ground_truth.action）"""
    dist: Dict[str, int] = defaultdict(int)
    for s in samples:
        gt = s.get("ground_truth", {})
        if isinstance(gt, dict):
            action = gt.get("action", "unknown")
        else:
            action = "unknown"
        dist[action] += 1
    return dict(dist)


def _compute_split_stats(task_ids: List[str], tasks: dict) -> dict:
    """计算某个 split（任务子集）的统计信息"""
    # 收集对应的 steps
    all_steps = []
    for tid in task_ids:
        if tid in tasks:
            for adata in tasks[tid]["attempts"].values():
                all_steps.extend(adata["steps"])

    positive_steps = [s for s in all_steps if s.get("is_positive", False)]
    action_dist = _compute_action_distribution(all_steps)
    action_dist_positive = _compute_action_distribution(positive_steps)

    # 统计成功任务数
    success_tasks = sum(
        1 for tid in task_ids
        if tid in tasks and any(
            a.get("overall_success", False)
            for a in tasks[tid]["attempts"].values()
        )
    )

    return {
        "task_count": len(task_ids),
        "success_task_count": success_tasks,
        "success_task_ratio": success_tasks / len(task_ids) if task_ids else 0.0,
        "total_step_count": len(all_steps),
        "positive_step_count": len(positive_steps),
        "positive_step_ratio": len(positive_steps) / len(all_steps) if all_steps else 0.0,
        "action_distribution_all": action_dist,
        "action_distribution_positive": action_dist_positive,
    }


def _tasks_to_samples(task_ids: List[str], tasks: dict) -> List[dict]:
    """从任务字典中抽出指定任务 ID 的所有 step 样本"""
    result = []
    for tid in task_ids:
        if tid not in tasks:
            continue
        for adata in tasks[tid]["attempts"].values():
            result.extend(adata["steps"])
    return result


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def create_scaling_splits(
    data_path: str,
    output_dir: str,
    task_counts: List[int],
    step: int | None = None,
    seed: int = 42,
    # 可选：切分前做轻量清洗（仅剔除明显异常，不做 SR/infeasible 等业务过滤）
    pre_clean_loop: bool = False,
    pre_clean_loop_threshold: int = 7,
) -> None:
    """
    在所有业务过滤（SR / infeasible / positive_only）之前，
    按原始任务数切分数据集，生成严格层级包含的子集。

    各 split 保存的是原始步骤（含负样本），
    SR / infeasible / positive_only 等过滤由训练脚本在加载时完成。
    这样不同规模的实验唯一的变量就是"原始任务数量"。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: 加载原始数据 ──
    print(f"[ScalingSplit] 加载数据: {data_path}")
    all_samples = _load_json_files(data_path)
    print(f"[ScalingSplit] 原始样本数: {len(all_samples)}")

    # ── Step 2: 补全必要字段 ──
    for sample in all_samples:
        metadata = sample.get("metadata", {})
        if "task_id" not in sample:
            sample["task_id"] = metadata.get("task_id", f"unknown_{id(sample)}")
        if "is_positive" not in sample:
            impact = metadata.get("impact", "")
            sample["is_positive"] = impact == "positive"

    # ── Step 2.5: 从输入路径推导文件名前缀 ──
    # 如果 data_path 是目录，则用目录名；如果是文件，则用去掉扩展名的文件名
    # 例如：mobileforge_grpo_20260307_081156.json → 前缀 mobileforge_grpo_20260307_081156
    raw_name = os.path.basename(data_path.rstrip("/"))
    file_prefix = os.path.splitext(raw_name)[0]  # 去掉 .json
    print(f"[ScalingSplit] 输出文件前缀: {file_prefix!r}")

    # ── Step 3: 建立原始任务索引（不做任何业务过滤）──
    tasks = _build_task_index(all_samples)
    raw_task_count = len(tasks)
    print(f"[ScalingSplit] 原始任务数（未过滤）: {raw_task_count}")

    # ── Step 4（可选）: 仅做轻量清洗——死循环剔除 ──
    # 这类 attempt 通常是数据采集 bug，不代表真实任务，可以在切分前清掉
    pre_clean_note = "未做任何预清洗"
    if pre_clean_loop:
        tasks, n = _filter_loop_attempts(tasks, pre_clean_loop_threshold)
        pre_clean_note = f"死循环剔除 (k>={pre_clean_loop_threshold}): 移除 {n} 个 attempt"
        print(f"[ScalingSplit] 预清洗 - {pre_clean_note}，剩余 {len(tasks)} 任务")

    # ── Step 5: 准备全量任务列表 ──
    # sorted() 先保证在不同机器/Python版本上输入顺序一致，作为采样的 pool
    all_task_ids = sorted(tasks.keys())
    available_count = len(all_task_ids)
    print(f"[ScalingSplit] 可用原始任务数: {available_count}，base seed={seed}")

    # ── Step 6: 自动推算切分点 ──
    if step is not None:
        # 按 step 自动生成：step, 2*step, 3*step, ... 不超过 available_count
        task_counts_sorted = list(range(step, available_count + 1, step))
        if not task_counts_sorted:
            raise ValueError(
                f"step={step} 大于可用任务数 {available_count}，无法生成任何 split。"
            )
        print(f"[ScalingSplit] 自动生成切分点（step={step}）: {task_counts_sorted}")
    else:
        task_counts_sorted = sorted(task_counts)
        overflow = [c for c in task_counts_sorted if c > available_count]
        if overflow:
            print(
                f"[ScalingSplit] ⚠️  警告: {overflow} 超过可用任务数 {available_count}，已自动跳过"
            )
            task_counts_sorted = [c for c in task_counts_sorted if c <= available_count]
        if not task_counts_sorted:
            raise ValueError("所有指定的 task_counts 都超过了可用任务数，无法生成任何 split。")

    # ── Step 7: 各规模独立随机采样 ──
    # 每个规模使用 seed ^ count 作为独立种子，保证：
    #   - 同一 (seed, count) 永远得到相同的 N 个任务（可复现）
    #   - 不同 count 之间相互独立（不要求包含关系）
    manifest = {
        "source_data": os.path.abspath(data_path),
        "seed": seed,
        "split_mode": "pre_filter_independent",
        "note": (
            "各规模独立随机采样（seed ^ count），互不包含。"
            "各 split 保存原始步骤（含负样本），"
            "SR/infeasible/positive_only 等过滤在训练时由脚本参数控制。"
        ),
        "pre_clean": {
            "enabled": pre_clean_loop,
            "detail": pre_clean_note,
        },
        "raw_task_count": raw_task_count,
        "available_task_count": available_count,
        "splits": {},
    }

    print(f"\n[ScalingSplit] 开始生成 {len(task_counts_sorted)} 个 split（各规模独立随机采样）...")
    for count in task_counts_sorted:
        # 每个规模独立的随机种子，异或保证不同 count 得到不同结果
        split_seed = seed ^ count
        rng = random.Random(split_seed)
        task_subset = rng.sample(all_task_ids, count)  # 独立随机采样，无包含关系
        samples = _tasks_to_samples(task_subset, tasks)
        stats = _compute_split_stats(task_subset, tasks)

        fname = f"{file_prefix}_tasks_{count}.json"
        out_path = os.path.join(output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=None)

        manifest["splits"][str(count)] = {
            "file": fname,
            "abs_path": os.path.abspath(out_path),
            "split_seed": split_seed,
            "task_ids": sorted(task_subset),  # 记录实际采样的任务 ID，方便核查
            **stats,
        }

        print(
            f"  {fname}  "
            f"原始任务={count:5d}  "
            f"总步骤={stats['total_step_count']:6d}  "
            f"正样本={stats['positive_step_count']:6d}({stats['positive_step_ratio']:.1%})  "
            f"成功任务={stats['success_task_count']:5d}({stats['success_task_ratio']:.1%})  "
            f"split_seed={split_seed}"
        )

    # ── Step 8: 保存 manifest ──
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[ScalingSplit] ✅ 完成！")
    print(f"  输出目录  : {os.path.abspath(output_dir)}")
    print(f"  manifest  : {manifest_path}")
    print(f"  （各规模的任务 ID 列表已记录在 manifest.json 的 splits[N].task_ids 字段）")
    print()
    print("─" * 70)
    print("各 split 保存的是原始数据（不含过滤），训练时请保持相同的过滤参数，例如：")
    print()
    for count in task_counts_sorted:
        fname = f"{file_prefix}_tasks_{count}.json"
        print(f"  bash examples/qwen3_vl_8b_mobileforge_grpo.sh \\")
        print(f"      --data_path {os.path.join(os.path.abspath(output_dir), fname)} \\")
        print(f"      --val_data_path <your_val_path> \\")
        print(f"      --experiment_name scaling_law_{count}tasks_seed{seed} \\")
        print(f"      --filter_sr_min 0.1 --filter_sr_max 1.0 --filter_infeasible_k 1")
        print()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "为 Scaling Law ablation 生成层级包含的数据集切分。\n"
            "切分在所有业务过滤（SR / infeasible / positive_only）之前进行，\n"
            "各 split 的过滤参数在训练时由训练脚本统一控制。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_path", required=True,
        help="原始数据文件路径（JSON 文件或目录）"
    )
    parser.add_argument(
        "--output_dir", default="data/scaling_splits",
        help="输出目录（默认: data/scaling_splits）"
    )
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--step", type=int, default=None,
        help=(
            "自动按等步长生成切分点。"
            "例如共 810 个任务，--step 200 → 生成 200 400 600 800；"
            "--step 400 → 生成 400 800"
        ),
    )
    size_group.add_argument(
        "--task_counts", nargs="+", type=int, default=None,
        help="手动指定切分点，例如 --task_counts 100 300 700"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )

    pre_clean_group = parser.add_argument_group(
        "可选预清洗（切分前的轻量清洗，默认不启用）"
    )
    pre_clean_group.add_argument(
        "--pre_clean_loop", action="store_true", default=False,
        help="切分前剔除死循环 attempt（数据采集 bug，不影响任务粒度的公平性）"
    )
    pre_clean_group.add_argument(
        "--pre_clean_loop_threshold", type=int, default=7,
        help="死循环剔除阈值（默认 7）"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_scaling_splits(
        data_path=args.data_path,
        output_dir=args.output_dir,
        task_counts=args.task_counts or [],
        step=args.step,
        seed=args.seed,
        pre_clean_loop=args.pre_clean_loop,
        pre_clean_loop_threshold=args.pre_clean_loop_threshold,
    )
