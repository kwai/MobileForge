"""
数据筛选模块

8 种可任意组合的筛选策略，每个方法返回 (筛选后数据, 示例列表)。
"""

import copy
from typing import Dict, List, Tuple


class DataFilter:
    """数据筛选器"""

    @staticmethod
    def _copy(tasks: Dict) -> Dict:
        return copy.deepcopy(tasks)

    @staticmethod
    def _recalc(tasks: Dict):
        """重新计算所有任务的 avg_sr"""
        for t in tasks.values():
            atts = t["attempts"]
            ok = sum(1 for a in atts.values() if a.get("overall_success"))
            t["avg_sr"] = ok / len(atts) if atts else 0.0

    @staticmethod
    def _prune_empty(tasks: Dict) -> Dict:
        return {k: v for k, v in tasks.items() if v["attempts"]}

    # ──── 筛选 1: 最优轨迹选择 ──── #
    @staticmethod
    def best_trajectory(tasks: Dict) -> Tuple[Dict, List]:
        """每个任务仅保留一条最优 attempt (优先成功, 其次 positive 占比最高)"""
        f = DataFilter._copy(tasks)
        examples = []
        for tid, t in f.items():
            atts = t["attempts"]
            if len(atts) <= 1:
                continue
            scored = []
            for aid, a in atts.items():
                steps = a.get("steps", [])
                pos = sum(1 for s in steps if s.get("impact") == "positive")
                ratio = pos / len(steps) if steps else 0
                scored.append((aid, a["overall_success"], ratio))
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best = scored[0]
            removed = [s[0] for s in scored[1:]]
            if removed and len(examples) < 8:
                examples.append({
                    "task_id": tid, "kept": best[0], "removed": removed,
                    "reason": f"success={best[1]}, pos_ratio={best[2]:.2f}",
                })
            t["attempts"] = {best[0]: atts[best[0]]}
        DataFilter._recalc(f)
        return DataFilter._prune_empty(f), examples

    # ──── 筛选 2: Infeasible 任务剔除 ──── #
    @staticmethod
    def remove_infeasible(tasks: Dict, threshold: int = 1) -> Tuple[Dict, List]:
        """同一任务 infeasible 投票 >= threshold → 剔除整个任务"""
        f = DataFilter._copy(tasks)
        examples = []
        to_del = []
        for tid, t in f.items():
            inf_count = sum(
                1 for a in t["attempts"].values() if a.get("task_feasible") is False
            )
            if inf_count >= threshold:
                to_del.append(tid)
                if len(examples) < 8:
                    examples.append({
                        "task_id": tid, "infeasible_votes": inf_count,
                        "total_attempts": len(t["attempts"]),
                        "desc": t.get("task_description", "")[:100],
                    })
        for tid in to_del:
            del f[tid]
        return f, examples

    # ──── 筛选 3: SR 范围筛选 ──── #
    @staticmethod
    def filter_by_sr(tasks: Dict, sr_min: float, sr_max: float) -> Tuple[Dict, List]:
        """保留 avg_sr ∈ [sr_min, sr_max] 的任务"""
        f = DataFilter._copy(tasks)
        removed_low, removed_high = [], []
        to_del = []
        for tid, t in f.items():
            sr = t.get("avg_sr", 0.0)
            if sr < sr_min:
                to_del.append(tid)
                removed_low.append({"task_id": tid, "sr": round(sr, 4)})
            elif sr > sr_max:
                to_del.append(tid)
                removed_high.append({"task_id": tid, "sr": round(sr, 4)})
        for tid in to_del:
            del f[tid]
        examples = [{
            "range": [sr_min, sr_max],
            "removed_below": removed_low[:5],
            "removed_above": removed_high[:5],
            "count_below": len(removed_low),
            "count_above": len(removed_high),
        }]
        return f, examples

    # ──── 筛选 4: 仅保留 positive steps ──── #
    @staticmethod
    def positive_steps_only(tasks: Dict) -> Tuple[Dict, List]:
        """删除所有 impact != positive 的步骤"""
        f = DataFilter._copy(tasks)
        examples = []
        total_removed = total_kept = 0
        for t in f.values():
            for a in t["attempts"].values():
                orig = len(a["steps"])
                a["steps"] = [s for s in a["steps"] if s.get("impact") == "positive"]
                rem = orig - len(a["steps"])
                total_removed += rem
                total_kept += len(a["steps"])
                if rem > 0 and len(examples) < 5:
                    examples.append({
                        "task_id": t["task_id"], "attempt": a["attempt_id"],
                        "before": orig, "after": len(a["steps"]),
                    })
        examples.append({"_summary": True, "removed": total_removed, "kept": total_kept})
        return f, examples

    # ──── 筛选 5: 仅保留成功 attempt ──── #
    @staticmethod
    def success_only(tasks: Dict) -> Tuple[Dict, List]:
        """删除 overall_success=false 的 attempt"""
        f = DataFilter._copy(tasks)
        examples = []
        for t in f.values():
            to_del = [aid for aid, a in t["attempts"].items() if not a.get("overall_success")]
            for aid in to_del:
                if len(examples) < 5:
                    examples.append({
                        "task_id": t["task_id"], "attempt": aid,
                        "final_result": t["attempts"][aid].get("final_result"),
                    })
                del t["attempts"][aid]
        f = DataFilter._prune_empty(f)
        DataFilter._recalc(f)
        return f, examples

    # ──── 筛选 6: 死循环 attempt 剔除 ──── #
    @staticmethod
    def remove_loops(tasks: Dict, k: int = 3) -> Tuple[Dict, List]:
        """连续 >= k 次相同 action → 剔除该 attempt"""
        f = DataFilter._copy(tasks)
        examples = []
        for t in f.values():
            to_del = []
            for aid, a in t["attempts"].items():
                if a.get("max_consecutive_same_actions", 0) >= k:
                    to_del.append(aid)
                    if len(examples) < 5:
                        examples.append({
                            "task_id": t["task_id"], "attempt": aid,
                            "consecutive": a["max_consecutive_same_actions"],
                            "detail": a.get("loop_detail", "")[:120],
                        })
            for aid in to_del:
                del t["attempts"][aid]
        f = DataFilter._prune_empty(f)
        DataFilter._recalc(f)
        return f, examples

    # ──── 筛选 7: 评估异常 attempt 剔除 ──── #
    @staticmethod
    def remove_errors(tasks: Dict) -> Tuple[Dict, List]:
        """删除 final_result ∉ {0,1} 的异常 attempt"""
        f = DataFilter._copy(tasks)
        examples = []
        for t in f.values():
            to_del = []
            for aid, a in t["attempts"].items():
                if a.get("final_result", -1) not in (0, 1):
                    to_del.append(aid)
                    if len(examples) < 5:
                        examples.append({
                            "task_id": t["task_id"], "attempt": aid,
                            "final_result": a.get("final_result"),
                        })
            for aid in to_del:
                del t["attempts"][aid]
        f = DataFilter._prune_empty(f)
        DataFilter._recalc(f)
        return f, examples

    # ──── 筛选 8: 步骤数范围筛选 ──── #
    @staticmethod
    def filter_by_step_count(tasks: Dict, min_s: int, max_s: int) -> Tuple[Dict, List]:
        """仅保留步骤数 ∈ [min_s, max_s] 的 attempt"""
        f = DataFilter._copy(tasks)
        examples = []
        for t in f.values():
            to_del = []
            for aid, a in t["attempts"].items():
                n = len(a.get("steps", []))
                if n < min_s or n > max_s:
                    to_del.append(aid)
                    if len(examples) < 5:
                        examples.append({
                            "task_id": t["task_id"], "attempt": aid, "steps": n,
                            "range": [min_s, max_s],
                        })
            for aid in to_del:
                del t["attempts"][aid]
        f = DataFilter._prune_empty(f)
        DataFilter._recalc(f)
        return f, examples
