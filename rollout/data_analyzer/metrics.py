"""
指标计算引擎

对任意子集的任务数据计算全面的统计指标。
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Any


def _att_index(att_id: str) -> int:
    """从 attempt_id (如 'attempt_3') 中提取序号"""
    m = re.search(r"(\d+)", att_id or "")
    return int(m.group(1)) if m else 0


class MetricsComputer:
    """计算一组任务数据的各项指标"""

    @staticmethod
    def compute(tasks: Dict[str, Dict], label: str = "") -> Dict[str, Any]:
        """
        计算完整指标集。

        Args:
            tasks: {task_id: task_data}
            label: 指标集标签

        Returns:
            指标字典
        """
        if not tasks:
            return {"label": label, "empty": True}

        total_tasks = len(tasks)
        total_attempts = success_att = failed_att = error_att = 0
        total_steps = step_ok = step_fail = 0
        impact_c: Counter = Counter()
        reas_c: Counter = Counter()
        infeasible_tasks = feasible_tasks = unk_feasible = 0
        sr_values: List[float] = []
        per_task_impacts: List[Dict] = []
        steps_per_attempt: List[int] = []
        loop_counts: Counter = Counter()
        action_type_c: Counter = Counter()
        # eval hint 统计
        hint_generated = 0   # 生成了 eval_hint 的 attempt 数
        hint_consumed = 0    # 使用了 hints_input 的 attempt 数
        hint_both = 0        # 同时生成且使用的 attempt 数
        # pass@k 统计：记录每个任务的每个 attempt 序号的成功情况
        max_k = 0
        task_attempt_results: List[List[bool]] = []  # 每个任务按序号排列的 success 列表
        

        for tid, task in tasks.items():
            sr_values.append(task.get("avg_sr", 0.0))
            task_imp: Counter = Counter()
            inf_votes = feas_votes = 0

            # 按 attempt 序号排序
            sorted_atts = sorted(task["attempts"].items(),
                                 key=lambda x: _att_index(x[0]))
            att_successes: List[bool] = []

            for att_id, att in sorted_atts:
                total_attempts += 1
                fr = att.get("final_result", -1)
                is_success = (fr == 1)
                att_successes.append(is_success)
                if is_success:
                    success_att += 1
                elif fr == 0:
                    failed_att += 1
                else:
                    error_att += 1

                tf = att.get("task_feasible")
                if tf is True:
                    feas_votes += 1
                elif tf is False:
                    inf_votes += 1

                mc = att.get("max_consecutive_same_actions", 0)
                for threshold in [3, 5, 7, 10]:
                    if mc >= threshold:
                        loop_counts[threshold] += 1

                # hint 统计
                h_gen = att.get("has_eval_hint", False)
                h_inp = att.get("has_hints_input", False)
                if h_gen:
                    hint_generated += 1
                if h_inp:
                    hint_consumed += 1
                if h_gen and h_inp:
                    hint_both += 1

                steps = att.get("steps", [])
                steps_per_attempt.append(len(steps))
                for s in steps:
                    total_steps += 1
                    if s.get("step_success", False):
                        step_ok += 1
                    else:
                        step_fail += 1
                    imp = s.get("impact", "unknown")
                    impact_c[imp] += 1
                    task_imp[imp] += 1
                    reas_c[s.get("reasonableness", "unknown")] += 1
                    action = s.get("action")
                    if isinstance(action, list) and action:
                        action_type_c[action[0]] += 1

            task_attempt_results.append(att_successes)
            if len(att_successes) > max_k:
                max_k = len(att_successes)

            if inf_votes > feas_votes and inf_votes > 0:
                infeasible_tasks += 1
            elif feas_votes > 0:
                feasible_tasks += 1
            else:
                unk_feasible += 1

            per_task_impacts.append({
                "task_id": tid,
                "total": sum(task_imp.values()),
                "positive": task_imp.get("positive", 0),
                "negative": task_imp.get("negative", 0),
                "neutral": task_imp.get("neutral", 0),
                "unknown": task_imp.get("unknown", 0),
            })

        pct = lambda n, t: round(n / t * 100, 2) if t else 0.0
        avg = lambda vals: round(sum(vals) / len(vals), 2) if vals else 0

        # SR 分布 — 按精确 SR 值统计（SR 是离散的：成功数/总数）
        sr_counter: Counter = Counter()
        for v in sr_values:
            label = f"{round(v * 100)}%"
            sr_counter[label] += 1
        # 按数值排序
        sr_dist = dict(sorted(sr_counter.items(), key=lambda x: float(x[0].rstrip('%'))))

        # pass@k 计算
        # pass@k = 在前 k 次 attempt 中至少有一次成功的任务比例
        pass_at_k: Dict[int, float] = {}
        for k in range(1, max_k + 1):
            passed = 0
            for att_results in task_attempt_results:
                # 如果该任务的 attempt 数不足 k，则用已有的全部
                first_k = att_results[:k]
                if any(first_k):
                    passed += 1
            pass_at_k[k] = round(passed / total_tasks * 100, 2) if total_tasks else 0.0

        return {
            "label": label, "empty": False,
            "task_count": total_tasks,
            "attempt_count": total_attempts,
            "step_count": total_steps,
            # 可行性
            "infeasible_tasks": infeasible_tasks,
            "feasible_tasks": feasible_tasks,
            "unk_feasible_tasks": unk_feasible,
            "infeasible_pct": pct(infeasible_tasks, total_tasks),
            # 轨迹
            "success_attempts": success_att,
            "failed_attempts": failed_att,
            "error_attempts": error_att,
            "success_att_pct": pct(success_att, total_attempts),
            "failed_att_pct": pct(failed_att, total_attempts),
            "error_att_pct": pct(error_att, total_attempts),
            # 步骤
            "step_ok": step_ok, "step_fail": step_fail,
            "step_ok_pct": pct(step_ok, total_steps),
            "step_fail_pct": pct(step_fail, total_steps),
            # impact
            "impact": {k: impact_c.get(k, 0) for k in ["positive", "negative", "neutral", "unknown"]},
            "impact_pct": {k: pct(impact_c.get(k, 0), total_steps)
                          for k in ["positive", "negative", "neutral", "unknown"]},
            # reasonableness
            "reas": {k: reas_c.get(k, 0) for k in ["reasonable", "unreasonable", "unknown"]},
            "reas_pct": {k: pct(reas_c.get(k, 0), total_steps)
                        for k in ["reasonable", "unreasonable", "unknown"]},
            # SR
            "avg_sr": avg(sr_values),
            "sr_dist": sr_dist,
            # pass@k
            "pass_at_k": pass_at_k,
            "max_k": max_k,
            # 每任务步骤
            "avg_steps_per_task": avg([p["total"] for p in per_task_impacts]),
            "avg_positive_per_task": avg([p["positive"] for p in per_task_impacts]),
            "avg_negative_per_task": avg([p["negative"] for p in per_task_impacts]),
            "avg_neutral_per_task": avg([p["neutral"] for p in per_task_impacts]),
            "avg_unknown_per_task": avg([p["unknown"] for p in per_task_impacts]),
            # 轨迹步骤数
            "avg_steps_per_att": avg(steps_per_attempt),
            "max_steps_per_att": max(steps_per_attempt) if steps_per_attempt else 0,
            "min_steps_per_att": min(steps_per_attempt) if steps_per_attempt else 0,
            # 死循环
            "loop_counts": dict(loop_counts),
            "loop_pct_k3": pct(loop_counts.get(3, 0), total_attempts),
            # Action 类型
            "action_types": dict(action_type_c.most_common(15)),
            # Eval Hint
            "hint_generated": hint_generated,
            "hint_consumed": hint_consumed,
            "hint_both": hint_both,
            "hint_generated_pct": pct(hint_generated, total_attempts),
            "hint_consumed_pct": pct(hint_consumed, total_attempts),
            "hint_none": total_attempts - hint_generated - hint_consumed + hint_both,
        }

    @staticmethod
    def compute_per_app(tasks: Dict[str, Dict]) -> Dict[str, Dict]:
        """按 App 分组计算指标"""
        app_groups: Dict[str, Dict] = defaultdict(dict)
        for tid, t in tasks.items():
            app_groups[t.get("app", "Unknown")][tid] = t
        return {
            app: MetricsComputer.compute(grp, label=app)
            for app, grp in sorted(app_groups.items())
        }
