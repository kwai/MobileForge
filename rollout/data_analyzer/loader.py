"""
数据加载模块

从 rollout 结果目录加载结构化数据，包括：
  - evaluation_summary.json  → 步骤 impact / reasonableness
  - final_decision.json      → 任务可行性 (task_feasible)
  - detailed_model_logs.json → 步骤执行成功状态 (step_success)
  - log.json                 → Action 序列 (死循环检测)
  - task_metadata.json       → 任务级元数据 (app_name, golden_steps 等)
  - results.csv              → App 名称映射 (fallback，当 task_metadata.json 不存在时)
"""

import csv
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

KNOWN_AGENTS = ["UITARS", "UITARS_1_5", "M3A", "M3A_MultiTurn", "Qwen3VL"]


class RolloutDataLoader:
    """从 rollout 结果目录加载结构化数据"""

    def __init__(self, rollout_dirs: List[str], max_tasks: Optional[int] = None):
        self.rollout_dirs = [Path(d) for d in rollout_dirs]
        self.max_tasks = max_tasks
        # fallback：仅在 task_metadata.json 中找不到 app_name 时使用
        self._csv_app_cache: Dict[str, str] = {}
        self._csv_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._csv_loaded = False

    def load_all(self) -> Dict[str, Dict]:
        """
        加载所有任务数据。

        Returns:
            {task_id: task_data} 字典，其中 task_data 包含:
              - task_id, task_description, app, avg_sr
              - task_metadata: 来自 task_metadata.json 或 results.csv 的完整元数据
              - attempts: {attempt_id: attempt_data}
        """
        all_task_dirs: List[Path] = []
        for rollout_dir in self.rollout_dirs:
            for item in sorted(rollout_dir.iterdir()):
                if not item.is_dir() or item.name.startswith("."):
                    continue
                if self._looks_like_task_dir(item):
                    all_task_dirs.append(item)

        if self.max_tasks:
            all_task_dirs = all_task_dirs[: self.max_tasks]

        logger.info(f"发现 {len(all_task_dirs)} 个任务目录")

        tasks: Dict[str, Dict] = {}
        for i, task_dir in enumerate(all_task_dirs, 1):
            task_id = task_dir.name
            if i % 20 == 0 or i == len(all_task_dirs):
                logger.info(f"加载进度: [{i}/{len(all_task_dirs)}]")
            try:
                td = self._load_task(task_dir)
                if td and td["attempts"]:
                    tasks[task_id] = td
            except Exception as e:
                logger.error(f"加载任务 {task_id} 失败: {e}")

        logger.info(f"成功加载 {len(tasks)} 个任务, "
                     f"共 {sum(len(t['attempts']) for t in tasks.values())} 条轨迹")
        return tasks

    # ──────────────── internal ──────────────── #

    def _looks_like_task_dir(self, d: Path) -> bool:
        for agent in KNOWN_AGENTS:
            if (d / agent).is_dir():
                return True
        for sub in d.iterdir():
            if sub.is_dir() and list(sub.glob("attempt_*")):
                return True
        return False

    def _ensure_csv_fallback(self, rollout_dir: Path):
        """懒加载 results.csv 作为 fallback，仅在首次需要时加载"""
        if self._csv_loaded:
            return
        for d in self.rollout_dirs:
            for candidate in [d / "results.csv", d.parent / "results.csv"]:
                if not candidate.exists():
                    continue
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            tid = row.get("task_identifier", "")
                            if not tid:
                                continue
                            app = row.get("app_name", "")
                            if app:
                                self._csv_app_cache[tid] = app
                            # 缓存完整行数据，供 fallback
                            self._csv_metadata_cache[tid] = {
                                "app_name": row.get("app_name", ""),
                                "app_package": row.get("app_package", ""),
                                "golden_steps": _safe_int(row.get("golden_steps", "")),
                                "trajectory_id": row.get("trajectory_id", ""),
                                "original_goal": row.get("original_goal", ""),
                                "task_reasonable": _safe_bool(row.get("task_reasonable", "")),
                                "task_completed": _safe_bool(row.get("task_completed", "")),
                                "task_id": row.get("task_id", ""),
                                "difficulty_level": row.get("difficulty_level", ""),
                                "core_functionality": row.get("core_functionality", ""),
                                "variation_type": row.get("variation_type", ""),
                                "prerequisites": row.get("prerequisites", ""),
                            }
                    logger.info(f"Fallback: 从 results.csv 加载了 {len(self._csv_app_cache)} 条任务元数据")
                except Exception as e:
                    logger.warning(f"读取 {candidate} 失败: {e}")
        self._csv_loaded = True

    def _get_task_metadata(self, task_id: str, first_att_dir: Optional[Path]) -> Dict[str, Any]:
        """
        获取任务级元数据。

        优先级：
          1. attempt 目录下的 task_metadata.json
          2. results.csv (fallback)
          3. 从 task_description 推断 app_name
        """
        # 1) 尝试从 task_metadata.json 读取
        if first_att_dir is not None:
            meta_path = first_att_dir / "task_metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("app_name"):
                        return meta
                except Exception as e:
                    logger.debug(f"读取 task_metadata.json 失败 ({meta_path}): {e}")

        # 2) fallback: results.csv
        if task_id not in self._csv_metadata_cache:
            self._ensure_csv_fallback(Path("."))
        if task_id in self._csv_metadata_cache:
            logger.debug(f"任务 {task_id}: 使用 results.csv fallback 获取元数据")
            return self._csv_metadata_cache[task_id]

        # 3) 空
        return {}

    def _load_task(self, task_dir: Path) -> Optional[Dict]:
        task_id = task_dir.name
        task_data: Dict[str, Any] = {
            "task_id": task_id,
            "task_description": "",
            "app": "",
            "attempts": {},
            "avg_sr": 0.0,
            "task_metadata": {},
        }

        agent_dirs = self._detect_agent_dirs(task_dir)
        if not agent_dirs:
            return None

        # 先扫描所有 attempt 目录列表，后续用于获取 task_metadata
        first_att_dir: Optional[Path] = None
        success_count = total_count = 0
        for agent_name, agent_dir in agent_dirs:
            for att_dir in sorted(agent_dir.glob("attempt_*")):
                if first_att_dir is None:
                    first_att_dir = att_dir
                att = self._load_attempt(att_dir, agent_name)
                if att is None:
                    continue
                task_data["attempts"][att_dir.name] = att
                total_count += 1
                if att["overall_success"]:
                    success_count += 1
                if not task_data["task_description"] and att.get("task_description"):
                    task_data["task_description"] = att["task_description"]

        if not task_data["attempts"]:
            return None

        task_data["avg_sr"] = success_count / total_count if total_count else 0.0

        # 获取任务元数据（优先 task_metadata.json，fallback results.csv）
        meta = self._get_task_metadata(task_id, first_att_dir)
        task_data["task_metadata"] = meta
        task_data["app"] = meta.get("app_name", "")
        if not task_data["app"] and task_data["task_description"]:
            task_data["app"] = extract_app_name(task_data["task_description"])

        return task_data

    def _detect_agent_dirs(self, task_dir: Path) -> List[Tuple[str, Path]]:
        found = []
        for name in KNOWN_AGENTS:
            ad = task_dir / name
            if ad.is_dir():
                found.append((name, ad))
        known = {n for n, _ in found}
        for sub in task_dir.iterdir():
            if sub.is_dir() and sub.name not in known:
                if list(sub.glob("attempt_*")):
                    found.append((sub.name, sub))
        return found

    def _load_attempt(self, att_dir: Path, agent_name: str) -> Optional[Dict]:
        eval_path = att_dir / "evaluation_summary.json"
        if not eval_path.exists():
            return None

        att: Dict[str, Any] = {
            "attempt_id": att_dir.name,
            "agent_name": agent_name,
            "overall_success": False,
            "final_result": -1,
            "task_feasible": None,
            "task_feasible_reason": "",
            "task_barriers": [],
            "task_description": "",
            "steps": [],
            "log_actions": [],
            "max_consecutive_same_actions": 0,
            "loop_detail": "",
            "total_steps": 0,
            # eval hint 相关
            "has_eval_hint": False,       # 该 attempt 是否生成了 eval_hint
            "has_hints_input": False,     # 该 attempt 是否使用了来自前序 attempt 的 hint
        }

        # 1) evaluation_summary.json
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            att["task_description"] = eval_data.get("task_description", "")
            fr = eval_data.get("final_result", -1)
            att["final_result"] = fr
            att["overall_success"] = (fr == 1)

            sr = eval_data.get("step_reasonableness_analysis", {})
            reasonable = set(sr.get("reasonable_steps", []))
            unreasonable = set(sr.get("unreasonable_steps", []))
            step_analysis = sr.get("step_analysis", {})

            for sn_str, analysis in step_analysis.items():
                sn = int(sn_str)
                att["steps"].append({
                    "step_number": sn,
                    "step_success": True,
                    "impact": analysis.get("impact", "unknown"),
                    "reasonableness": (
                        "reasonable" if sn in reasonable
                        else ("unreasonable" if sn in unreasonable else "unknown")
                    ),
                    "explanation": analysis.get("explanation", ""),
                    "action": None,
                })
            att["steps"].sort(key=lambda s: s["step_number"])
        except Exception as e:
            logger.warning(f"evaluation_summary.json 读取出错 ({att_dir}): {e}")
            return None

        # 2) final_decision.json
        fd_path = att_dir / "final_decision.json"
        if fd_path.exists():
            try:
                with open(fd_path, "r", encoding="utf-8") as f:
                    fd = json.load(f)
                att["task_feasible"] = fd.get("task_feasible")
                att["task_feasible_reason"] = fd.get("task_feasible_reason", "")
                att["task_barriers"] = fd.get("task_barriers", [])
            except Exception:
                pass

        # 3) detailed_model_logs.json → step_success
        dm_path = att_dir / "detailed_model_logs.json"
        if dm_path.exists():
            try:
                with open(dm_path, "r", encoding="utf-8") as f:
                    dm = json.load(f)
                ss_map = {}
                for entry in dm:
                    if isinstance(entry, dict) and "step" in entry:
                        ss_map[entry["step"]] = entry.get("success", False)
                for step in att["steps"]:
                    if step["step_number"] in ss_map:
                        step["step_success"] = ss_map[step["step_number"]]
            except Exception:
                pass

        # 4) log.json → action sequence + loop detection
        log_path = att_dir / "log.json"
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                actions = []
                for entry in log_data:
                    if isinstance(entry, dict):
                        actions.append({
                            "step": entry.get("step", 0),
                            "action": entry.get("action"),
                        })
                att["log_actions"] = actions
                action_by_step = {a["step"]: a["action"] for a in actions}
                for step in att["steps"]:
                    if step["step_number"] in action_by_step:
                        step["action"] = action_by_step[step["step_number"]]
                mc, detail = compute_max_consecutive(actions)
                att["max_consecutive_same_actions"] = mc
                att["loop_detail"] = detail
            except Exception:
                pass

        # 5) eval_hint.json / hints_input.json → hint 生成与使用情况
        att["has_eval_hint"] = (att_dir / "eval_hint.json").exists()
        att["has_hints_input"] = (att_dir / "hints_input.json").exists()

        att["total_steps"] = len(att["steps"])
        return att


# ──────────────── 工具函数 ──────────────── #

def _safe_int(val) -> Optional[int]:
    """安全转换为 int，失败返回 None"""
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_bool(val) -> Optional[bool]:
    """安全转换为 bool，失败返回 None"""
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def compute_max_consecutive(actions: List[Dict]) -> Tuple[int, str]:
    """计算最大连续相同 action 数量"""
    if len(actions) < 2:
        return 1 if actions else 0, ""
    strs = [json.dumps(a.get("action"), sort_keys=True) for a in actions]
    max_c = cur_c = 1
    max_action = ""
    max_start = 0
    for i in range(1, len(strs)):
        if strs[i] == strs[i - 1]:
            cur_c += 1
            if cur_c > max_c:
                max_c = cur_c
                max_action = strs[i][:120]
                max_start = actions[i - cur_c + 1]["step"]
        else:
            cur_c = 1
    detail = (f"步骤 {max_start} 起连续 {max_c} 次: {max_action}" if max_c >= 2 else "")
    return max_c, detail


def extract_app_name(desc: str) -> str:
    """从任务描述中提取 app 名称"""
    if not desc:
        return "Unknown"
    patterns = [
        r"^(?:In|Using|Open|Launch)\s+(?:the\s+)?(.+?)(?:\s+app)?\s*,",
        r"^(.+?):\s+",
    ]
    for p in patterns:
        m = re.match(p, desc, re.IGNORECASE)
        if m:
            app = m.group(1).strip()
            if 2 <= len(app) <= 40:
                return app
    return "Unknown"
