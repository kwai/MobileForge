#!/usr/bin/env python3
"""Filter hard tasks from MobileForge results CSV files.

Hard task default: a task has no successful evaluated attempt in an input
results CSV. Multiple input files are combined by union by default.

python scripts/filter_hard_tasks.py results/session-mobileforge-benchmark-v26040501.csv results/session-mobileforge-benchmark-v26040801.csv

"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_REFERENCE_TASK_CSV = (
    "data/rollout/26020401-20apps-400tasks/26020401-20apps-1000tasks.csv"
)
DEFAULT_OUTPUT_DIR = "data/rolling-hard-task"
EVALUATED_FAILURE_VALUES = {"F", "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter hard tasks from one or more MobileForge results CSV files."
    )
    parser.add_argument(
        "results_csv",
        nargs="+",
        help="Input results.csv file(s).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the output task CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--reference-task-csv",
        default=DEFAULT_REFERENCE_TASK_CSV,
        help=(
            "CSV whose header defines the raw task columns to keep. "
            f"Default: {DEFAULT_REFERENCE_TASK_CSV}"
        ),
    )
    parser.add_argument(
        "--combine-mode",
        choices=("intersection", "union"),
        default="union",
        help=(
            "How to combine hard tasks from multiple inputs. "
            "Default: union."
        ),
    )
    parser.add_argument(
        "--strict-failed-only",
        action="store_true",
        help=(
            "Require every evaluation attempt to be exactly F. "
            "By default, a task is hard if it has no S and has at least one F/E."
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def get_reference_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def get_eval_columns(fieldnames: list[str]) -> list[str]:
    columns = [
        col
        for col in fieldnames
        if re.search(r"_attempt_\d+_evaluation$", col)
    ]

    def sort_key(col: str) -> tuple[str, int, str]:
        attempt_match = re.search(r"_attempt_(\d+)_evaluation$", col)
        prefix = col[: attempt_match.start()] if attempt_match else col
        attempt = int(attempt_match.group(1)) if attempt_match else 0
        return prefix, attempt, col

    return sorted(columns, key=sort_key)


def is_hard_task(
    row: dict[str, str],
    eval_columns: list[str],
    strict_failed_only: bool,
) -> bool:
    values = [row.get(col, "").strip() for col in eval_columns]
    if not values:
        return False
    if strict_failed_only:
        return all(value == "F" for value in values)
    return "S" not in values and any(value in EVALUATED_FAILURE_VALUES for value in values)


def extract_identifier(path: Path) -> str:
    normalized_path = path.as_posix()
    version_matches = re.findall(r"v\d+", normalized_path)
    if version_matches:
        return version_matches[-1]
    stem = path.stem
    return re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-") or "results"


def build_output_path(input_paths: list[Path], output_dir: Path) -> Path:
    identifiers = [extract_identifier(path) for path in input_paths]
    return output_dir / f"hard-tasks-{'_'.join(identifiers)}.csv"


def validate_raw_columns(raw_columns: list[str], fieldnames: list[str], path: Path) -> None:
    missing = [col for col in raw_columns if col not in fieldnames]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{path} is missing required raw task columns: {missing_str}")


def main() -> None:
    args = parse_args()

    input_paths = [Path(path) for path in args.results_csv]
    output_dir = Path(args.output_dir)
    reference_path = Path(args.reference_task_csv)

    raw_columns = get_reference_columns(reference_path)
    hard_sets: list[set[str]] = []
    task_sets: list[set[str]] = []
    rows_by_file: list[dict[str, dict[str, str]]] = []
    order_by_file: list[list[str]] = []

    for path in input_paths:
        fieldnames, rows = read_csv_rows(path)
        validate_raw_columns(raw_columns, fieldnames, path)
        eval_columns = get_eval_columns(fieldnames)
        if not eval_columns:
            raise ValueError(f"{path} has no evaluation columns.")

        by_id = {row["task_identifier"]: row for row in rows}
        hard_ids = {
            row["task_identifier"]
            for row in rows
            if is_hard_task(row, eval_columns, args.strict_failed_only)
        }

        rows_by_file.append(by_id)
        order_by_file.append([row["task_identifier"] for row in rows])
        task_sets.append(set(by_id))
        hard_sets.append(hard_ids)

        print(
            f"Input: {path} | rows={len(rows)} | "
            f"evaluation_columns={len(eval_columns)} | hard_tasks={len(hard_ids)}"
        )

    if len(task_sets) > 1:
        task_union = set.union(*task_sets)
        task_intersection = set.intersection(*task_sets)
        print(
            f"Task ID overlap: union={len(task_union)} | "
            f"intersection={len(task_intersection)}"
        )

    if args.combine_mode == "intersection":
        selected_ids = set.intersection(*hard_sets) if hard_sets else set()
    else:
        selected_ids = set.union(*hard_sets) if hard_sets else set()

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for task_id in order_by_file[0]:
        if task_id in selected_ids and task_id not in seen:
            ordered_ids.append(task_id)
            seen.add(task_id)
    if args.combine_mode == "union":
        for order in order_by_file[1:]:
            for task_id in order:
                if task_id in selected_ids and task_id not in seen:
                    ordered_ids.append(task_id)
                    seen.add(task_id)

    merged_rows: dict[str, dict[str, str]] = {}
    for by_id in rows_by_file:
        merged_rows.update(by_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(input_paths, output_dir)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns)
        writer.writeheader()
        for task_id in ordered_ids:
            source_row = merged_rows[task_id]
            writer.writerow({col: source_row.get(col, "") for col in raw_columns})

    print(f"Combine mode: {args.combine_mode}")
    print(f"Selected hard tasks: {len(ordered_ids)}")
    if len(input_paths) > 1 and args.combine_mode == "intersection" and not ordered_ids:
        print(
            "No hard tasks were shared by all inputs. "
            "If the inputs are different task subsets, use --combine-mode union."
        )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
