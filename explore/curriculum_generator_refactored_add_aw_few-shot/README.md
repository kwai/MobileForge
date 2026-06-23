# MobileGym-Curriculum Generator

This module generates executable mobile GUI tasks from MobileForge exploration traces. It is the curriculum-generation version used in the paper experiments, with AndroidWorld few-shot examples included in the prompt.

## What It Does

The generator reads processed exploration trajectories, evaluates whether the original interaction is useful, analyzes step quality, and asks an MLLM to produce new task instructions grounded in real app behavior.

Main capabilities:

- Generate diverse user-facing tasks from app interaction traces.
- Use trajectory evidence rather than only app names or landing screenshots.
- Estimate task difficulty and expected step count.
- Save CSV, JSON, statistics, and summary reports.
- Keep debug artifacts for prompt inspection and failure analysis.

## Layout

```text
curriculum_generator_refactored_add_aw_few-shot/
|-- main.py                     # CLI entry point
|-- trajectory_parser.py        # Exploration trajectory parser
|-- action_visualizer.py        # Action screenshot visualization
|-- unified_task_processor.py   # Joint trajectory evaluation and task generation
|-- result_saver.py             # CSV/JSON/statistics writer
|-- few_shot_examples.py        # AndroidWorld few-shot examples
`-- README.md
```

## Setup

```bash
pip install pillow opencv-python requests python-dotenv
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL_NAME=your-vision-language-model
```

## Usage

Generate tasks for all apps:

```bash
python main.py \
  --input_dir /path/to/exploration_output_final \
  --output_dir generated_tasks
```

Generate tasks for one app:

```bash
python main.py \
  --input_dir /path/to/exploration_output_final \
  --output_dir generated_tasks \
  --app_package com.android.settings \
  --limit 10
```

Show all options:

```bash
python main.py --help
```

Common arguments:

- `--input_dir`: processed exploration trajectory directory.
- `--app_package`: target app package name, or `all`.
- `--output_dir`: output directory.
- `--limit`: maximum number of trajectories to process.

## Outputs

```text
generated_tasks/
|-- <app_package>/
|   |-- generated_tasks.csv
|   |-- raw_results.json
|   |-- statistics.json
|   `-- summary_report.md
|-- all_generated_tasks.csv
`-- master_summary.md
```

`generated_tasks.csv` contains:

- `trajectory_id`: source trajectory ID.
- `original_goal`: original exploration goal.
- `task_reasonable`: whether the original goal is usable.
- `task_completed`: whether the trajectory completed the original goal.
- `task_id`: generated task ID.
- `instruction`: generated task instruction.
- `estimated_steps`: expected number of steps.
- `difficulty_level`: `easy`, `medium`, or `hard`.
- `core_functionality`: app functionality covered by the task.
- `variation_type`: generation strategy.
- `prerequisites`: setup assumptions if any.

Debug artifacts are written under `debug_output/` when enabled, including visualized screenshots and raw model responses.

## Released Data

The generated tasks used in the MobileForge paper are available at [🤗 `lgy0404/mobileforge-generated-tasks`](https://huggingface.co/datasets/lgy0404/mobileforge-generated-tasks).
