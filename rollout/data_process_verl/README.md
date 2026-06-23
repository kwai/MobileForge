# MobileForge Rollout Data Processor

`mobileforge_data_processor.py` converts MobileForge rollout sessions into step-level GRPO training JSON files. It reads rollout logs, MobileGym-Critic summaries, screenshots, and hint artifacts, then produces training samples with task, attempt, step, and feedback metadata.

## Features

- Process one or more rollout directories.
- Convert completed attempts into MobileForge GRPO samples.
- Merge positive and negative step-level samples into one JSON file.
- Preserve task, attempt, step, success, hint, and evaluation metadata.
- Detect and repair placeholder-image logs when raw base64 screenshots are not embedded.
- Generate statistics for task difficulty, trajectory outcomes, step labels, errors, and filtering.
- Support optional parallel processing and resumable processing utilities.

## Layout

```text
data_process_verl/
|-- mobileforge_data_processor.py  # Main CLI entry
|-- fix_grpo_format.py             # Utility for repairing GRPO-format outputs
|-- core/
|   |-- __init__.py
|   |-- processor.py               # Rollout parser and sample builder
|   |-- data_saver.py              # JSON/statistics writer
|   `-- parallel_processor.py      # Parallel processing utilities
|-- docs/
`-- README.md
```

## Basic Usage

```bash
python mobileforge_data_processor.py \
  --rollout_dir /path/to/rollout_session \
  --output_dir processed_data \
  --format grpo
```

Process multiple rollout directories:

```bash
python mobileforge_data_processor.py \
  --rollout_dir /path/to/session1 /path/to/session2 \
  --output_dir processed_data \
  --format grpo
```

Limit the number of tasks for debugging:

```bash
python mobileforge_data_processor.py \
  --rollout_dir /path/to/rollout_session \
  --output_dir processed_data \
  --max_tasks 20
```

## Common Filtering Options

```bash
python mobileforge_data_processor.py \
  --rollout_dir /path/to/rollout_session \
  --output_dir processed_data \
  --format grpo \
  --filter_sr_min 0.0 \
  --filter_sr_max 0.9 \
  --filter_best_trajectory \
  --remove_evaluation_hints false
```

Important options:

| Option | Description |
| --- | --- |
| `--rollout_dir` | One or more rollout-result directories. |
| `--output_dir` | Output directory; defaults to `processed_data`. |
| `--max_tasks` | Maximum number of tasks to process. |
| `--format` | Output format; `grpo` is the main MobileForge format. |
| `--parallel` | Enable parallel processing. |
| `--max_workers` | Number of parallel workers. |
| `--save_interval` | Save interval for resumable processing. |
| `--no_resume` | Disable resume mode. |
| `--test` | Process a small subset for smoke testing. |
| `--filter_sr_min` | Minimum average task success rate to keep. |
| `--filter_sr_max` | Maximum average task success rate to keep. |
| `--filter_best_trajectory` | Keep only the best trajectory per task. |
| `--remove_evaluation_hints` | Remove evaluation hints from prompts before saving. |

## Output Layout

```text
processed_data/
`-- session_YYYYMMDD_HHMMSS/
    |-- mobileforge_grpo_YYYYMMDD_HHMMSS.json
    |-- mobileforge_grpo_stats_YYYYMMDD_HHMMSS.json
    |-- mobileforge_positive_samples_YYYYMMDD_HHMMSS.json
    |-- mobileforge_negative_samples_YYYYMMDD_HHMMSS.json
    |-- mobileforge_processing_stats_YYYYMMDD_HHMMSS.json
    |-- session_summary.json
    `-- README.md
```

The merged `mobileforge_grpo_*.json` file is the main training input used by `training/examples/qwen3_vl_8b_mobileforge_grpo.sh`.

## GRPO Sample Format

Each sample contains:

- `conversations`: prompt and response messages for the local GUI decision.
- `bad_step`: whether the selected step is a negative sample.
- `metadata`: task ID, attempt ID, step number, outcome labels, process feedback, hint information, app name, action metadata, and filtering statistics.

The samples are sorted by:

1. `task_id`
2. `attempt_id`
3. `step_number`

## Metrics

Trajectory-level metrics:

| Metric | Meaning |
| --- | --- |
| `successful_trajectories` | Attempts with valid execution and successful final evaluation. |
| `failed_trajectories` | Attempts with valid execution but failed final evaluation. |
| `error_trajectories` | Attempts with missing, invalid, or failed evaluation artifacts. |

Task-difficulty metrics:

| Metric | Meaning |
| --- | --- |
| `easy_tasks` | All attempts succeed. |
| `pass1` | The first attempt fails but at least one later attempt succeeds. |
| `pass2` | The first two attempts fail but a later attempt succeeds. |
| `pass3` | The first three attempts fail and the fourth succeeds. |
| `hard_tasks` | All attempts fail. |

Step-level labels:

| Field | Values | Meaning |
| --- | --- | --- |
| `impact` | `positive`, `negative`, `neutral`, `unknown` | Step contribution estimated by MobileGym-Critic. |
| `reasonableness` | `reasonable`, `unreasonable`, `unknown` | Local action-quality label. |
| `step_label` | `positive`, `negative`, `unknown` | Training-oriented step label. |
| `step_success` | `true`, `false` | Whether the action executed successfully. |
| `overall_success` | `true`, `false` | Whether the full attempt succeeded. |

## Training Integration

Use the generated data with the MobileForge GRPO training script:

```bash
cd /path/to/MobileForge-Opensource/training

bash examples/qwen3_vl_8b_mobileforge_grpo.sh \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path /path/to/mobileforge_grpo_YYYYMMDD_HHMMSS.json \
  --val_data_path /path/to/validation.json \
  --experiment_name qwen3_vl_8b_mobileforge_grpo \
  --remove_evaluation_hints false \
  --reward_action_type_weight 0.2 \
  --reward_action_params_weight 0.8
```

Released training files are available at [🤗 `lgy0404/mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data).

## Troubleshooting

- Empty output: check that each attempt directory contains `evaluation_summary.json`, `final_decision.json`, `log.json`, and `detailed_model_logs.json`.
- Image mismatch: use the placeholder-image repair path by keeping the corresponding PNG screenshots in the attempt directory.
- Too many trivial tasks: tighten `--filter_sr_min` and `--filter_sr_max`.
- Missing useful hints: make sure `--remove_evaluation_hints false` is used when training hint-contextualized GRPO.
