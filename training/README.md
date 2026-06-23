# MobileForge Training

This repository contains the lightweight training code used by MobileForge for
hint-contextualized step-level GRPO training of mobile GUI agents. The code is
based on a veRL-style training stack and keeps only the public training path
needed for the MobileForge Qwen3-VL-8B experiment.

## Contents

- `verl/`: training framework.
- `examples/qwen3_vl_8b_mobileforge_grpo.sh`: main MobileForge GRPO training entry.
- `examples/config.yaml`: default training configuration.
- `examples/format_prompt/`: Qwen3-VL GUI-agent prompt templates.
- `examples/reward_function/adaptive_gui.py`: adaptive GUI action reward.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Format

The training script expects MobileForge step-level GRPO JSON data. A training
file should contain conversation-style samples with the target GUI action and
metadata used by the adaptive GUI reward. A directory can also be passed; in
that case, files matching `mobileforge_grpo_*.json` are loaded.

Required paths:

- `--data_path`: MobileForge training JSON file or directory.
- `--val_data_path`: MobileForge validation JSON file or directory.

## Qwen3-VL-8B MobileForge GRPO Training

```bash
bash examples/qwen3_vl_8b_mobileforge_grpo.sh \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path /path/to/mobileforge_train.json \
  --val_data_path /path/to/mobileforge_val.json \
  --experiment_name qwen3_vl_8b_mobileforge_grpo \
  --total_epochs 4 \
  --val_freq 50 \
  --filter_sr_min 0.0 \
  --filter_sr_max 0.9 \
  --filter_best_trajectory true \
  --filter_keep_hard_task_best_path false \
  --remove_evaluation_hints false \
  --reward_action_type_weight 0.2 \
  --reward_action_params_weight 0.8
```

The most important MobileForge options are:

- `--remove_evaluation_hints false`: keep corrective hints in the prompt.
- `--filter_sr_min` and `--filter_sr_max`: select tasks by rollout success rate.
- `--filter_best_trajectory true`: keep the best trajectory for each task.
- `--reward_action_type_weight` and `--reward_action_params_weight`: combine
  action-type and action-parameter rewards.

By default, logs are written locally through the file logger. Configure external
loggers explicitly if needed.

## Auxiliary Data Tools

The original training repository includes two optional preprocessing helpers:

- `tools/create_scaling_splits.py`: creates 200/400/900 task scaling splits for ablation studies.
- `tools/extract_images_to_files.py`: extracts large inline images to files when a JSON dataset is too large to process comfortably in memory.

The released JSON files are available in [🤗 `lgy0404/mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data), so these helpers are not required for reproducing the paper training runs.
