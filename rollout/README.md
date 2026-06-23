# MobileForge Rollout

This directory contains MobileForge rollout execution, hint-guided multi-attempt data collection, MobileGym-Critic evaluation, and rollout-to-GRPO data processing.

## Install

```bash
conda create -n mobileforge-rollout python=3.10 -y
conda activate mobileforge-rollout
pip install -r requirements.txt
```

Prepare Android SDK, ADB, and either AndroidWorld AVDs or physical Android devices. Then copy `config.yaml.example` to `config.yaml` and fill in local paths, model endpoints, and API keys.

## Run Rollout

```bash
python run.py \
  --agents Qwen3VL \
  --mode full \
  --session_id mobileforge-rollout-demo \
  --max_attempts 4 \
  --reset_app_data
```

When `RUN_MODE=rollout`, MobileForge runs the HiFPO data-collection loop. Attempts for the same task are executed sequentially, so later attempts can use corrective hints generated from earlier failed attempts. Different tasks can be scheduled across multiple devices.

## Key Outputs

```text
results/rollout/<session>/
  results.csv
  <task_identifier>/<agent>/attempt_<k>/
    task_metadata.json
    log.json
    detailed_model_logs.json
    final_decision.json
    evaluation_summary.json
    eval_hint.json
    hints_input.json
```

These files correspond to the paper feedback signals:

- `final_decision.json`: trajectory outcome feedback.
- `evaluation_summary.json`: step-level process feedback.
- `eval_hint.json`: corrective evaluation hint.
- `hints_input.json`: hints reused by later attempts.

## Convert Rollouts to GRPO Data

```bash
python data_process_verl/mobileforge_data_processor.py \
  --rollout_dir results/rollout/mobileforge-rollout-demo \
  --output_dir processed_data \
  --format grpo \
  --filter_sr_min 0.0 \
  --filter_sr_max 0.9 \
  --filter_best_trajectory \
  --remove_evaluation_hints false
```

The released training JSON files are available at [🤗 `lgy0404/mobileforge-training-data`](https://huggingface.co/datasets/lgy0404/mobileforge-training-data) and can be used directly by the scripts in `training/`.
