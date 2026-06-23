#!/bin/bash
# MobileForge Training Script - Qwen3-VL-8B GRPO
# Uses MobileForge format data produced by data_process_verl/mobileforge_data_processor.py
# Supports multiple composable training data filtering strategies (aligned with data_analyzer)
# Training and validation sets are independent data files using the same MobileForge format.
#
# Usage:
#   # Method 1: Use default filtering config (positive_only + loop>=7 removal)
#   bash examples/qwen3_vl_8b_mobileforge_grpo.sh \
#       --data_path /path/to/train_grpo.json \
#       --val_data_path /path/to/val_grpo.json
#
#   # Method 2: Custom filtering parameters
#   bash examples/qwen3_vl_8b_mobileforge_grpo.sh \
#       --data_path /path/to/train_grpo.json \
#       --val_data_path /path/to/val_grpo.json \
#       --filter_loop_threshold 7 \
#       --filter_best_trajectory true \
#       --filter_infeasible_k 2 \
#       --filter_sr_min 0.0 --filter_sr_max 0.75
#
#   # Method 3: Use directories of JSON files (all mobileforge_grpo_*.json will be loaded)
#   bash examples/qwen3_vl_8b_mobileforge_grpo.sh \
#       --data_path /path/to/train_session/ \
#       --val_data_path /path/to/val_session/
#
# Filtering strategies (aligned with data_analyzer/filters.py):
#   -- Mandatory --
#   positive_only:          Keep only impact=positive steps (default true)
#   filter_loop_threshold:  Remove loop attempts, consecutive same action >= k (default 7)
#
#   -- Optional --
#   filter_best_trajectory: Keep only the best trajectory per task (default false)
#   filter_infeasible_k:    Remove task if infeasible votes >= k (default 0=disabled)
#   filter_sr_min/max:      Keep tasks with avg_sr in [min, max] (default -1=disabled)

set -x

# Default parameters
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
DATA_PATH=""            # Training data path (required)
VAL_DATA_PATH=""        # Validation data path (required)
EXPERIMENT_NAME="qwen3_vl_8b_mobileforge_grpo"
REMOVE_EVALUATION_HINTS=true  # Whether to remove EVALUATION HINTS blocks from user prompts

# -- Training parameters --
N_GPUS_PER_NODE=8
TOTAL_EPOCHS=9
VAL_FREQ=1
VAL_GENERATIONS_TO_LOG=1
TRAIN_GENERATIONS_TO_LOG=0

# -- Reward weights --
REWARD_ACTION_TYPE_WEIGHT=0.5      # Weight for action_type score in overall reward
REWARD_ACTION_PARAMS_WEIGHT=0.5    # Weight for action_params score in overall reward
# -- system_button reward mode --
# "default":     button verified in action_params (current scheme)
# "independent": each system_button treated as independent action, button verified in action_type, params=1.0
REWARD_SYSTEM_BUTTON_MODE="default"

# -- Mandatory filters --
POSITIVE_ONLY=true                  # Keep only positive steps
FILTER_LOOP_THRESHOLD=7             # Loop removal threshold (consecutive same action >= 7)

# -- Optional filters (disabled by default) --
FILTER_BEST_TRAJECTORY=false        # Whether to keep only the best trajectory
FILTER_INFEASIBLE_K=0               # Infeasible removal threshold (0=disabled)
FILTER_SR_MIN=-1.0                  # SR filter lower bound (<0=disabled)
FILTER_SR_MAX=-1.0                  # SR filter upper bound (<0=disabled)
FILTER_KEEP_HARD_TASK_BEST_PATH=false  # For tasks where ALL attempts failed (avg_sr=0),
                                    # extract the longest consecutive success prefix
                                    # (step_success=True from step 1) of the best attempt
                                    # and add those steps to training data.

# -- Zero-variance filter / retry options (ablatable independently) --
# 1. exclude_zv_perfect: drop groups whose mean score >= 0.99 (fully saturated, no GRPO signal)
EXCLUDE_ZV_PERFECT=false
EXCLUDE_ZV_PERFECT_ONCE=false       # once-only: skip detected perfect tasks in all future epochs
# 2. exclude_zv_wrong: drop groups whose mean score < action_type_weight * 0.5 (type totally wrong)
EXCLUDE_ZV_WRONG=false
EXCLUDE_ZV_WRONG_ONCE=false          # once-only: skip detected wrong tasks in all future epochs
# 3. retry_zv_wrong: re-sample zv_wrong prompts instead of dropping them immediately
RETRY_ZV_WRONG=false
RETRY_ZV_WRONG_MAX_ATTEMPTS=3       # total attempts per prompt (includes first); suggest 2~5
# 4. retry_zv_perfect: re-sample zv_perfect prompts instead of dropping them immediately
RETRY_ZV_PERFECT=false
RETRY_ZV_PERFECT_MAX_ATTEMPTS=3     # total attempts per prompt (includes first); suggest 2~5

# -- Adaptive Hint options --
# First rollout without hint; if group is zero-variance with wrong action type, retry with hint
# (Only effective when remove_evaluation_hints=false, i.e., data has hints)
# Note: only retry "wrong" groups (action type totally wrong), NOT "perfect" groups (already correct)
ADAPTIVE_HINT=false
ADAPTIVE_HINT_ONLY_ZV=true          # Only retry zv groups with hints (keep non_zv without hint)
ADAPTIVE_HINT_RETRY_WRONG=true      # Retry zv_wrong groups with hints (strongly recommended)
ADAPTIVE_HINT_RETRY_PERFECT=false   # Do NOT retry zv_perfect groups (already correct, no need)
ADAPTIVE_HINT_MAX_ATTEMPTS=2        # Max hint retry attempts (includes first no-hint rollout)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --val_data_path)
            VAL_DATA_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --positive_only)
            POSITIVE_ONLY="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --remove_evaluation_hints)
            REMOVE_EVALUATION_HINTS="$2"
            shift 2
            ;;
        # -- Filter parameters --
        --filter_loop_threshold)
            FILTER_LOOP_THRESHOLD="$2"
            shift 2
            ;;
        --filter_best_trajectory)
            FILTER_BEST_TRAJECTORY="$2"
            shift 2
            ;;
        --filter_infeasible_k)
            FILTER_INFEASIBLE_K="$2"
            shift 2
            ;;
        --filter_sr_min)
            FILTER_SR_MIN="$2"
            shift 2
            ;;
        --filter_sr_max)
            FILTER_SR_MAX="$2"
            shift 2
            ;;
        --filter_keep_hard_task_best_path)
            FILTER_KEEP_HARD_TASK_BEST_PATH="$2"
            shift 2
            ;;
        # -- Zero-variance filter / retry options --
        --exclude_zv_perfect)
            EXCLUDE_ZV_PERFECT="$2"
            shift 2
            ;;
        --exclude_zv_wrong)
            EXCLUDE_ZV_WRONG="$2"
            shift 2
            ;;
        --exclude_zv_perfect_once)
            EXCLUDE_ZV_PERFECT_ONCE="$2"
            shift 2
            ;;
        --exclude_zv_wrong_once)
            EXCLUDE_ZV_WRONG_ONCE="$2"
            shift 2
            ;;
        --retry_zv_wrong)
            RETRY_ZV_WRONG="$2"
            shift 2
            ;;
        --retry_zv_wrong_max_attempts)
            RETRY_ZV_WRONG_MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --retry_zv_perfect)
            RETRY_ZV_PERFECT="$2"
            shift 2
            ;;
        --retry_zv_perfect_max_attempts)
            RETRY_ZV_PERFECT_MAX_ATTEMPTS="$2"
            shift 2
            ;;
        # -- Adaptive Hint options --
        --adaptive_hint)
            ADAPTIVE_HINT="$2"
            shift 2
            ;;
        --adaptive_hint_only_zv)
            ADAPTIVE_HINT_ONLY_ZV="$2"
            shift 2
            ;;
        --adaptive_hint_retry_wrong)
            ADAPTIVE_HINT_RETRY_WRONG="$2"
            shift 2
            ;;
        --adaptive_hint_retry_perfect)
            ADAPTIVE_HINT_RETRY_PERFECT="$2"
            shift 2
            ;;
        --adaptive_hint_max_attempts)
            ADAPTIVE_HINT_MAX_ATTEMPTS="$2"
            shift 2
            ;;
        # -- Training parameters --
        --n_gpus_per_node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --total_epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --val_freq)
            VAL_FREQ="$2"
            shift 2
            ;;
        --val_generations_to_log)
            VAL_GENERATIONS_TO_LOG="$2"
            shift 2
            ;;
        --train_generations_to_log)
            TRAIN_GENERATIONS_TO_LOG="$2"
            shift 2
            ;;
        # -- Reward weight parameters --
        --reward_action_type_weight)
            REWARD_ACTION_TYPE_WEIGHT="$2"
            shift 2
            ;;
        --reward_action_params_weight)
            REWARD_ACTION_PARAMS_WEIGHT="$2"
            shift 2
            ;;
        --reward_system_button_mode)
            REWARD_SYSTEM_BUTTON_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    echo "Usage: bash $0 --data_path /path/to/train_grpo.json --val_data_path /path/to/val_grpo.json"
    exit 1
fi

if [ -z "$VAL_DATA_PATH" ]; then
    echo "Error: --val_data_path is required"
    echo "Usage: bash $0 --data_path /path/to/train_grpo.json --val_data_path /path/to/val_grpo.json"
    exit 1
fi

# Check data paths exist (file or directory)
if [ ! -f "$DATA_PATH" ] && [ ! -d "$DATA_PATH" ]; then
    echo "Error: Training data path not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$VAL_DATA_PATH" ] && [ ! -d "$VAL_DATA_PATH" ]; then
    echo "Error: Validation data path not found: $VAL_DATA_PATH"
    exit 1
fi

echo "============================================"
echo "MobileForge Training - Qwen3-VL-8B GRPO"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Train Data: $DATA_PATH"
echo "Val Data:   $VAL_DATA_PATH"
echo "Positive Only: $POSITIVE_ONLY"
echo "Remove Evaluation Hints: $REMOVE_EVALUATION_HINTS"
echo "-- Filter Config --"
echo "  Loop Threshold: $FILTER_LOOP_THRESHOLD (remove attempts with >= this many consecutive same actions)"
echo "  Best Trajectory: $FILTER_BEST_TRAJECTORY"
echo "  Infeasible K: $FILTER_INFEASIBLE_K (0=disabled)"
echo "  SR Range: [$FILTER_SR_MIN, $FILTER_SR_MAX] (<0=disabled)"
echo "  Hard Task Best Path: $FILTER_KEEP_HARD_TASK_BEST_PATH"
echo "-- Zero-Variance Filter Config --"
echo "  exclude_zv_perfect: $EXCLUDE_ZV_PERFECT  (once=$EXCLUDE_ZV_PERFECT_ONCE)"
echo "  exclude_zv_wrong:   $EXCLUDE_ZV_WRONG  (once=$EXCLUDE_ZV_WRONG_ONCE)"
echo "  retry_zv_wrong:     $RETRY_ZV_WRONG  (max_attempts=$RETRY_ZV_WRONG_MAX_ATTEMPTS)"
echo "  retry_zv_perfect:   $RETRY_ZV_PERFECT  (max_attempts=$RETRY_ZV_PERFECT_MAX_ATTEMPTS)"
echo "-- Adaptive Hint Config --"
echo "  adaptive_hint:       $ADAPTIVE_HINT"
echo "  adaptive_hint_only_zv: $ADAPTIVE_HINT_ONLY_ZV"
echo "  adaptive_hint_retry_wrong: $ADAPTIVE_HINT_RETRY_WRONG"
echo "  adaptive_hint_retry_perfect: $ADAPTIVE_HINT_RETRY_PERFECT"
echo "  adaptive_hint_max_attempts: $ADAPTIVE_HINT_MAX_ATTEMPTS"
echo "-- Training Config --"
echo "  GPUs per node: $N_GPUS_PER_NODE"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Val freq: $VAL_FREQ"
echo "  Val generations to log: $VAL_GENERATIONS_TO_LOG"
echo "  Train generations to log: $TRAIN_GENERATIONS_TO_LOG"
echo "-- Reward Config --"
echo "  Action type weight: $REWARD_ACTION_TYPE_WEIGHT"
echo "  Action params weight: $REWARD_ACTION_PARAMS_WEIGHT"
echo "  System button mode: $REWARD_SYSTEM_BUTTON_MODE"
echo "Experiment: $EXPERIMENT_NAME"
echo "============================================"

# Adaptive reward function - supports both GUIOwl and Qwen3VL formats
# Automatically detects model type from system prompt in training data
REWARD_FUNCTION="examples/reward_function/adaptive_gui.py:compute_score"
SYSTEM_PROMPT="examples/format_prompt/r1gui_qwen3vl_system.jinja"
FORMAT_PROMPT="examples/format_prompt/r1gui_qwen3vl_user.jinja"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_PATH} \
    data.val_files=${VAL_DATA_PATH} \
    data.use_mobileforge_format=true \
    data.positive_only=${POSITIVE_ONLY} \
    data.remove_evaluation_hints=${REMOVE_EVALUATION_HINTS} \
    data.filter_loop_threshold=${FILTER_LOOP_THRESHOLD} \
    data.filter_best_trajectory=${FILTER_BEST_TRAJECTORY} \
    data.filter_infeasible_k=${FILTER_INFEASIBLE_K} \
    data.filter_sr_min=${FILTER_SR_MIN} \
    data.filter_sr_max=${FILTER_SR_MAX} \
    data.filter_keep_hard_task_best_path=${FILTER_KEEP_HARD_TASK_BEST_PATH} \
    data.exclude_zv_perfect=${EXCLUDE_ZV_PERFECT} \
    data.exclude_zv_wrong=${EXCLUDE_ZV_WRONG} \
    data.exclude_zv_perfect_once=${EXCLUDE_ZV_PERFECT_ONCE} \
    data.exclude_zv_wrong_once=${EXCLUDE_ZV_WRONG_ONCE} \
    data.retry_zv_wrong=${RETRY_ZV_WRONG} \
    data.retry_zv_wrong_max_attempts=${RETRY_ZV_WRONG_MAX_ATTEMPTS} \
    data.retry_zv_perfect=${RETRY_ZV_PERFECT} \
    data.retry_zv_perfect_max_attempts=${RETRY_ZV_PERFECT_MAX_ATTEMPTS} \
    data.adaptive_hint=${ADAPTIVE_HINT} \
    data.adaptive_hint_only_zv=${ADAPTIVE_HINT_ONLY_ZV} \
    data.adaptive_hint_retry_wrong=${ADAPTIVE_HINT_RETRY_WRONG} \
    data.adaptive_hint_retry_perfect=${ADAPTIVE_HINT_RETRY_PERFECT} \
    data.adaptive_hint_max_attempts=${ADAPTIVE_HINT_MAX_ATTEMPTS} \
    data.system_prompt=${SYSTEM_PROMPT} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.reward.reward_function_kwargs.action_type_weight=${REWARD_ACTION_TYPE_WEIGHT} \
    worker.reward.reward_function_kwargs.action_params_weight=${REWARD_ACTION_PARAMS_WEIGHT} \
    worker.reward.reward_function_kwargs.system_button_mode=${REWARD_SYSTEM_BUTTON_MODE} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.val_generations_to_log=${VAL_GENERATIONS_TO_LOG} \
    trainer.train_generations_to_log=${TRAIN_GENERATIONS_TO_LOG} \
    data.max_pixels=1258291 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=false \
    data.rollout_batch_size=128 \
    data.val_batch_size=13
