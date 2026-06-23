#!/bin/bash
# 批量评估脚本：使用不同的模型组合评估9个session

# 模型API映射
declare -A MODEL_API
MODEL_API["qwen3vl-8b"]="app-cwtopb-1764155124595787091"
MODEL_API["qwen3vl-235b"]="app-qe58sp-1764230750467949118"
MODEL_API["gemini25pro"]="app-nu0fg7-1754119470355380516"

# 配置文件路径
CONFIG_FILE="config.yaml"
RESULTS_DIR="results/rollout"

# 9个session文件夹（命名格式：step_desc_model+final_decision_model）
SESSIONS=(
    "qwen3vl-8b+qwen3vl-8b"
    "qwen3vl-8b+qwen3vl-235b"
    "qwen3vl-8b+gemini25pro"
    "qwen3vl-235b+qwen3vl-8b"
    "qwen3vl-235b+qwen3vl-235b"
    "qwen3vl-235b+gemini25pro"
    "gemini25pro+qwen3vl-8b"
    "gemini25pro+qwen3vl-235b"
    "gemini25pro+gemini25pro"
)

# 备份原始config
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

for session in "${SESSIONS[@]}"; do
    echo "=============================================="
    echo "Processing session: $session"
    echo "=============================================="
    
    # 解析模型名称
    STEP_MODEL=$(echo "$session" | cut -d'+' -f1)
    FINAL_MODEL=$(echo "$session" | cut -d'+' -f2)
    
    # 获取API
    STEP_API="${MODEL_API[$STEP_MODEL]}"
    FINAL_API="${MODEL_API[$FINAL_MODEL]}"
    
    echo "Step Description Model: $STEP_MODEL -> $STEP_API"
    echo "Final Decision Model: $FINAL_MODEL -> $FINAL_API"
    
    # 修改config.yaml中的模型设置
    sed -i "s|^MOBILEGYM_CRITIC_STEP_DESC_MODEL:.*|MOBILEGYM_CRITIC_STEP_DESC_MODEL: \"$STEP_API\"  # $STEP_MODEL|" "$CONFIG_FILE"
    sed -i "s|^MOBILEGYM_CRITIC_FINAL_DECISION_MODEL:.*|MOBILEGYM_CRITIC_FINAL_DECISION_MODEL: \"$FINAL_API\"  # $FINAL_MODEL|" "$CONFIG_FILE"
    
    # 构建session_id（不带session-前缀）
    SESSION_ID="mobileforge-rollout-v26021201-filter-${session}"
    
    echo "Running evaluation for session: $SESSION_ID"
    
    # 运行评估（使用--overwrite覆盖已有结果）
    python run.py --mode eval --overwrite --session_id "$SESSION_ID"
    
    echo "Completed: $session"
    echo ""
done

# 恢复原始config
mv "${CONFIG_FILE}.backup" "$CONFIG_FILE"

echo "=============================================="
echo "All evaluations completed!"
echo "=============================================="

