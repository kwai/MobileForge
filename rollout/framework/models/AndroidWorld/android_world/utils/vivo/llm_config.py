# encoding: utf-8
"""
LLM API调用的配置文件
"""

import os
# 默认模型参数
DEFAULT_MODEL = "gemini-2.5-pro" #   qwen-vl-max-2025-01-25    qwen-vl-max-2024-12-30    gemini-1.5-pro-001
DEFAULT_PROVIDER = "google" # aliyun
DEFAULT_MAX_TOKENS = 6500
DEFAULT_TEMPERATURE = 0.01

# 默认API参数
# 改为从环境变量中获取DEFAULT_APP_ID和DEFAULT_APP_KEY
DEFAULT_APP_ID = os.getenv("DEFAULT_APP_ID")
DEFAULT_APP_KEY = os.getenv("DEFAULT_APP_KEY")
DEFAULT_API_URL = "http://chatgpt-api-pre.vmic.xyz/chatgpt/completions"

# 默认重试参数
DEFAULT_MAX_RETRIES = 20
DEFAULT_RETRY_DELAY = 5

# 模型配置字典，可以根据不同任务设置不同的参数
MODEL_CONFIGS = {
    "default": {
        "model": DEFAULT_MODEL,
        "provider": DEFAULT_PROVIDER,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "app_id": DEFAULT_APP_ID,
        "app_key": DEFAULT_APP_KEY,
        "api_url": DEFAULT_API_URL,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_delay": DEFAULT_RETRY_DELAY,
    },
    "high_precision": {
        "model": DEFAULT_MODEL,
        "provider": DEFAULT_PROVIDER,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": 0.01,  # 低温度，更确定性的输出
        "app_id": DEFAULT_APP_ID,
        "app_key": DEFAULT_APP_KEY,
        "api_url": DEFAULT_API_URL,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_delay": DEFAULT_RETRY_DELAY,
    },
    "high_creativity": {
        "model": DEFAULT_MODEL,
        "provider": DEFAULT_PROVIDER,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": 1.0,  # 高温度，更有创造性的输出
        "app_id": DEFAULT_APP_ID,
        "app_key": DEFAULT_APP_KEY,
        "api_url": DEFAULT_API_URL,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retry_delay": DEFAULT_RETRY_DELAY,
    },
}


def get_model_config(config_name="default"):
    """
    获取指定名称的模型配置

    Args:
        config_name: 配置名称，默认为"default"

    Returns:
        配置字典
    """
    return MODEL_CONFIGS.get(config_name, MODEL_CONFIGS["default"])
