# encoding: utf-8
"""
LLM API调用的配置文件
从config.yaml中读取模型配置，支持模式预设系统
"""

import os
import sys

# Add project root to path for config_loader import
_project_root = os.path.join(os.path.dirname(__file__), "../../..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config_loader import get_config


try:
    # Use get_config() to get cached config with mode presets applied
    _config = get_config(verbose=False)
except FileNotFoundError:
    raise FileNotFoundError(
        "config.yaml file not found. Please ensure config.yaml exists in the project root."
    )
except Exception as e:
    raise Exception(f"Error loading config.yaml: {e}")

# 验证必需的配置项
required_keys = [
    "MOBILEGYM_CRITIC_STEP_DESC_MODEL",
    "MOBILEGYM_CRITIC_STEP_DESC_PROVIDER",
    "MOBILEGYM_CRITIC_STEP_DESC_BASE_URL",
    "MOBILEGYM_CRITIC_FINAL_DECISION_MODEL",
    "MOBILEGYM_CRITIC_FINAL_DECISION_PROVIDER",
    "MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL",
    "MOBILEGYM_CRITIC_API_KEY",
    "MOBILEGYM_CRITIC_MAX_RETRIES",
]

missing_keys = [key for key in required_keys if key not in _config]
if missing_keys:
    raise ValueError(
        f"Missing required configuration keys in config.yaml: {missing_keys}"
    )

# 从config.yaml读取模型参数 - 不再提供默认值
# 最终决策模型配置
DEFAULT_MODEL = _config["MOBILEGYM_CRITIC_FINAL_DECISION_MODEL"]
DEFAULT_PROVIDER = _config["MOBILEGYM_CRITIC_FINAL_DECISION_PROVIDER"]
DEFAULT_API_URL = _config["MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL"]

# 步骤描述模型配置
DEFAULT_DESC_MODEL = _config["MOBILEGYM_CRITIC_STEP_DESC_MODEL"]
DEFAULT_DESC_PROVIDER = _config["MOBILEGYM_CRITIC_STEP_DESC_PROVIDER"]
DEFAULT_DESC_API_URL = _config["MOBILEGYM_CRITIC_STEP_DESC_BASE_URL"]

DEFAULT_MAX_TOKENS = None  # None means not specified, let the API use its default
DEFAULT_TEMPERATURE = 0.01

# 不再需要的参数
DEFAULT_APP_ID = None
DEFAULT_APP_KEY = None

# 从config.yaml读取重试参数
DEFAULT_MAX_RETRIES = _config["MOBILEGYM_CRITIC_MAX_RETRIES"]
DEFAULT_RETRY_DELAY = 2

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
