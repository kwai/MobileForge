# encoding: utf-8
import os
import sys
import uuid
import json
import base64
import time
from openai import OpenAI

# Add project root to path for config_loader import
_project_root = os.path.join(os.path.dirname(__file__), "../../..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config_loader import get_config


try:
    # Use get_config() to get cached config with mode presets applied
    _config = get_config(verbose=False)
    MOBILEGYM_CRITIC_API_KEY = _config.get("MOBILEGYM_CRITIC_API_KEY")
    # Step description config (BASE_URL already resolved from mode presets)
    MOBILEGYM_CRITIC_STEP_DESC_BASE_URL = _config.get("MOBILEGYM_CRITIC_STEP_DESC_BASE_URL")
    MOBILEGYM_CRITIC_STEP_DESC_MODEL = _config.get("MOBILEGYM_CRITIC_STEP_DESC_MODEL")
    # Final decision config (BASE_URL already resolved from mode presets)
    MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL = _config.get("MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL")
    MOBILEGYM_CRITIC_FINAL_DECISION_MODEL = _config.get("MOBILEGYM_CRITIC_FINAL_DECISION_MODEL")
except Exception as e:
    # If config loading fails, fallback to environment variables
    print(
        f"Warning: Failed to load config.yaml, falling back to environment variables: {e}"
    )
    MOBILEGYM_CRITIC_API_KEY = os.environ.get("MOBILEGYM_CRITIC_API_KEY")
    MOBILEGYM_CRITIC_STEP_DESC_BASE_URL = (
        "https://your-openai-compatible-endpoint/v1"
    )
    MOBILEGYM_CRITIC_STEP_DESC_MODEL = None
    MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL = (
        "https://your-openai-compatible-endpoint/v1"
    )
    MOBILEGYM_CRITIC_FINAL_DECISION_MODEL = None

if not MOBILEGYM_CRITIC_API_KEY:
    raise ValueError(
        "MOBILEGYM_CRITIC_API_KEY not found in config.yaml or environment variables"
    )

# 客户端缓存，避免重复创建
_client_cache = {}


def _get_client(base_url):
    """获取或创建指定 base_url 的 OpenAI 客户端"""
    if base_url not in _client_cache:
        _client_cache[base_url] = OpenAI(base_url=base_url, api_key=MOBILEGYM_CRITIC_API_KEY)
    return _client_cache[base_url]


# 默认客户端
client = _get_client(MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL)

# Model pricing (USD per million tokens)
MODEL_PRICING = {
    "gemini-2.5-pro": {
        "input_price_per_million": 1.25,
        "output_price_per_million": 10.0,
    },
    "gemini-2.5-flash": {
        "input_price_per_million": 0.3,
        "output_price_per_million": 2.5,
    },
    "gemini-1.5-pro-001": {
        "input_price_per_million": 1.25,
        "output_price_per_million": 5.0,
    },
    "gemini-1.5-pro-002": {
        "input_price_per_million": 1.25,
        "output_price_per_million": 5.0,
    },
    "qwen-vl-max-2025-01-25": {
        "input_price_per_million": 2.0,
        "output_price_per_million": 6.0,
    },
    "qwen-vl-max-2024-12-30": {
        "input_price_per_million": 2.0,
        "output_price_per_million": 6.0,
    },
}

# Default API call settings
DEFAULT_MAX_RETRIES = 200
DEFAULT_RETRY_DELAY = 2
DEFAULT_MODEL = "app-nu0fg7-1754119470355380516"
DEFAULT_MAX_TOKENS = None  # None means not specified, let the API use its default
DEFAULT_TEMPERATURE = 0.01


def extract_token_usage(usage_info):
    """Extract token usage from API response."""
    if not usage_info:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = usage_info.get("prompt_tokens", 0)
    completion_tokens = usage_info.get("completion_tokens", 0)
    total_tokens = usage_info.get("total_tokens", prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def calculate_api_cost(usage_info, model):
    """Calculate API cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    extracted_usage = extract_token_usage(usage_info)
    prompt_tokens = extracted_usage["prompt_tokens"]
    completion_tokens = extracted_usage["completion_tokens"]

    input_cost = (prompt_tokens / 1_000_000) * pricing["input_price_per_million"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_price_per_million"]

    return input_cost + output_cost


def inference_chat_gemini_2_image(
    system_prompt,
    user_prompt,
    image1,
    image2,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
    model=DEFAULT_MODEL,
    provider=None,  # Not used in new format
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=0.01,
    app_id=None,  # Not used in new format
    app_key=None,  # Not used in new format
    api_url=None,  # Not used in new format
):
    """
    使用 OpenAI 兼容的客户端进行对话推理，并上传两张图片（文件路径）。
    返回服务端的回复内容，会一直重试直到成功获取有效回复。
    """
    # 从本地读取并转换图片为 Base64
    with open(image1, "rb") as f:
        image1_base64 = base64.b64encode(f.read()).decode("utf-8")

    with open(image2, "rb") as f:
        image2_base64 = base64.b64encode(f.read()).decode("utf-8")

    retry_count = 0
    while True:
        try:
            # Build API call kwargs, only include max_tokens if specified
            api_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image1_base64}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image2_base64}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens

            completion = client.chat.completions.create(**api_kwargs)

            content = completion.choices[0].message.content
            if content:
                usage_info = (
                    completion.usage.__dict__
                    if hasattr(completion.usage, "__dict__")
                    else {}
                )
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)

                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost,
                }

                return result
            else:
                print("Response content is empty")
                retry_count += 1
                print(f"Will retry in {retry_delay} seconds, attempt {retry_count}...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            retry_count += 1
            print(f"Request failed, will retry in {retry_delay} seconds, attempt {retry_count}...")
            time.sleep(retry_delay)
            continue


def inference_chat_gemini_1_image(
    system_prompt,
    user_prompt,
    image1,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
    model=DEFAULT_MODEL,
    provider=None,  # Not used in new format
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    app_id=None,  # Not used in new format
    app_key=None,  # Not used in new format
    api_url=None,  # 支持指定 base_url
):
    """
    使用 OpenAI 兼容的客户端进行对话推理，并上传一张图片（文件路径）。
    返回服务端的回复内容，会一直重试直到成功获取有效回复。

    Args:
        api_url: 可选，指定 API 的 base_url。如果不指定，使用默认的 MOBILEGYM_CRITIC_FINAL_DECISION_BASE_URL
    """
    # 根据 api_url 获取对应的客户端
    if api_url:
        api_client = _get_client(api_url)
    else:
        api_client = client

    # 从本地读取并转换图片为 Base64
    with open(image1, "rb") as f:
        image1_base64 = base64.b64encode(f.read()).decode("utf-8")

    retry_count = 0
    while True:
        try:
            # Build API call kwargs, only include max_tokens if specified
            api_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image1_base64}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens

            completion = api_client.chat.completions.create(**api_kwargs)

            content = completion.choices[0].message.content
            if content:
                usage_info = (
                    completion.usage.__dict__
                    if hasattr(completion.usage, "__dict__")
                    else {}
                )
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)

                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost,
                }

                return result
            else:
                print("Response content is empty")
                retry_count += 1
                print(f"Will retry in {retry_delay} seconds, attempt {retry_count}...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            retry_count += 1
            print(f"Request failed, will retry in {retry_delay} seconds, attempt {retry_count}...")
            time.sleep(retry_delay)
            continue


def inference_chat_gemini_wo_image(
    system_prompt,
    user_prompt,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
    model=DEFAULT_MODEL,
    provider=None,  # Not used in new format
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=0.01,
    app_id=None,  # Not used in new format
    app_key=None,  # Not used in new format
    api_url=None,  # Not used in new format
):
    """
    Use OpenAI compatible client for chat inference without images.
    Returns server response content, will retry until valid response is obtained.
    """
    retry_count = 0
    while True:
        try:
            # Build API call kwargs, only include max_tokens if specified
            api_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                api_kwargs["max_tokens"] = max_tokens

            completion = client.chat.completions.create(**api_kwargs)

            content = completion.choices[0].message.content
            if content:
                usage_info = (
                    completion.usage.__dict__
                    if hasattr(completion.usage, "__dict__")
                    else {}
                )
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)

                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost,
                }

                return result
            else:
                print("Response content is empty")
                retry_count += 1
                print(f"Will retry in {retry_delay} seconds, attempt {retry_count}...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            retry_count += 1
            print(f"Request failed, will retry in {retry_delay} seconds, attempt {retry_count}...")
            time.sleep(retry_delay)
            continue
