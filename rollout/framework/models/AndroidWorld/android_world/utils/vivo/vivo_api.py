# encoding: utf-8
import uuid
import json
import requests
import base64
import time
import os
import sys

from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 OPENAI_API_KEY 中
# 初始化 OpenAI 客户端，从环境变量中读取您的 API Key
client = OpenAI(
    base_url="https://your-openai-compatible-endpoint/v1",
    # 从环境变量中获取您的 API Key
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加当前目录到sys.path
sys.path.append(current_dir)

from auth_util import gen_sign_headers


# Model pricing (USD per million tokens)
MODEL_PRICING = {
    'gemini-2.5-pro': {
        'input_price_per_million': 1.25,
        'output_price_per_million': 10.0
    },
    'gemini-2.5-flash': {
        'input_price_per_million': 0.3,
        'output_price_per_million': 2.5
    },
    'gemini-1.5-pro-001': {
        'input_price_per_million': 1.25,
        'output_price_per_million': 5.0
    },
    'gemini-1.5-pro-002': {
        'input_price_per_million': 1.25,
        'output_price_per_million': 5.0
    },
    'qwen-vl-max-2025-01-25': {
        'input_price_per_million': 2.0,
        'output_price_per_million': 6.0
    },
    'qwen-vl-max-2024-12-30': {
        'input_price_per_million': 2.0,
        'output_price_per_million': 6.0
    }
}

# Default API call settings
DEFAULT_MAX_RETRIES = 200
DEFAULT_RETRY_DELAY = 2
DEFAULT_MODEL = "app-nu0fg7-1754119470355380516"
DEFAULT_MAX_TOKENS = 6500
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
        "total_tokens": total_tokens
    }

def calculate_api_cost(usage_info, model):
    """Calculate API cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model]
    extracted_usage = extract_token_usage(usage_info)
    prompt_tokens = extracted_usage['prompt_tokens']
    completion_tokens = extracted_usage['completion_tokens']
    
    input_cost = (prompt_tokens / 1_000_000) * pricing['input_price_per_million']
    output_cost = (completion_tokens / 1_000_000) * pricing['output_price_per_million']
    
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
    api_key=None,  # Not used in new format
    base_url=None,  # Not used in new format
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
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_base64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image2_base64}"}}
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = completion.choices[0].message.content
            if content:
                usage_info = completion.usage.__dict__ if hasattr(completion.usage, '__dict__') else {}
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)
                
                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost
                }
                
                return result
            else:
                print("响应内容为空")
                retry_count += 1
                print(f"将在{retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"发生异常: {str(e)}")
            retry_count += 1
            print(f"请求异常，{retry_delay}秒后进行第{retry_count}次重试...")
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
    api_key=None,  # Not used in new format
    base_url=None,  # Not used in new format
):
    """
    使用 OpenAI 兼容的客户端进行对话推理，并上传一张图片（文件路径）。
    返回服务端的回复内容，会一直重试直到成功获取有效回复。
    """
    # 从本地读取并转换图片为 Base64
    with open(image1, "rb") as f:
        image1_base64 = base64.b64encode(f.read()).decode("utf-8")

    retry_count = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_base64}"}}
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = completion.choices[0].message.content
            if content:
                usage_info = completion.usage.__dict__ if hasattr(completion.usage, '__dict__') else {}
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)
                
                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost
                }
                
                return result
            else:
                print("响应内容为空")
                retry_count += 1
                print(f"将在{retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"发生异常: {str(e)}")
            retry_count += 1
            print(f"请求异常，{retry_delay}秒后进行第{retry_count}次重试...")
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
    api_key=None,  # Not used in new format
    base_url=None,  # Not used in new format
):
    """
    使用 OpenAI 兼容的客户端进行对话推理，不传入图片。
    返回服务端的回复内容，会一直重试直到成功获取有效回复。
    """
    retry_count = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = completion.choices[0].message.content
            if content:
                usage_info = completion.usage.__dict__ if hasattr(completion.usage, '__dict__') else {}
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)
                
                result = {
                    "content": content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost
                }
                
                return result
            else:
                print("响应内容为空")
                retry_count += 1
                print(f"将在{retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"发生异常: {str(e)}")
            retry_count += 1
            print(f"请求异常，{retry_delay}秒后进行第{retry_count}次重试...")
            time.sleep(retry_delay)
            continue

def inference_chat_gemini_multi_images(
    system_prompt,
    user_prompt,
    image_list,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
    model=DEFAULT_MODEL,
    provider=None,  # Not used in new format
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=0.01,
    app_id=None,  # Not used in new format
    api_key=None,  # Not used in new format
    base_url=None,  # Not used in new format
    max_images=None,  # 设置为None表示不限制图片数量
):
    """
    使用 OpenAI 兼容的客户端进行对话推理，支持传入多张图片。
    该函数可以处理多种图片输入格式：
    1. 列表内容为图片路径字符串
    2. 列表内容为base64格式图片
    3. 列表内容为 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}} 格式

    注意：不同版本的模型可能对图片数量有不同的限制，请根据实际情况设置max_images参数

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        image_list: 图片列表，可以是文件路径列表或包含base64编码的图片数据的列表
        max_retries: 最大重试次数
        retry_delay: 重试间隔时间(秒)
        model: 模型名称
        provider: 提供商 (稳定扩展参数，实际不使用)
        max_tokens: 最大生成token数
        temperature: 温度参数
        app_id: 应用ID (稳定扩展参数，实际不使用)
        api_key: API密钥 (稳定扩展参数，实际不使用)
        base_url: API地址 (稳定扩展参数，实际不使用)
        max_images: 最大处理图片数量，默认为None（不限制）。如API有限制，可设置具体数值。

    Returns:
        模型生成的回复内容
    """
    if not image_list:  # 如果没有图片，调用无图片的接口
        return inference_chat_gemini_wo_image(
            system_prompt,
            user_prompt,
            max_retries,
            retry_delay,
            model,
            provider,
            max_tokens,
            temperature,
            app_id,
            api_key,
            base_url,
        )

    # 处理不同格式的图片列表
    processed_images = []

    for i, img in enumerate(image_list):
        # 如果设置了max_images限制且已经达到限制，则停止处理更多图片
        if max_images is not None and i >= max_images:
            print(f"警告：根据设置的限制，最多处理{max_images}张图片，忽略剩余图片")
            break

        # 判断图片格式并处理
        if isinstance(img, dict) and "image_url" in img and "url" in img["image_url"]:
            # 格式为 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "high"}}
            # 提取base64部分
            base64_data = (
                img["image_url"]["url"].split(",")[1]
                if "," in img["image_url"]["url"]
                else img["image_url"]["url"]
            )
            processed_images.append(base64_data)
        elif isinstance(img, str):
            if img.startswith("data:image"):
                # 图片是base64字符串
                base64_data = img.split(",")[1] if "," in img else img
                processed_images.append(base64_data)
            elif os.path.exists(img):
                # 图片是文件路径
                with open(img, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                    processed_images.append(base64_data)
            else:
                print(f"警告：无法处理图片 {img}，跳过")
        else:
            print(f"警告：不支持的图片格式 {type(img)}，跳过")

    # 如果没有处理出有效的图片，调用无图片的接口
    if not processed_images:
        print("警告：没有有效的图片，切换到无图片模式")
        return inference_chat_gemini_wo_image(
            system_prompt,
            user_prompt,
            max_retries,
            retry_delay,
            model,
            provider,
            max_tokens,
            temperature,
            app_id,
            api_key,
            base_url,
        )

    # 构建消息内容，首先是文本
    content = [{"type": "text", "text": user_prompt}]

    # 添加图片
    for base64_data in processed_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}
        })

    retry_count = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            response_content = completion.choices[0].message.content
            if response_content:
                usage_info = completion.usage.__dict__ if hasattr(completion.usage, '__dict__') else {}
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)

                result = {
                    "content": response_content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost
                }

                return result
            else:
                print("响应内容为空")
                retry_count += 1
                print(f"将在{retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"发生异常: {str(e)}")
            retry_count += 1
            print(f"请求异常，{retry_delay}秒后进行第{retry_count}次重试...")
            time.sleep(retry_delay)
            continue


def inference_chat_gemini_multiturn(
    messages,
    image_paths=None,
    max_retries=DEFAULT_MAX_RETRIES,
    retry_delay=DEFAULT_RETRY_DELAY,
    model=DEFAULT_MODEL,
    provider=None,  # Not used in new format
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=0.01,
    app_id=None,  # Not used in new format
    api_key=None,  # Not used in new format
    base_url=None,  # Not used in new format
):
    """
    使用 OpenAI 兼容的客户端进行多轮对话推理。
    返回服务端的回复内容，会一直重试直到成功获取有效回复。

    Args:
        messages: 对话历史列表
        session_id: 会话ID (稳定扩展参数，实际不使用)
        image_paths: (可选) 图片路径列表或单个图片路径
        max_retries: 最大重试次数 (稳定扩展参数，实际不使用)
        retry_delay: 重试间隔时间(秒)
        model: 模型名称
        provider: 提供商 (稳定扩展参数，实际不使用)
        max_tokens: 最大生成token数
        temperature: 温度参数
        app_id: 应用ID (稳定扩展参数，实际不使用)
        api_key: API密钥 (稳定扩展参数，实际不使用)
        base_url: API地址 (稳定扩展参数，实际不使用)
    """
    # 创建消息列表的本地副本，以避免修改原始列表
    request_messages = []

    # 转换消息格式为OpenAI兼容格式
    for msg in messages:
        if isinstance(msg, dict):
            # 检查是否是旧格式的消息（带contentType字段）
            if "contentType" in msg:
                # 旧格式：{"role": "user", "content": "...", "contentType": "text/image"}
                if msg.get("contentType") == "text":
                    # 检查content是否是数组格式（包含图片的多模态消息）
                    if isinstance(msg.get("content"), list):
                        # 已经是OpenAI标准的多模态格式，直接复制（去掉contentType）
                        request_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    else:
                        # 普通文本消息
                        request_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                elif msg.get("contentType") == "image":
                    # 旧的独立图片消息格式，需要添加到最后一条用户消息中
                    if request_messages and request_messages[-1]["role"] == "user":
                        # 将文本消息转换为多模态格式
                        if isinstance(request_messages[-1]["content"], str):
                            request_messages[-1]["content"] = [
                                {"type": "text", "text": request_messages[-1]["content"]}
                            ]
                        # 添加图片
                        request_messages[-1]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": msg["content"]}
                        })
                    else:
                        # 如果没有用户消息，创建一个新的
                        request_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": msg["content"]}}
                            ]
                        })
            else:
                # 新格式（OpenAI标准格式）：直接复制
                request_messages.append(msg)
        else:
            # 如果不是字典，假设是字符串消息
            request_messages.append({"role": "user", "content": str(msg)})

    # 收集历史图片（从assistant消息中的_screenshot_base64字段）
    historical_images = []
    for msg in messages:
        if (isinstance(msg, dict) and
            msg.get("role") == "assistant" and
            "_screenshot_base64" in msg):
            historical_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{msg['_screenshot_base64']}"}
            })

    # 处理额外的图片路径（当前轮的图片）
    image_contents = []
    if image_paths:
        if isinstance(image_paths, str):
            image_paths = [image_paths]  # 统一处理为列表

        for image_path in image_paths:
            try:
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
                # 构建图片内容
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            except FileNotFoundError:
                print(f"警告：图片文件未找到: {image_path}")
            except Exception as e:
                print(f"警告：处理图片 {image_path} 时出错: {e}")

    # This block handles the injection of historical screenshots and adds descriptive text.
    # It assumes the most recent user message in `request_messages` already contains the current turn's images.

    # Find the last user message to modify.
    last_user_idx = -1
    for i in range(len(request_messages) - 1, -1, -1):
        if request_messages[i]["role"] == "user":
            last_user_idx = i
            break

    if last_user_idx >= 0:
        # Build the descriptive text for historical images.
        image_description = ""
        if historical_images:
            image_description += f"\n\nNote: The following {len(historical_images)} images are historical screenshots from previous steps."

        # Extract original text and current images from the last user message.
        original_content = request_messages[last_user_idx]["content"]
        original_text = ""
        current_images = []
        if isinstance(original_content, str):
            original_text = original_content
        elif isinstance(original_content, list):
            for part in original_content:
                if part.get("type") == "text":
                    original_text = part.get("text", "")
                elif part.get("type") == "image_url":
                    current_images.append(part)

        enhanced_text = original_text + image_description

        # Reconstruct the message content with historical images first, then the current content.
        # This correctly orders the context for the model and prevents duplication.
        new_content = [{"type": "text", "text": enhanced_text}] + historical_images + current_images
        request_messages[last_user_idx]["content"] = new_content

    retry_count = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=request_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            response_content = completion.choices[0].message.content
            if response_content:
                usage_info = completion.usage.__dict__ if hasattr(completion.usage, '__dict__') else {}
                extracted_usage = extract_token_usage(usage_info)
                api_cost = calculate_api_cost(usage_info, model)

                result = {
                    "content": response_content,
                    "usage": extracted_usage,
                    "model": model,
                    "provider": "openai_compatible",
                    "api_cost": api_cost
                }

                return result
            else:
                print("响应内容为空")
                retry_count += 1
                print(f"将在{retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"发生异常: {str(e)}")
            retry_count += 1
            print(f"请求异常，{retry_delay}秒后进行第{retry_count}次重试...")
            time.sleep(retry_delay)
            continue
