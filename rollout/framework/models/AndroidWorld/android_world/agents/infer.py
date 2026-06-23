# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests

from android_world.utils.vivo.vivo_api import (
    inference_chat_gemini_2_image,
    inference_chat_gemini_1_image,
    inference_chat_gemini_wo_image,
    inference_chat_gemini_multiturn, 
)


ERROR_CALLING_LLM = "Error calling LLM"


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="JPEG")
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.

        Returns:
          Text output, is_safe, and raw output.
        """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          Text output and raw output.
        """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (types.HarmBlockThreshold.BLOCK_NONE),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Gemini GCP interface."""

    def __init__(
        self,
        model_name: str | None = None,
        max_retry: int = 3,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_safety_checks: bool = True,
    ):
        if "GCP_API_KEY" not in os.environ:
            raise RuntimeError("GCP API key not set.")
        genai.configure(api_key=os.environ["GCP_API_KEY"])
        self.llm = genai.GenerativeModel(
            model_name,
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_types.GenerationConfig(
                temperature=temperature, top_p=top_p, max_output_tokens=1000
            ),
        )
        if max_retry <= 0:
            max_retry = 3
            print("Max_retry must be positive. Reset it to 3")
        self.max_retry = min(max_retry, 5)

    def predict(
        self,
        text_prompt: str,
        enable_safety_checks: bool = True,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [], enable_safety_checks, generation_config)

    def is_safe(self, raw_response):
        try:
            return (
                raw_response.candidates[0].finish_reason
                != answer_types.FinishReason.SAFETY
            )
        except Exception:  # pylint: disable=broad-exception-caught
            #  Assume safe if the response is None or doesn't have candidates.
            return True

    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray],
        enable_safety_checks: bool = True,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        counter = self.max_retry
        retry_delay = 1.0
        output = None
        while counter > 0:
            try:
                output = self.llm.generate_content(
                    [text_prompt] + [Image.fromarray(image) for image in images],
                    safety_settings=None
                    if enable_safety_checks
                    else SAFETY_SETTINGS_BLOCK_NONE,
                    generation_config=generation_config,
                )
                return output.text, True, output
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print("Error calling LLM, will retry in {retry_delay} seconds")
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2

        if (output is not None) and (not self.is_safe(output)):
            return ERROR_CALLING_LLM, False, output
        return ERROR_CALLING_LLM, None, None

    def generate(
        self,
        contents: (content_types.ContentsType | list[str | np.ndarray | Image.Image]),
        safety_settings: safety_types.SafetySettingOptions | None = None,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Any]:
        """Exposes the generate_content API.

        Args:
          contents: The input to the LLM.
          safety_settings: Safety settings.
          generation_config: Generation config.

        Returns:
          The output text and the raw response.
        Raises:
          RuntimeError:
        """
        counter = self.max_retry
        retry_delay = 1.0
        response = None
        if isinstance(contents, list):
            contents = self.convert_content(contents)
        while counter > 0:
            try:
                response = self.llm.generate_content(
                    contents=contents,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )
                return response.text, response
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print("Error calling LLM, will retry in {retry_delay} seconds")
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2
        raise RuntimeError(f"Error calling LLM. {response}.")

    def convert_content(
        self,
        contents: list[str | np.ndarray | Image.Image],
    ) -> content_types.ContentsType:
        """Converts a list of contents to a ContentsType."""
        converted = []
        for item in contents:
            if isinstance(item, str):
                converted.append(item)
            elif isinstance(item, np.ndarray):
                converted.append(Image.fromarray(item))
            elif isinstance(item, Image.Image):
                converted.append(item)
        return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
    """OpenAI GPT4 wrapper.

    Attributes:
      openai_api_key: The class gets the OpenAI api key either explicitly, or
        through env variable in which case just leave this empty.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
      model: GPT model to use based on if it is multimodal.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.0,
    ):
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        if max_retry <= 0:
            max_retry = 3
            print("Max_retry must be positive. Reset it to 3")
        self.max_retry = min(max_retry, 5)
        self.temperature = temperature
        self.model = model_name

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode("utf-8")

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            "max_tokens": 1000,
        }

        # Gpt-4v supports multiple images, just need to insert them in the content
        # list.
        for image in images:
            payload["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    },
                }
            )

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            try:
                response = requests.post(
                    "https://api.chatanywhere.tech/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.ok and "choices" in response.json():
                    return (
                        response.json()["choices"][0]["message"]["content"],
                        None,
                        response,
                    )
                print(
                    "Error calling OpenAI API with error message: "
                    + response.json()["error"]["message"]
                )
                time.sleep(wait_seconds)
                wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                wait_seconds *= 2
                counter -= 1
                print("Error calling LLM, will retry soon...")
                print(e)
        return ERROR_CALLING_LLM, None, None


class VivoGeminiWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Vivo Gemini API wrapper.

    Attributes:
      model_name: The model name to use for inference.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
    """

    def __init__(
        self,
        model_name: str,
        max_retry: int = 3,
        temperature: float = 0.01,
        config: dict = None,
        max_conversation_turns: int = 0,
        max_image_history: int = 4,  # 改为4，这样加上当前轮总共5张raw
    ):
        # Use config parameters if provided, otherwise fall back to defaults
        if config:
            self.model = config.get("M3A_MODEL", model_name)
            self.base_url = config.get("M3A_BASE_URL")
            self.api_key = config.get("M3A_API_KEY")
        else:
            self.model = model_name
            self.base_url = None
            self.api_key = None

        if max_retry <= 0:
            max_retry = 8
            print("Max_retry must be positive. Reset it to 8")
        self.max_retry = min(max_retry, 8)
        self.temperature = temperature
        self.messages = []  # 初始化对话历史
        self.max_conversation_turns = max_conversation_turns  # 最大对话轮数限制
        self.max_image_history = max_image_history  # 最大图片历史数量

    def _trim_image_history(self):
        """
        限制图片历史数量，保留最近的max_image_history张图片
        图片存储在assistant消息的_screenshot_base64字段中
        注意：max_image_history=4，这样加上当前轮总共5张raw_screenshot
        """
        if self.max_image_history <= 0:
            return  # 不限制图片数量

        # 找到所有包含图片的assistant消息
        image_count = 0
        for i in range(len(self.messages) - 1, -1, -1):  # 从后往前遍历
            msg = self.messages[i]
            if (msg.get("role") == "assistant" and
                "_screenshot_base64" in msg):
                image_count += 1
                # 如果超过限制，删除这张图片
                if image_count > self.max_image_history:
                    del msg["_screenshot_base64"]
                    print(f"删除较旧的图片，当前保留{self.max_image_history}张历史图片")

    def _trim_conversation_history(self):
        """
        根据max_conversation_turns限制对话历史长度
        保留system消息和最近k轮的对话（用户-助手对）
        注意：所有文本交互内容不会被删除，只会修剪图片
        """
        if self.max_conversation_turns <= 0 or len(self.messages) <= 1:
            return  # 不限制或消息太少，无需修剪

        # 找到system消息
        system_messages = []
        conversation_messages = []

        for msg in self.messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)

        # 计算需要保留的对话轮数
        # 每轮对话包含一个用户消息和一个助手消息
        max_messages = self.max_conversation_turns * 2

        if len(conversation_messages) > max_messages:
            # 保留最近的对话（文本内容保留，但可能需要修剪图片）
            conversation_messages = conversation_messages[-max_messages:]
            print(f"对话历史已修剪，保留最近{self.max_conversation_turns}轮对话")

        # 重新组合消息：system消息 + 最近的对话
        self.messages = system_messages + conversation_messages

        # 修剪图片历史
        self._trim_image_history()

    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_text(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Text-only prediction method for memory functionality."""
        return self.predict_mm(text_prompt, [])
    
    def predict_multiturn(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        """Multiturn prediction method."""
        # 在添加新消息前，先修剪对话历史
        self._trim_conversation_history()

        # 将当前用户输入添加到对话历史（使用OpenAI标准格式）
        self.messages.append({"role": "user", "content": text_prompt})

        # 将原图转换为base64格式保存
        raw_screenshot_base64 = None
        if images and len(images) > 0:
            # 使用第一张图片作为raw_screenshot（原始截图）
            raw_image = images[0]
            # 转换为base64
            image_pil = Image.fromarray(raw_image)
            import io
            buffered = io.BytesIO()
            image_pil.save(buffered, format="JPEG")
            raw_screenshot_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        import tempfile
        temp_image_paths = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                Image.fromarray(image).save(tmp_file, format="JPEG")
                temp_image_paths.append(tmp_file.name)

        api_response = inference_chat_gemini_multiturn(
            messages=self.messages,
            image_paths=temp_image_paths,
            model=self.model,
            temperature=self.temperature,
        )

        # 提取内容和token使用信息
        content = api_response["content"]
        usage_info = api_response.get("usage", {})

        # 将模型回复添加到对话历史
        # Assistant消息只包含文本，图片历史单独管理
        assistant_message = {
            "role": "assistant",
            "content": content
        }

        # 如果有截图，将其添加到图片历史中
        if raw_screenshot_base64:
            # 在assistant消息中添加自定义字段来存储图片（不影响API调用）
            assistant_message["_screenshot_base64"] = raw_screenshot_base64

        self.messages.append(assistant_message)

        # 修剪图片历史（确保不超过max_image_history张图片）
        self._trim_image_history()

        for path in temp_image_paths:
            if os.path.exists(path):
                os.remove(path)

        class MockResponse:
            def __init__(self, content, usage):
                self.content = content
                self.usage = usage

            def json(self):
                return {"usage": self.usage}

        # 从API响应中提取真实的token使用统计
        # 根据用户要求：输入token = promptTokens + mediaTokens, 输出token = completionTokens + thinkingTokens
        prompt_tokens = usage_info.get("promptTokens", 0) + usage_info.get("mediaTokens", 0)
        completion_tokens = usage_info.get("completionTokens", 0) + usage_info.get("thinkingTokens", 0)

        mock_response = MockResponse(
            content=content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

        return content, True, mock_response

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        # Create a mock response object to maintain compatibility with the existing code
        class MockResponse:
            def __init__(self, content, usage):
                self.content = content
                self.usage = usage

            def json(self):
                return {"usage": self.usage}

        # Split the text prompt into system and user parts
        # For simplicity, we'll use a default system prompt if not provided
        system_prompt = (
            "You are an agent who can operate an Android phone on behalf of a user."
        )
        user_prompt = text_prompt

        # Save images to temporary files if needed
        import tempfile

        temp_image_paths = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                Image.fromarray(image).save(tmp_file, format="JPEG")
                temp_image_paths.append(tmp_file.name)

        # Call the appropriate inference function based on the number of images
        # Pass base_url and api_key if configured
        call_kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model": self.model,
            "temperature": self.temperature,
        }
        
        # Add base_url and api_key if provided in config
        if self.base_url:
            call_kwargs["base_url"] = self.base_url
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        
        if len(images) == 0:
            result = inference_chat_gemini_wo_image(**call_kwargs)
        elif len(images) == 1:
            call_kwargs["image1"] = temp_image_paths[0]
            result = inference_chat_gemini_1_image(**call_kwargs)
        elif len(images) >= 2:
            # For more than 2 images, we'll just use the first two
            call_kwargs["image1"] = temp_image_paths[0]
            call_kwargs["image2"] = temp_image_paths[1]
            result = inference_chat_gemini_2_image(**call_kwargs)
        
        # Extract content and usage from the result
        content = result["content"]
        actual_usage = result["usage"]

        # Clean up temporary files
        for path in temp_image_paths:
            if os.path.exists(path):
                os.remove(path)

        # Create a mock response with actual token usage from API
        mock_response = MockResponse(
            content=content,
            usage=actual_usage,
        )

        return content, True, mock_response

    def generate(
        self,
        contents: (content_types.ContentsType | list[str | np.ndarray | Image.Image]),
        safety_settings: safety_types.SafetySettingOptions | None = None,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Any]:
        """Exposes the generate_content API.

        Args:
          contents: The input to the LLM.
          safety_settings: Safety settings.
          generation_config: Generation config.

        Returns:
          The output text and the raw response.
        Raises:
          RuntimeError:
        """
        counter = self.max_retry
        retry_delay = 1.0
        response = None
        if isinstance(contents, list):
            contents = self.convert_content(contents)
        while counter > 0:
            try:
                response = self.llm.generate_content(
                    contents=contents,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )
                return response.text, response
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print("Error calling LLM, will retry in {retry_delay} seconds")
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2
        raise RuntimeError(f"Error calling LLM. {response}.")

    def convert_content(
        self,
        contents: list[str | np.ndarray | Image.Image],
    ) -> content_types.ContentsType:
        """Converts a list of contents to a ContentsType."""
        converted = []
        for item in contents:
            if isinstance(item, str):
                converted.append(item)
            elif isinstance(item, np.ndarray):
                converted.append(Image.fromarray(item))
            elif isinstance(item, Image.Image):
                converted.append(item)
        return converted
    

