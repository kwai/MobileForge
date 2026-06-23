"""
使用方式： python -m utils.retrieval
"""

import copy
from typing import Any, List, Dict, Tuple, Union
from PIL import Image
from utils.utils import str_to_md5
import os

import uuid
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.memory import load_memories, KnowledgeStore

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

__KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH = None
__MEMORY: dict[str, KnowledgeStore] = {}

import base64
import io
from PIL import Image


def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL Image to a base64 encoded string.

    Args:
        pil_image (Image.Image): The PIL Image object.

    Returns:
        str: The base64 encoded string.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="WEBP", quality=95)
    return base64.b64encode(buffered.getvalue()).decode()


def ndarray_to_base64(ndarray) -> str:
    """Convert a numpy array to a base64 encoded string.

    Args:
        ndarray (np.ndarray): The numpy array.

    Returns:
        str: The base64 encoded string.
    """
    return pil_to_base64(Image.fromarray(ndarray))


import requests
import time
import os


def retrieval_api(
    query: Image.Image, top_k: int = 1, threshold: float = 0.9, package_name: str = None
) -> list[dict]:  # NOTE:将这个函数复制到需要调用retrieval_api的地方即可
    """检索出query对应的knowledge

    Returns:
        List[dict[str,Any]]: 返回的结果列表（注意长度可能小于top_k）
    """
    ret, rsp, max_retries = None, None, 3
    data = {
        "package_name": package_name,  # "com.example.app",
        "query": pil_to_base64(query),  # "base64 image",
        "top_k": top_k,
        "threshold": threshold,
    }
    url = os.getenv("RAG_SERVER_ENDPOINT", "http://localhost:8769") + "/retrieval"
    for i in range(max_retries):
        try:
            rsp = requests.post(url, json=data, timeout=300)
            ret = rsp.json()
            return ret["results"]
        except Exception as e:
            print(f"retrieval_api error: {e} retrying {i+1}/{max_retries}")
            if i == max_retries - 1:
                raise e
            time.sleep(1)


def retrieval_batch_api(
    queries: list[Image.Image],
    top_k: int = 1,
    threshold: float = 0.9,
    package_name: str = None,
) -> list[list[dict]]:
    """检索出query对应的knowledge

    Returns:
        List[List[dict[str,Any]]]: 返回的结果列表（注意长度可能小于top_k）
    """
    ret, rsp, max_retries = None, None, 3
    data = {
        "package_name": package_name,  # "com.example.app",
        "queries": [pil_to_base64(query) for query in queries],  # "base64 image",
        "top_k": top_k,
        "threshold": threshold,
    }
    url = os.getenv("RAG_SERVER_ENDPOINT", "http://localhost:8769") + "/retrieval_batch"
    for i in range(max_retries):
        try:
            rsp = requests.post(url, json=data, timeout=300)
            ret = rsp.json()
            return ret["results"]
        except Exception as e:
            print(f"retrieval_api error: {e} retrying {i+1}/{max_retries}")
            if i == max_retries - 1:
                raise e
            time.sleep(1)


def base64_to_pil(base64_str: str) -> Image.Image:
    """Convert a base64 encoded string to a PIL Image.

    Args:
        base64_str (str): The base64 string representing the image.

    Returns:
        Image.Image: A PIL Image object.
    """
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


@app.post("/retrieval")
async def retrieval(request: Request):
    """
    body: {
        "package_name": "com.example.app",  # 在哪个app对应的知识库中检索，如果不指定包名就在所有的知识库中检索
        "query": "base64 image",  # base64编码的图片
        "top_k": 1,  # 返回的结果数量 Optional
        "threshold": 0  # 距离或者相似度的阈值
        "similarity": "cosine" or "l2"  # 相似度计算方式（目前暂时限定为cosine） Optional TODO:等待后续支持l2
    }

    response: {
        "results": List[dict[str,Any]]  # 返回的结果列表（注意长度可能小于top_k）。{"knowledge": str,"similarity": float,}
    }
    """
    try:
        # 从请求中解析原始 JSON
        data = await request.json()
        query = data.get("query", None)
        top_k = data.get("top_k", 1)
        package_name = data.get("package_name", None)
        threshold = data.get("threshold", 0.9)
        # similarity = data.get("similarity", "l2") #TODO:等待后续支持l2
        result = {"results": []}
        if query is not None:
            memory = __MEMORY["fusion"]
            if package_name is not None:
                if package_name in __MEMORY:
                    memory = __MEMORY[package_name]

            if memory is not None:
                result["results"] = memory.search(
                    base64_to_pil(query), top_k, threshold
                )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieval_batch")
async def retrieval_batch(request: Request):
    """
    批量检索知识库中的信息
    body: {
        "package_name": "com.example.app",  # 在哪个app对应的知识库中检索，如果不指定包名就在所有的知识库中检索
        "queries": ["base64 image"],  # base64编码的图片
        "top_k": 1,  # 返回的结果数量 Optional
        "threshold": 0  # 距离或者相似度的阈值
        "similarity": "cosine" or "l2"  # 相似度计算方式（目前暂时限定为cosine） Optional TODO:等待后续支持l2
    }

    response: {
        "results": List[List[dict[str,Any]]]  # 返回的结果列表（注意长度可能小于top_k）。{"knowledge": str,"similarity": float,}
    }
    """
    try:
        # 从请求中解析原始 JSON
        data = await request.json()
        queries = data.get("queries", [])
        top_k = data.get("top_k", 1)
        package_name = data.get("package_name", None)
        threshold = data.get("threshold", 0.9)
        # similarity = data.get("similarity", "l2") #TODO:等待后续支持l2
        result = {"results": []}
        if queries:
            memory = __MEMORY["fusion"]
            if package_name is not None:
                if package_name in __MEMORY:
                    memory = __MEMORY[package_name]

            if memory is not None:
                pil_images = [base64_to_pil(query) for query in queries]
                res = memory.search_batch(pil_images, top_k, threshold)
                result["results"] = res
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    Usage: python -m utils.retrieval
    """

    __KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH = os.getenv("KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH")

    if __KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH is not None and os.path.exists(
        __KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH
    ):
        print(f"Using knowledge base at {__KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH}")
    else:
        print(
            f"WARNING: No knowledge base found at {__KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH}, please set KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH in environment variable or .env file"
        )
        exit(1)

    os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"
    print("Retrieval Service")
    print("Loading Memory...")
    __MEMORY = load_memories(__KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH)

    print("Fast API is starting")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8769, timeout_graceful_shutdown=3)

    exit(0)
