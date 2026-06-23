"""
首先用 knowledge_generation.py 从执行轨迹生成经验，
然后用本文件的 load_knowledge_raw_data 把生成的所有app的经验都读取进内存，
然后用 knowledge_raw_data_to_memory 将这些经验转成向量数据库。（也可以直接用load_memories()完成这两步）
"""

import torch
import os
import shutil
import json
from tqdm import tqdm

import gc
from utils.embedding_pipeline import SiglipMultimodalEmbeddingPipeline
from typing import List, Dict, Any, Union, Tuple

from PIL import Image
import numpy as np
import faiss
import numpy as np
from utils.utils import save_object_to_disk, load_object_from_disk


class FaissCosineSearcher:
    def __init__(
        self, vector_database_np: np.ndarray = None, embedding_dim: int = 1152
    ):
        """
        初始化Faiss余弦相似度搜索器。

        Args:
            vector_database_np (np.ndarray): n*m 的二维NumPy数组，作为向量数据库。
            embedding_dim (int): 向量的维度，默认为1152。
        """
        self.m = embedding_dim  # 等于vector_database_np.shape[1]  # 向量维度

        # 构建Faiss索引 (IndexFlatL2 用于精确搜索)
        # 对于余弦相似度，我们在归一化向量上使用L2距离索引，因为将向量归一化之后 cosine_similarity = 1 - (L2_distance^2 / 2)
        self.index = faiss.IndexFlatL2(self.m)
        if vector_database_np is not None:
            self.normalized_db = self._normalize_vectors(
                vector_database_np.astype(np.float32)
            )
            if self.normalized_db.shape[0] > 0:  # 只有当数据库非空时才添加
                self.index.add(self.normalized_db)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        对向量进行L2归一化。
        如果向量的范数为0，则返回零向量。
        """
        if vectors.ndim == 1:  # 单个向量
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else np.zeros_like(vectors)
        elif vectors.ndim == 2:  # 向量批次
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1.0, norms)  # 范数为0时用1替代，避免除以0
            normalized = vectors / safe_norms
            normalized[norms.squeeze() == 0] = 0.0  # 将原始范数为0的行设为0向量
            return normalized
        else:
            raise ValueError("vectors must be 1D or 2D NumPy arrays.")

    def add_vectors(self, new_vectors_np: np.ndarray):
        """
        向Faiss索引中添加新的向量。

        Args:
            new_vectors_np (np.ndarray): n*m 的二维NumPy数组，作为新向量。
        """
        if not isinstance(new_vectors_np, np.ndarray) or new_vectors_np.ndim != 2:
            raise ValueError("new_vectors_np must be a 2D NumPy array.")
        if new_vectors_np.shape[0] == 0:
            raise ValueError("new_vectors_np cannot be empty.")

        # 归一化新向量
        normalized_new_vectors = self._normalize_vectors(
            new_vectors_np.astype(np.float32)
        )
        self.index.add(normalized_new_vectors)

    def search(
        self, query_vector_np: np.ndarray, k: int, similarity_threshold: float
    ) -> list:
        """
        使用Faiss和余弦相似度检索top K个结果 (单个查询)。

        Args:
            query_vector_np (np.ndarray): 1*m 的一维或二维NumPy数组，作为查询向量。
            k (int): 需要检索的top K结果数量。
            similarity_threshold (float): 余弦相似度阈值 (范围通常在 -1.0 到 1.0)。低于此阈值的结果将被去除。

        Returns:
            list: 一个列表，包含满足条件的检索结果。每个结果是一个字典，
                格式为 {'index': 数据库中的原始索引, 'similarity': 余弦相似度}。
                列表按相似度降序排列。
        """
        if not isinstance(query_vector_np, np.ndarray) or query_vector_np.ndim > 2:
            raise ValueError("query_vector_np must be a 1D or 2D NumPy array.")

        if query_vector_np.ndim == 1:
            query_vector_np_batch = query_vector_np.reshape(1, -1)  # 转换为 (1, m)
        else:  # 已经是二维了
            query_vector_np_batch = query_vector_np

        results_batch = self.search_batch(
            query_vector_np_batch, k, similarity_threshold
        )
        return results_batch[0]  # 因为是单个查询，所以取第一个结果列表

    def search_batch(
        self, query_vectors_np: np.ndarray, k: int, similarity_threshold: float
    ) -> list[list[dict]]:
        """
        使用Faiss和余弦相似度批量检索top K个结果。

        Args:
            query_vectors_np (np.ndarray): b*m 的二维NumPy数组，b是查询数量，m是向量维度。
            k (int): 每个查询需要检索的top K结果数量。
            similarity_threshold (float): 余弦相似度阈值 (范围通常在 -1.0 到 1.0)。低于此阈值的结果将被去除。

        Returns:
            list[list[dict]]: 一个列表的列表。外层列表对应每个查询，内层列表包含该查询满足条件的检索结果。每个结果是一个字典，
                            格式为 {'index': 数据库中的原始索引, 'similarity': 余弦相似度}。内层列表按相似度降序排列。
        """
        if not isinstance(query_vectors_np, np.ndarray) or query_vectors_np.ndim != 2:
            raise ValueError("query_vectors_np must be a 2D NumPy array (b * m).")

        num_queries = query_vectors_np.shape[0]
        if num_queries == 0:
            return []

        if query_vectors_np.shape[1] != self.m:
            raise ValueError(
                f"Query vector dimension ({query_vectors_np.shape[1]}) does not match database vector dimension ({self.m})."
            )
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if self.index.ntotal == 0:  # 如果数据库为空
            print("警告：向量数据库为空，无法执行搜索。")
            return [[] for _ in range(num_queries)]

        normalized_queries = self._normalize_vectors(
            query_vectors_np.astype(np.float32)
        )

        is_zero_query = np.all(normalized_queries == 0, axis=1)

        clamped_threshold = max(-1.0, min(1.0, similarity_threshold))
        radius_sq = 2.0 - 2.0 * clamped_threshold
        if radius_sq < 0:
            radius_sq = 0.0

        lims, D_sq_range, I_range = self.index.range_search(
            normalized_queries, radius_sq
        )

        all_results = []
        for i in range(num_queries):
            if is_zero_query[i]:
                print(
                    f"警告：批量中的第 {i} 个查询向量归一化后为零向量，该查询返回空结果。"
                )
                all_results.append([])
                continue

            start_offset = lims[i]
            end_offset = lims[i + 1]

            query_results_above_threshold = []
            for j in range(start_offset, end_offset):
                db_index = I_range[j]
                dist_sq = D_sq_range[j]

                dist_sq = max(0.0, min(4.0, dist_sq))
                cosine_sim = 1.0 - dist_sq / 2.0
                cosine_sim = max(-1.0, min(1.0, cosine_sim))

                if cosine_sim >= clamped_threshold:
                    query_results_above_threshold.append(
                        {"index": db_index, "similarity": cosine_sim}
                    )

            # 按相似度降序排序
            query_results_above_threshold.sort(
                key=lambda x: x["similarity"], reverse=True
            )
            all_results.append(query_results_above_threshold[:k])  # 取top K

        return all_results


class KnowledgeStore:
    def __init__(
        self,
        embedding_pipeline: SiglipMultimodalEmbeddingPipeline,
        knowledge_items: list[dict] = None,
    ):
        self.embedding_pipeline = embedding_pipeline
        self.knowledge_items = knowledge_items or []
        if len(self.knowledge_items) == 0:
            dummy_image = Image.new("RGB", (224, 224), (255, 255, 255))
            dummy_embedding = np.array(
                self.embedding_pipeline([dummy_image]), dtype=np.float32
            )
            self.searcher = FaissCosineSearcher(embedding_dim=dummy_embedding.shape[1])
        else:
            images = []
            for item in self.knowledge_items:
                img = Image.fromarray(item["image"]).convert("RGB")
                images.append(img)
            database_vectors = np.array(
                self.embedding_pipeline(images), dtype=np.float32
            )
            self.searcher = FaissCosineSearcher(
                database_vectors, embedding_dim=database_vectors.shape[1]
            )

    def add_knowledge_items(self, new_knowledge_items: list[dict]):
        """
        向知识库中添加新的知识项。

        Args:
            new_knowledge_items (list[dict]): 新的知识项列表，每个知识项是一个字典。
        """
        self.knowledge_items.extend(new_knowledge_items)
        images = []
        for item in new_knowledge_items:
            img = Image.fromarray(item["image"]).convert("RGB")
            images.append(img)
        new_vectors = np.array(self.embedding_pipeline(images), dtype=np.float32)
        self.searcher.add_vectors(new_vectors)

    def search(
        self, query_image: Image.Image, k: int = 5, similarity_threshold: float = 0.5
    ) -> list[dict]:
        query_vector = np.array(self.embedding_pipeline(query_image), dtype=np.float32)
        results = self.searcher.search(query_vector, k, similarity_threshold)
        return [
            {
                "knowledge": self.knowledge_items[result["index"]]["info"],
                "similarity": result["similarity"],
                "index": result["index"],
            }
            for result in results
        ]

    def search_batch(
        self,
        query_images: list[Image.Image],
        k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> list[list[dict]]:
        query_vectors = np.array(
            self.embedding_pipeline(query_images), dtype=np.float32
        )
        results = self.searcher.search_batch(query_vectors, k, similarity_threshold)
        return [
            [
                {
                    "knowledge": str(self.knowledge_items[result["index"]]["info"]),
                    "similarity": float(result["similarity"]),
                    "index": int(result["index"]),
                }
                for result in result_list
            ]
            for result_list in results
        ]


def load_memory(knowledge_items: list[dict]) -> KnowledgeStore:
    device = os.getenv(
        "CLIENT_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    embedding_pipeline = SiglipMultimodalEmbeddingPipeline(
        model_id="google/siglip-so400m-patch14-384",
        device=device,
        server_endpoint=os.getenv("SERVER_EMBEDDING_ENDPOINT"),
    )
    memory = KnowledgeStore(
        knowledge_items=knowledge_items,
        embedding_pipeline=embedding_pipeline,
    )
    return memory


def load_knowledge_raw_data(knowledge_base_root_path: str = None):
    """
    Load knowledge raw data from the specified root path.

    Args:
        knowledge_base_root_path (str): The root path of the knowledge base.

    Returns:
        Dict[str, Any]: The raw knowledge data.
    """
    if knowledge_base_root_path is None:
        knowledge_base_root_path = os.path.abspath(
            os.getenv("KNOWLEDGE_BASE_ABSOLUTE_ROOT_PATH")
        )
    fp = os.path.join(knowledge_base_root_path, "knowledge_data.pkl")
    assert os.path.exists(fp), f"{fp} not exists."
    return load_object_from_disk(fp)


def knowledge_raw_data_to_memory(
    knowledge_raw_data: Dict[str, Any],
) -> dict[str, KnowledgeStore]:
    """Convert knowledge raw data to memory
    Args:
        knowledge_raw_data: Dict[str, Any]: knowledge raw data. Generated by function load_knowledge_raw_data.
    Returns:
        dict[str, KnowledgeStore]: memory representation of the knowledge raw data.
    """
    memories = {}
    for package_name, app_knowledge in tqdm(
        knowledge_raw_data.items(), desc="Converting knowledge raw data to memory"
    ):
        knowledge = app_knowledge["knowledge"]
        gc.collect()
        torch.cuda.empty_cache()
        memories[package_name] = load_memory(knowledge)

    print("Fusion knowledge raw data to memory. This may take a while...")
    fusion_knowledge = []
    for k, v in knowledge_raw_data.items():
        fusion_knowledge.extend(v["knowledge"])
    gc.collect()
    torch.cuda.empty_cache()
    memories["fusion"] = load_memory(fusion_knowledge)
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")
    return memories


def load_memories(
    knowledge_base_root_path: str = None,
) -> dict[str, KnowledgeStore]:
    """
    Load memories from the specified root path.

    Args:
        knowledge_base_root_path (str): The root path of the knowledge base.

    Returns:
        Dict[str, KnowledgeStore]: The loaded memories.
    """
    knowledge_raw_data = load_knowledge_raw_data(knowledge_base_root_path)
    memories = knowledge_raw_data_to_memory(knowledge_raw_data)
    return memories
