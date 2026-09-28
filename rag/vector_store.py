"""向量存储与检索（Vector Store）。

默认实现轻量内存向量库，支持稀疏（TF-IDF dict）与稠密（list[float]）两种向量，
使用余弦相似度排序召回。数据量增大后可替换为 FAISS / Milvus / pgvector，
接口保持一致即可（见 search 方法）。
"""
from __future__ import annotations

import math
from typing import Any


def cosine_sim(a: Any, b: Any) -> float:
    """余弦相似度，兼容稀疏字典与稠密列表。"""
    if not a or not b:
        return 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        inter = set(a) & set(b)
        num = sum(a[i] * b[i] for i in inter)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
    else:
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        # 稠密：逐元素相乘（长度以较短为准）
        n = min(len(a), len(b))
        num = sum(a[i] * b[i] for i in range(n))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class VectorStore:
    """极简内存向量库：add 入库，search 返回 top_k 命中。"""

    def __init__(self) -> None:
        # 每条记录: (doc_id, vector, metadata)
        self.records: list[tuple[str, Any, dict]] = []

    def add(self, doc_id: str, vector: Any, meta: dict | None = None) -> None:
        self.records.append((doc_id, vector, meta or {}))

    def search(self, query_vec: Any, top_k: int = 5) -> list[tuple[str, float, dict]]:
        scored = [
            (doc_id, cosine_sim(query_vec, vec), meta)
            for doc_id, vec, meta in self.records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.records)
