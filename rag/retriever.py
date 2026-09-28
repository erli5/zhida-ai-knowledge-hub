"""检索器（Retriever）。

把 chunker / embedder / vector_store 组合成"建索引 + 查询"的统一入口。
"""
from __future__ import annotations

from .chunker import chunk_text
from .embedder import TfidfVectorizer
from .vector_store import VectorStore


class Retriever:
    """文档检索器：build_index 建立索引，search 返回相关 chunk。"""

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.store = VectorStore()
        self.fitted = False

    def build_index(self, documents: list[dict]) -> None:
        """建立索引。

        Args:
            documents: 形如 [{"doc_id": str, "text": str, "source": str}, ...]
        """
        all_chunks: list[str] = []
        meta: list[dict] = []
        for doc in documents:
            chunks = chunk_text(doc["text"])
            for idx, ch in enumerate(chunks):
                all_chunks.append(ch)
                meta.append(
                    {
                        "doc_id": doc["doc_id"],
                        "source": doc.get("source", doc["doc_id"]),
                        "chunk_index": idx,
                        "text": ch,
                    }
                )
        self.vectorizer.fit(all_chunks)
        vecs = self.vectorizer.embed(all_chunks)
        self.store = VectorStore()
        for m, v in zip(meta, vecs):
            self.store.add(m["doc_id"], v, m)
        self.fitted = True

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        if not self.fitted:
            raise RuntimeError("请先调用 build_index 建立索引")
        qvec = self.vectorizer.embed_query(query)
        return self.store.search(qvec, top_k=top_k)
