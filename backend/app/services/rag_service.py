"""业务服务层：封装 RAG 知识库的建库、查询与文档管理。

把"算法细节"与"接口层"解耦，接口层只依赖本服务的公开方法。
"""
from __future__ import annotations

import os

from backend.app.config import RAG_TOP_K, SAMPLE_DOCS_DIR
from rag.pipeline import RAGPipeline


class RAGService:
    def __init__(self) -> None:
        self.pipeline = RAGPipeline(top_k=RAG_TOP_K)
        self.documents: list[dict] = []
        self._load_samples()

    def _load_samples(self) -> None:
        docs: list[dict] = []
        if os.path.isdir(SAMPLE_DOCS_DIR):
            for fname in sorted(os.listdir(SAMPLE_DOCS_DIR)):
                if fname.endswith((".txt", ".md")):
                    path = os.path.join(SAMPLE_DOCS_DIR, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append({"doc_id": fname, "text": text, "source": fname})
        self.documents = docs
        if docs:
            self.pipeline.index(docs)

    def list_documents(self) -> list[dict]:
        return [
            {
                "doc_id": d["doc_id"],
                "source": d.get("source", d["doc_id"]),
                "length": len(d["text"]),
            }
            for d in self.documents
        ]

    def add_document(self, source: str, text: str) -> str:
        doc_id = source or f"doc_{len(self.documents) + 1}"
        self.documents.append(
            {"doc_id": doc_id, "text": text, "source": source or doc_id}
        )
        # 数据量小，重建索引即可；生产可改为增量建库
        self.pipeline.index(self.documents)
        return doc_id

    def ask(self, question: str) -> dict:
        return self.pipeline.query(question)


# 单例服务（应用生命周期内共享）
rag_service = RAGService()
