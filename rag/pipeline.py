"""RAG 端到端流水线（Pipeline）。

把 Retriever（检索）与 AnswerSynthesizer（答案合成）组合成统一接口：
index(documents) 建库，query(question) 问答。
"""
from __future__ import annotations

from .llm import BaseAnswerSynthesizer, ExtractiveSynthesizer
from .retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        top_k: int = 5,
        synthesizer: BaseAnswerSynthesizer | None = None,
    ):
        self.retriever = Retriever()
        self.synthesizer = synthesizer or ExtractiveSynthesizer()
        self.top_k = top_k

    def index(self, documents: list[dict]) -> None:
        """建立知识库索引。documents: [{"doc_id","text","source"}]"""
        self.retriever.build_index(documents)

    def query(self, question: str) -> dict:
        """问答。返回 {"question","answer","sources","contexts"}。"""
        results = self.retriever.search(question, top_k=self.top_k)
        out = self.synthesizer.synthesize(question, results)
        contexts = [m.get("text", "") for _, _, m in results]
        return {
            "question": question,
            "answer": out["answer"],
            "sources": out["sources"],
            "contexts": contexts,
        }
