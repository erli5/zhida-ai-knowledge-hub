"""智答 RAG 核心模块。

设计目标：零第三方依赖即可运行"检索 → 答案合成"全流程。
默认使用纯 Python 实现的 TF-IDF 向量化；若安装 sentence-transformers，
可切换到语义向量（见 embedder.py）。
"""
from .chunker import chunk_text, split_into_sentences
from .embedder import TfidfVectorizer, BaseEmbedder
from .vector_store import VectorStore, cosine_sim
from .retriever import Retriever
from .llm import ExtractiveSynthesizer, BaseAnswerSynthesizer
from .pipeline import RAGPipeline

__all__ = [
    "chunk_text",
    "split_into_sentences",
    "TfidfVectorizer",
    "BaseEmbedder",
    "VectorStore",
    "cosine_sim",
    "Retriever",
    "ExtractiveSynthesizer",
    "BaseAnswerSynthesizer",
    "RAGPipeline",
]
