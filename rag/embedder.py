"""向量化（Embedder）。

默认实现：纯 Python 的 TF-IDF 向量化器，零依赖、零下载、可离线运行，
适合面试现场演示。同时提供 BaseEmbedder 抽象基类，方便切换到
sentence-transformers 等语义向量模型（见文件底部示例）。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英文混合分词：英文/数字按词，中文按单字（适合纯 Python TF-IDF）。

    中文按字切分可在不引入 jieba 的情况下获得可用的词频统计；
    英文单词与数字保持完整 token，避免语义被切碎。
    """
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    tokens += _CJK_RE.findall(text)
    return tokens


class BaseEmbedder:
    """嵌入模型抽象接口。子类需实现 embed(texts) -> list[vector]。"""

    def embed(self, texts: list[str]) -> list:
        raise NotImplementedError

    def embed_query(self, text: str):
        return self.embed([text])[0]


class TfidfVectorizer(BaseEmbedder):
    """从零实现的 TF-IDF 向量化器（稀疏向量，键为词表下标）。

    优点：无需任何第三方库、无需下载模型、确定性强。
    缺点：无语义能力（近义词不敏感），生产可替换为句向量。
    """

    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, documents: list[str]):
        n = len(documents)
        df: Counter = Counter()
        for doc in documents:
            for term in set(tokenize(doc)):
                df[term] += 1
        vocab_list = [t for t, _ in df.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(vocab_list)}
        self.idf = {
            t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab
        }
        self._fitted = True
        return self

    def _vector(self, text: str) -> dict[int, float]:
        counts = Counter(tokenize(text))
        length = sum(counts.values()) or 1
        vec: dict[int, float] = {}
        for term, c in counts.items():
            if term in self.vocab:
                idx = self.vocab[term]
                tf = c / length
                vec[idx] = tf * self.idf[term]
        return vec

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        if not self._fitted:
            raise RuntimeError("向量化器需要先 fit 才能 transform")
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> dict[int, float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """可选的语义向量实现（需安装 sentence-transformers）。

    用法：
        from rag.embedder import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5")
        vec = emb.embed_query("你好")
    该实现返回稠密向量（list[float]），可与 VectorStore 配合（需支持稠密向量检索）。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "使用语义向量需先安装：pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
