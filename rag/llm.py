"""答案合成（Answer Synthesis）。

默认实现：抽取式合成器（ExtractiveSynthesizer）。
从检索到的相关 chunk 中挑选与问题最相关的句子，拼接成答案，
并保留来源引用 —— 这样答案"有据可依"，显著降低幻觉。

同时提供 BaseAnswerSynthesizer 抽象接口，便于接入真实 LLM
（本地 HuggingFace 模型或 OpenAI 兼容 API），pipeline 无需改动。
"""
from __future__ import annotations

from .chunker import split_into_sentences
from .embedder import tokenize


def _query_terms(query: str) -> set[str]:
    return set(tokenize(query))


def _sentence_relevance(sentence: str, terms: set[str]) -> int:
    s = set(tokenize(sentence))
    return sum(1 for t in terms if t in s)


class BaseAnswerSynthesizer:
    """答案合成抽象接口。子类实现 synthesize(question, results) -> dict。"""

    def synthesize(self, question: str, results: list) -> dict:
        raise NotImplementedError


class ExtractiveSynthesizer(BaseAnswerSynthesizer):
    """抽取式答案合成：离线可用、答案可溯源。"""

    def synthesize(self, question: str, results: list) -> dict:
        terms = _query_terms(question)
        candidates: list[tuple[float, str, str]] = []
        for _doc_id, score, meta in results:
            for sent in split_into_sentences(meta.get("text", "")):
                rel = _sentence_relevance(sent, terms)
                if rel > 0:
                    # 相关度 = 句内命中词数 * (0.5 + 检索得分)
                    candidates.append((rel * (0.5 + score), sent, meta.get("source", "")))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:3]
        if not top:
            return {
                "answer": "抱歉，知识库中未找到与问题相关的内容。",
                "sources": [],
            }
        answer = "；".join(sent for _, sent, _ in top) + "。"
        # 去重保留顺序
        sources = list(dict.fromkeys(src for _, _, src in top if src))
        return {"answer": answer, "sources": sources}


class LLMAnswerSynthesizer(BaseAnswerSynthesizer):
    """可选：接入真实 LLM 做生成式答案（需配置模型或 API）。

    演示用法（伪代码）：
        synth = LLMAnswerSynthesizer(prompt_builder=..., generate_fn=call_llm)
        synth.synthesize(question, results)
    generate_fn 接收拼接好的 (context, question)，返回模型生成的文本。
    """

    def __init__(self, generate_fn=None, prompt_builder=None):
        self.generate_fn = generate_fn
        self.prompt_builder = prompt_builder or self._default_prompt

    @staticmethod
    def _default_prompt(question: str, contexts: list[str]) -> str:
        ctx = "\n".join(f"- {c}" for c in contexts)
        return (
            "你是一个严谨的问答助手，只根据下列资料回答，不得编造。\n"
            f"资料：\n{ctx}\n\n问题：{question}\n回答："
        )

    def synthesize(self, question: str, results: list) -> dict:
        if self.generate_fn is None:
            raise RuntimeError("LLMAnswerSynthesizer 需要传入 generate_fn")
        contexts = [m.get("text", "") for _, _, m in results]
        prompt = self.prompt_builder(question, contexts)
        answer = self.generate_fn(prompt)
        sources = list(dict.fromkeys(m.get("source", "") for _, _, m in results if m.get("source")))
        return {"answer": answer, "sources": sources}
