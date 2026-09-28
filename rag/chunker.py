"""文本切片（chunking）。

把长文档切成适合检索的小块。支持：
- 按句子切分（中文标点友好）
- 控制每块最大字符数，并在相邻块之间保留 overlap 重叠字符，减少边界信息丢失。
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_into_sentences(text: str) -> list[str]:
    """将文本按中英文句末标点切分成句子列表。"""
    raw = _SENT_SPLIT.split(text)
    return [s.strip() for s in raw if s and s.strip()]


def chunk_text(text: str, max_chars: int = 300, overlap: int = 50) -> list[str]:
    """按句子累积成块，单块长度控制在 max_chars 附近；块间保留 overlap 字符重叠。

    Args:
        text: 原始文档文本。
        max_chars: 单块最大字符数。
        overlap: 相邻块之间重叠的字符数（取前一块末尾）。
    """
    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += sent
        else:
            if current:
                chunks.append(current)
            # 超长单句单独处理
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i : i + max_chars])
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)

    # 重叠处理：每块开头补上前一块末尾 overlap 字符
    overlapped: list[str] = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap > 0:
            c = chunks[i - 1][-overlap:] + c
        overlapped.append(c)
    return overlapped


if __name__ == "__main__":
    demo = "大模型正在改变软件开发方式。RAG 让模型能引用外部知识。容器化让部署更简单。云原生提升了系统的弹性。"
    for i, c in enumerate(chunk_text(demo)):
        print(f"[{i}] {c}")
