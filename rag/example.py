"""RAG 核心可运行 Demo（零依赖）。

运行方式（任选其一）：
    cd rag && python example.py
    python -m rag.example        # 在仓库根目录执行

会从 data/sample_docs 读取示例文档，建立索引，并回答几个预设问题。
"""
from __future__ import annotations

import os
import sys

# 把仓库根目录加入路径，支持 `python example.py` 直接运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rag.pipeline import RAGPipeline  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_docs")

QUESTIONS = [
    "什么是 RAG？",
    "容器化部署有什么好处？",
    "大模型微调常用什么方法？",
]


def load_sample_documents() -> list[dict]:
    docs = []
    if not os.path.isdir(SAMPLE_DIR):
        print(f"[警告] 示例文档目录不存在: {SAMPLE_DIR}")
        return docs
    for fname in sorted(os.listdir(SAMPLE_DIR)):
        if fname.endswith((".txt", ".md")):
            path = os.path.join(SAMPLE_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"doc_id": fname, "text": text, "source": fname})
    return docs


def main() -> None:
    print("=" * 60)
    print("智答 RAG Demo（纯 Python，零依赖）")
    print("=" * 60)

    documents = load_sample_documents()
    if not documents:
        print("未加载到任何示例文档，退出。")
        return

    print(f"已加载 {len(documents)} 篇文档，开始建立索引...")
    pipeline = RAGPipeline(top_k=3)
    pipeline.index(documents)
    print("索引建立完成。\n")

    for q in QUESTIONS:
        print(f"❓ 问题：{q}")
        result = pipeline.query(q)
        print(f"💡 回答：{result['answer']}")
        if result["sources"]:
            print(f"📚 来源：{', '.join(result['sources'])}")
        print("-" * 60)


if __name__ == "__main__":
    main()
