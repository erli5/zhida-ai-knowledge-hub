"""智答 · 对话式知识库问答 Web 应用（Streamlit）。

定位：AI 应用开发方向的演示前端。通过调用后端 API 完成问答；
若后端未启动，则自动回退到本地 RAG 流水线，保证界面本身可独立运行演示。

运行：
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import requests
import streamlit as st

# 把项目根目录加入路径，使本地回退模式可 `from rag...`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_API = os.getenv("ZHIDA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="智答 · 知识库问答", page_icon="🤖", layout="centered")

st.title("🤖 智答 · 知识库问答")
st.caption("基于 RAG 的文档智能问答 Demo（AI 应用开发方向）")

api_url = st.sidebar.text_input("后端 API 地址", value=DEFAULT_API)
st.sidebar.markdown("若后端未启动，应用会自动使用本地 RAG 流水线。")

# 会话状态：问答历史
if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def get_local_pipeline():
    """本地回退用的 RAG 流水线（仅后端不可用时使用）。"""
    from rag.pipeline import RAGPipeline

    pipe = RAGPipeline(top_k=3)
    docs = []
    sample_dir = os.path.join(PROJECT_ROOT, "rag", "data", "sample_docs")
    if os.path.isdir(sample_dir):
        for f in sorted(os.listdir(sample_dir)):
            if f.endswith((".txt", ".md")):
                with open(os.path.join(sample_dir, f), "r", encoding="utf-8") as fh:
                    docs.append({"doc_id": f, "text": fh.read(), "source": f})
    if docs:
        pipe.index(docs)
    return pipe


def ask_backend(question: str) -> dict | None:
    try:
        resp = requests.post(
            f"{api_url}/api/ask", json={"question": question}, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def ask_local(question: str) -> dict:
    pipe = get_local_pipeline()
    return pipe.query(question)


def answer_question(question: str) -> dict:
    data = ask_backend(question)
    if data is None:
        data = ask_local(question)
        data["source_mode"] = "本地 RAG 回退"
    else:
        data["source_mode"] = "后端 API"
    return data


with st.form("qa_form", clear_on_submit=True):
    question = st.text_input("输入你的问题", placeholder="例如：什么是 RAG？")
    submitted = st.form_submit_button("提问")

if submitted and question.strip():
    with st.spinner("正在检索并生成答案..."):
        result = answer_question(question.strip())
    st.session_state.history.append((question.strip(), result))

# 展示最新回答
if st.session_state.history:
    q, r = st.session_state.history[-1]
    st.subheader("💡 回答")
    st.write(r.get("answer", ""))
    mode = r.get("source_mode", "")
    st.caption(f"回答来源模式：{mode}")
    if r.get("sources"):
        st.markdown("**📚 引用来源：** " + "、".join(r["sources"]))
    with st.expander("查看检索到的相关片段"):
        for i, ctx in enumerate(r.get("contexts", []), 1):
            st.markdown(f"**片段 {i}**：{ctx}")

    st.divider()
    st.subheader("🕘 历史对话")
    for q_old, r_old in reversed(st.session_state.history[:-1]):
        st.markdown(f"**问：** {q_old}")
        st.markdown(f"**答：** {r_old.get('answer', '')}")
