"""后端接口测试（pytest）。

运行：在仓库根目录执行
    pip install -r backend/requirements.txt
    pytest backend/tests
"""
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "service" in body


def test_list_documents():
    r = client.get("/api/docs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 1  # 至少加载了示例文档


def test_ask():
    r = client.post("/api/ask", json={"question": "什么是 RAG？"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and body["answer"]
    assert "sources" in body


def test_ask_empty():
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400


def test_upload_and_ask():
    r = client.post(
        "/api/upload",
        json={"source": "test.txt", "text": "智答是一个面向实习面试的知识库问答项目。"},
    )
    assert r.status_code == 200
    r2 = client.post("/api/ask", json={"question": "智答是什么？"})
    assert r2.status_code == 200
    assert "智答" in r2.json()["answer"]
