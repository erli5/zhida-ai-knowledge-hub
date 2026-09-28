"""API 路由：知识库与问答接口。"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    AskRequest,
    AskResponse,
    DocInfo,
    UploadResponse,
)
from backend.app.services.rag_service import rag_service

router = APIRouter()


@router.get("/docs", response_model=List[DocInfo], tags=["knowledge"])
def list_documents() -> List[dict]:
    """列出知识库中的文档。"""
    return rag_service.list_documents()


@router.post("/upload", response_model=UploadResponse, tags=["knowledge"])
def upload_document(payload: dict) -> dict:
    """向知识库新增一篇文档（JSON: {"source": str, "text": str}）。

    生产环境可扩展为文件上传（multipart）并做格式校验/异步处理。
    """
    source = payload.get("source")
    text = payload.get("text")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    doc_id = rag_service.add_document(source or "untitled", text)
    return {"doc_id": doc_id, "message": "已加入知识库"}


@router.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(req: AskRequest) -> dict:
    """问答接口：返回答案与引用来源。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")
    return rag_service.ask(req.question)
