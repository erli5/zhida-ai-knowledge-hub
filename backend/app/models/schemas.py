"""Pydantic 数据模型（请求/响应校验）。"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户提问")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    contexts: List[str] = []


class DocInfo(BaseModel):
    doc_id: str
    source: str
    length: int


class UploadResponse(BaseModel):
    doc_id: str
    message: str
