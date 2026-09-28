"""后端配置。"""
from __future__ import annotations

from backend.app.core.path import PROJECT_ROOT, SAMPLE_DOCS_DIR

# 服务基础配置
SERVICE_NAME = "zhida-backend"
API_VERSION = "1.0.0"

# 知识库
SAMPLE_DOCS_DIR = SAMPLE_DOCS_DIR
PROJECT_ROOT = PROJECT_ROOT

# RAG 检索返回的片段数
RAG_TOP_K = 3
