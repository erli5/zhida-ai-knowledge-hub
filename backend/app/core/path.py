"""路径与模块加载。

把项目根目录（rag / backend 的上级）加入 sys.path，
使得后端可以 `from rag...` 直接引用 RAG 核心模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

# backend/app/core/path.py -> parents[3] == 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_DOCS_DIR = PROJECT_ROOT / "rag" / "data" / "sample_docs"
