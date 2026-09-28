"""FastAPI 应用入口。

启动方式（在仓库根目录执行）：
    uvicorn backend.app.main:app --reload --port 8000
交互式文档： http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.api.routes import router as api_router
from backend.app.config import API_VERSION, SERVICE_NAME

app = FastAPI(
    title="智答 AI 知识库问答平台 API",
    description="基于 RAG 的文档智能问答后端服务",
    version=API_VERSION,
)

app.include_router(api_router, prefix="/api")

# ---- 简易监控指标（供 Prometheus 抓取）----
_REQUEST_COUNT = {"total": 0}


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        _REQUEST_COUNT["total"] += 1
        return await call_next(request)


app.add_middleware(MetricsMiddleware)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": SERVICE_NAME, "version": API_VERSION}


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    body = (
        "# HELP zhida_requests_total Total HTTP requests\n"
        "# TYPE zhida_requests_total counter\n"
        f"zhida_requests_total {_REQUEST_COUNT['total']}\n"
        "# HELP zhida_up Service up\n"
        "# TYPE zhida_up gauge\n"
        "zhida_up 1\n"
    )
    return Response(content=body, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
