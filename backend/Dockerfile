# 后端镜像（多阶段可省略，这里用 slim 直装以减小复杂度）
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# 先装依赖，利用层缓存
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# 拷贝后端与 RAG 核心模块（共享项目根）
COPY backend ./backend
COPY rag ./rag

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
