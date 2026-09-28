FROM python:3.11-slim

WORKDIR /app
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY ai_app/requirements.txt ./ai_app/requirements.txt
RUN pip install --no-cache-dir -r ai_app/requirements.txt

COPY ai_app ./ai_app
COPY rag ./rag

EXPOSE 8501
CMD ["streamlit", "run", "ai_app/app.py"]
