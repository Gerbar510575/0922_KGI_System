FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基本系統工具（字型/SSL/編譯少量套件需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY apps/rag-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/rag-service /app
# 讓容器可讀到專案根 .env（compose 會掛載）
ENV PYTHONPATH=/app

EXPOSE 8002
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","8002","--workers","1","--log-level","info"]
