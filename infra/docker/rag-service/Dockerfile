FROM python:3.11-slim

# ---------------- 基本環境變數 ----------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# ---------------- 安裝必要套件 ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates tini locales git \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /app

# ---------------- 安裝 Python 依賴 ----------------
COPY apps/rag-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---------------- 複製程式碼 ----------------
COPY apps/rag-service /app

# ---------------- 啟動服務 ----------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]




