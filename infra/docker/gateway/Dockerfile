FROM python:3.11-slim

# 基本環境
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 先安裝依賴（利用 Docker layer cache）
COPY apps/gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 複製應用程式
COPY apps/gateway/ /app/

EXPOSE 8000

# 啟動 Gateway（uvicorn）
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000","--log-level","info"]
