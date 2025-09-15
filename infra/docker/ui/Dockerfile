FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. 安裝中文字型 (Noto Sans CJK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY apps/ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/ui /app

EXPOSE 8501
CMD ["streamlit","run","streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]


