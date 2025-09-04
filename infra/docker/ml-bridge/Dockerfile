FROM python:3.11-slim

# 安裝 R 與常用套件（Debian/Ubuntu 系列）
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base curl ca-certificates default-jre \
    && rm -rf /var/lib/apt/lists/*

# 安裝 R 套件（視你的模型而定）
RUN R -q -e "install.packages(c('jsonlite'), repos='https://cloud.r-project.org')"

WORKDIR /app
COPY apps/ml-bridge /app

# Python 依賴（FastAPI + Uvicorn + PyYAML）
RUN pip install --no-cache-dir fastapi uvicorn pyyaml

ENV MLBRIDGE_CONFIG=/app/config/models.yaml \
    SUBPROC_TIMEOUT=15

EXPOSE 7000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "7000"]
