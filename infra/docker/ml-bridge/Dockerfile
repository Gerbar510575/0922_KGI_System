FROM python:3.11-slim

# 安裝 R 與必要編譯依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev build-essential \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    curl ca-certificates default-jre \
    && rm -rf /var/lib/apt/lists/*

# 安裝 R 套件：remotes + jsonlite + glmnet
RUN R -e "install.packages(c('remotes','jsonlite','glmnet'), repos='https://cloud.r-project.org')"

# 安裝 workflows / parsnip / hardhat（版本與本地一致）
RUN R -e "remotes::install_version('workflows', version='1.3.0', repos='https://cloud.r-project.org')" && \
    R -e "remotes::install_version('parsnip', version='1.3.2', repos='https://cloud.r-project.org')" && \
    R -e "remotes::install_version('hardhat', version='1.4.2', repos='https://cloud.r-project.org')"

WORKDIR /app
COPY apps/ml-bridge /app

RUN pip install --no-cache-dir fastapi uvicorn pyyaml

ENV MLBRIDGE_CONFIG=/app/config/models.yaml \
    SUBPROC_TIMEOUT=15

EXPOSE 7000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "7000"]


