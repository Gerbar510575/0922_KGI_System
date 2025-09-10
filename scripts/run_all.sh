#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${ROOT_DIR}/.env" ]; then
  echo "請先在專案根目錄建立 .env（可複製 configs/app.env.example 並填入 GENAI_API_KEY）"
  exit 1
fi

echo "[run] 啟動核心基礎服務..."
docker compose -f "${ROOT_DIR}/infra/docker-compose.yml" up -d redis ml-bridge

echo "[run] 建立/更新向量索引（Chroma + BAAI/bge-m3）..."
"${ROOT_DIR}/scripts/bootstrap_demo.sh"

echo "[run] 啟動應用服務..."
docker compose -f "${ROOT_DIR}/infra/docker-compose.yml" up -d rag gateway advisor report market ui

echo "[run] 全部就緒："
echo "  Gateway: http://localhost:8000"
echo "  UI     : http://localhost:8501"

