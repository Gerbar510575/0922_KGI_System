#!/usr/bin/env bash
set -euo pipefail

# 在容器環境下建立/重建 Qdrant 索引（需先 docker compose up -d qdrant）
# 請先把 PDF 放進 apps/rag-service/data/docs/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] 確認 Qdrant 服務狀態..."
docker compose -f "${ROOT_DIR}/infra/docker-compose.yml" up -d qdrant

echo "[bootstrap] 建立 RAG 索引（Gemini embeddings -> Qdrant）..."
docker compose -f "${ROOT_DIR}/infra/docker-compose.yml" run --rm \
  -e GENAI_API_KEY \
  rag python -m indexer_gemini

echo "[bootstrap] 完成。"

