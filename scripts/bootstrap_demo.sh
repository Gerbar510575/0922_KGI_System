#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[bootstrap] 建立/更新 Chroma 向量索引..."
docker compose -f "${ROOT_DIR}/infra/docker-compose.yml" run --rm \
  -e GENAI_API_KEY -e EMBEDDING_MODEL -e CHROMA_DIR -e COLLECTION_NAME \
  rag python build_index.py

echo "[bootstrap] 完成。"



