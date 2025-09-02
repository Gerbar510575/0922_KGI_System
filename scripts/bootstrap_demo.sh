#!/usr/bin/env bash
set -e

# Build vector index from PDFs and start all services
python apps/rag-service/indexer.py

docker compose -f infra/docker-compose.yml up -d
