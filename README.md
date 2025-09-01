# 0) 放入 PDF 到 apps/rag-service/data/docs/；準備 products.csv
# 1) 建索引

python apps/rag-service/indexer.py
# 2) 一鍵啟動
docker compose -f infra/docker-compose.yml up -d
# 3) 開前端
# http://localhost:8501
