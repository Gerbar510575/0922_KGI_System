# config.py
import os

# 資料 & DB
DATA_JSON = "data/processed/All_Funds_with_metadata_cleaned_normalized"
CHROMA_DIR = "db/chroma_db"

# 模型
EMBEDDING_MODEL = "BAAI/bge-m3"
GEMINI_MODEL = "gemini-2.5-flash"

# API key
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")

# RAG 設定
TOP_K = 5
SHOW_CHAR = 200


