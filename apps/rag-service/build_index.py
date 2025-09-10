import os, json
import chromadb

CHROMA_DIR = "./chroma_db"
DB_NAME = "funds"
JSON_PATH = "./All_Funds_with_embeddings.json"

# 初始化 Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
db = chroma_client.get_or_create_collection(name=DB_NAME, metadata={"hnsw:space": "cosine"})

# 讀 JSON（已經有 embedding）
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [d["text"] for d in data]
metadatas = [d["metadata"] for d in data]
ids = [str(d["id"]) for d in data]
embeddings = [d["embedding"] for d in data]

# 寫入 Chroma
db.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

print(f"✅ 已存入 {len(documents)} 筆資料到 {CHROMA_DIR}/{DB_NAME}")
