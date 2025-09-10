# build_index.py
import os, json
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
DB_NAME = "funds"
JSON_PATH = "./All_Funds_with_metadata_cleaned_normalized.json"
EMBED_MODEL = "BAAI/bge-m3"

# 初始化
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
db = chroma_client.get_or_create_collection(name=DB_NAME, metadata={"hnsw:space": "cosine"})
model = SentenceTransformer(EMBED_MODEL)

# 讀 JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [d["text"] for d in data]
metadatas = [d["metadata"] for d in data]
ids = [str(d["id"]) for d in data]

# 產生 embedding
embeddings = model.encode(
    [f"passage: {t}" for t in documents],
    normalize_embeddings=True,
    batch_size=64,
    convert_to_numpy=True
).tolist()

# 寫入 Chroma
db.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

print(f"✅ 已存入 {len(documents)} 筆資料到 {CHROMA_DIR}/{DB_NAME}")
