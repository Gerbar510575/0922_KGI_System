# apps/embed_funds.py
import os
import json
import uuid
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry

# ---------------- Paths / Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DB_NAME = os.getenv("DB_NAME", "funds")

CHUNKED_FILE = DATA_DIR / "metadata_all_chunked_para.json"

GENAI_API_KEY = os.getenv("GENAI_API_KEY", "your_api_key_here")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# ---------------- Google Gemini Client ----------------
genai_client = genai.Client(api_key=GENAI_API_KEY)

# ---------------- Embedding Function ----------------
is_retriable = lambda e: (
    isinstance(e, genai.errors.APIError) and getattr(e, "code", None) in {429, 503}
)

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, inputs: list[str]) -> list[list[float]]:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        resp = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=inputs,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in resp.embeddings]

# ---------------- Helpers ----------------
def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """將多層 dict 展平成 dot-notation，保留 *_json 原始字串，方便查詢"""
    def to_compact_string(x) -> str:
        if isinstance(x, dict):
            name = x.get("name") or x.get("ticker") or ""
            weight = x.get("weight") or x.get("pct") or ""
            if name or weight:
                return f"{name} {weight}".strip()
            return json.dumps(x, ensure_ascii=False)
        return str(x)

    flat = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat[f"{key}_json"] = json.dumps(v, ensure_ascii=False)
            flat.update(flatten_dict(v, key, sep=sep))
        elif isinstance(v, list):
            flat[f"{key}_json"] = json.dumps(v, ensure_ascii=False)
            flat[key] = ", ".join([to_compact_string(x) for x in v[:20]])
        else:
            if v is None or v == "":
                continue
            flat[key] = v
    return flat

def clean_metadata(meta: dict) -> dict:
    """確保 metadatas 的值都是 str/int/float/bool"""
    cleaned = {}
    for k, v in meta.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

def load_chunked_records():
    if not CHUNKED_FILE.exists():
        raise FileNotFoundError("❌ 找不到 metadata_all_chunked_para.json，請確認 data/ 目錄下有此檔案")

    with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise RuntimeError("❌ metadata_all_chunked_para.json 格式錯誤或為空")

    return records, CHUNKED_FILE.name

# ---------------- Main ----------------
def main():
    records, filename = load_chunked_records()

    documents, metadatas, ids = [], [], []
    for i, rec in enumerate(records):
        rec_id = rec.get("id") or ""
        chunk_id = rec.get("chunk_id")
        content = rec.get("content") or ""
        metadata = rec.get("metadata") or {}

        # 僅 embedding content
        documents.append(content)

        # 展平 metadata
        meta_flat = flatten_dict(metadata)
        meta_clean = clean_metadata(meta_flat)

        # 保留核心欄位
        for key in ["fund_code", "fund_name", "doc_type", "asof_date"]:
            if key not in meta_clean and metadata.get(key):
                meta_clean[key] = metadata[key]

        metadatas.append(meta_clean)

        # --- 修正 ID，保證全域唯一 ---
        if rec_id:
            base_id = rec_id
        elif chunk_id is not None:
            base_id = f"{metadata.get('fund_code','NA')}_{metadata.get('doc_type','NA')}_{chunk_id}"
        else:
            base_id = f"{metadata.get('fund_code','NA')}_{metadata.get('doc_type','NA')}_{i}"

        # 加上 uuid 短碼，避免重複
        unique_id = f"{base_id}_{uuid.uuid4().hex[:8]}"
        ids.append(unique_id)

    # 去重檢查
    if len(ids) != len(set(ids)):
        raise RuntimeError("❌ Duplicate IDs detected even after UUID patch!")

    print(f"🗄  Init Chroma at {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GeminiEmbeddingFunction(document_mode=True)

    try:
        chroma_client.delete_collection(DB_NAME)
        print(f"🗑  Old collection '{DB_NAME}' deleted")
    except Exception:
        pass

    db = chroma_client.create_collection(name=DB_NAME, embedding_function=embed_fn)

    print(f"➕ Adding {len(documents)} chunks (from {filename}) into collection '{DB_NAME}'")
    db.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ Embedding completed!")

if __name__ == "__main__":
    main()






