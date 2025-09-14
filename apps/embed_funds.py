import os
import json
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry

# ---------------- Paths / Config ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DB_NAME = "funds"

GENAI_API_KEY = os.getenv("GENAI_API_KEY", "your_api_key_here")
EMBED_MODEL = "models/gemini-embedding-001"

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
def _dedup_currency(cur):
    if not cur:
        return []
    if isinstance(cur, str):
        parts = [p.strip() for p in cur.replace("、", ",").split(",") if p.strip()]
    else:
        parts = []
        for c in cur:
            parts += [p.strip() for p in str(c).replace("、", ",").split(",") if p.strip()]
    uniq = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return uniq

def _safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    if isinstance(v, str) and v.strip().lower() in {"", "null", "none"}:
        return default
    return v

def enrich_metadata(doc: dict) -> dict:
    """根據 doc_type 自動補齊與規範化 metadata"""
    base = {
        "fund_code": _safe_get(doc, "fund_code"),
        "fund_name": _safe_get(doc, "fund_name"),
        "doc_type": _safe_get(doc, "doc_type"),
        "asof_date": _safe_get(doc, "asof_date"),
        "currency": _dedup_currency(_safe_get(doc, "currency", [])),
        "risk_level": _safe_get(doc, "risk_level"),
    }

    if base["doc_type"] == "月報":
        monthly = {
            "fund_manager": _safe_get(doc, "fund_manager"),
            "fund_size": _safe_get(doc, "fund_size"),
            "custodian": _safe_get(doc, "custodian"),
            "mgmt_fee": _safe_get(doc, "mgmt_fee"),
            "top_holdings": _safe_get(doc, "top_holdings", []),
            "industries": _safe_get(doc, "industries", []),
            "regions": _safe_get(doc, "regions", []),
            "performance": _safe_get(doc, "performance", {}),
            "strategy": _safe_get(doc, "strategy"),
        }
        base.update(monthly)

    if base["doc_type"] == "公開說明書":
        prospectus = {
            "establish_date": _safe_get(doc, "establish_date"),
            "management_company": _safe_get(doc, "management_company"),
            "custodian": _safe_get(doc, "custodian"),
            "fund_type": _safe_get(doc, "fund_type"),
            "duration": _safe_get(doc, "duration"),
            "distribution": _safe_get(doc, "distribution"),
            "investment_scope": _safe_get(doc, "investment_scope"),
            "features": _safe_get(doc, "features"),
            "fees": _safe_get(doc, "fees", {}),
            "nav_announcement": _safe_get(doc, "nav_announcement"),
            "suitability": _safe_get(doc, "suitability"),
        }
        base.update(prospectus)

    return base

def clean_metadata(meta: dict) -> dict:
    """確保 metadatas 的值都是 str/int/float/bool"""
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list):
            cleaned[k] = ", ".join(str(x) for x in v if x is not None)
        elif isinstance(v, dict):
            cleaned[k] = json.dumps({kk: vv for kk, vv in v.items() if vv is not None}, ensure_ascii=False)
        else:
            cleaned[k] = str(v)
    return cleaned

def load_json_docs():
    candidates = [DATA_DIR / "metadata_all.json", DATA_DIR / "funds.json"]
    chosen = None
    for p in candidates:
        if p.exists():
            chosen = p
            break
    if not chosen:
        raise FileNotFoundError("No JSON data found in data/ (metadata_all.json or funds.json)")
    with open(chosen, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs, chosen.name

# ---------------- Main ----------------
def main():
    docs, filename = load_json_docs()
    if not isinstance(docs, list) or not docs:
        raise RuntimeError("JSON 資料不是 list 或為空")

    documents, metadatas, ids = [], [], []
    for i, d in enumerate(docs):
        content = _safe_get(d, "content", "")
        documents.append(content)
        raw_meta = enrich_metadata(d)
        meta = clean_metadata(raw_meta)   # <<< 這裡保證沒有 None
        metadatas.append(meta)
        ids.append(f"{meta.get('fund_code','NA')}_{meta.get('doc_type','NA')}_{i}")

    print(f"🗄  Init Chroma at {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GeminiEmbeddingFunction(document_mode=True)

    try:
        chroma_client.delete_collection(DB_NAME)
        print(f"🗑  Old collection '{DB_NAME}' deleted")
    except Exception:
        pass

    db = chroma_client.create_collection(name=DB_NAME, embedding_function=embed_fn)

    print(f"➕ Adding {len(documents)} docs (from {filename}) into collection '{DB_NAME}'")
    db.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ Embedding completed!")

if __name__ == "__main__":
    main()



