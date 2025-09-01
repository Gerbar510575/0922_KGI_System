# apps/rag-service/service.py
from fastapi import FastAPI
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient

import google.genai as genai
from google.genai import types as gtypes

load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("請在 .env 設定 GENAI_API_KEY")
_gclient = genai.Client(api_key=GENAI_API_KEY)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

app = FastAPI(title="RAG Service (Gemini)")

def embed_query(text: str):
    r = _gclient.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=gtypes.EmbedContentConfig(task_type="retrieval_query"),
    )
    return r.embeddings[0].values

# --- 簡單檢索（Gemini 向量） ---
@app.post("/search")
def search(payload: dict):
    query = payload.get("query", "")
    topk  = int(payload.get("topk", 3))
    vec = embed_query(query)
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    res = qc.search(collection_name=COLLECTION, query_vector=list(vec), limit=topk)
    return {"query": query,
            "hits": [{"score": float(h.score), "source": h.payload.get("source",""), "chunk": h.payload["chunk"][:600]} for h in res]}

# --- Multi-Query（Gemini） ---
from .multi_query_rag import run_multi_query_rag

@app.post("/multi_query_search")
def multi_query_search(payload: dict):
    q = payload.get("query", "")
    backend = payload.get("backend", "qdrant")  # qdrant | chroma
    topk = int(payload.get("topk", 3))
    urls = payload.get("urls", None)
    return run_multi_query_rag(question=q, backend=backend, top_k=topk, urls=urls)

# === 新增：RAG-Fusion (RRF) ===
from .rag_fusion import run_rag_fusion

@app.post("/rag_fusion_search")
def rag_fusion_search(payload: dict):
    """
    payload:
      {
        "query": "你的問題",
        "backend": "qdrant" | "chroma",      # 預設 qdrant
        "topk": 3,                            # 每個子查詢取回的 K
        "urls": ["https://..."],              # backend=chroma 時可選
        "topn_context": 8                     # 最終回答拼接的段落數
      }
    """
    q = payload.get("query", "")
    backend = payload.get("backend", "qdrant")
    topk = int(payload.get("topk", 3))
    urls = payload.get("urls", None)
    topn = int(payload.get("topn_context", 8))
    return run_rag_fusion(question=q, backend=backend, top_k=topk, urls=urls, top_n_context=topn)

# apps/rag-service/service.py （新增段落）
from .decompose_rag import run_decompose_rag

@app.post("/decompose_search")
def decompose_search(payload: dict):
    """
    payload:
      {
        "query": "你的主問題",
        "backend": "qdrant" | "chroma",    # 預設 qdrant
        "topk": 3,                          # 每題檢索 K
        "urls": ["https://..."],            # backend=chroma 時可選
        "strategy": "accumulate" | "parallel",
        "topn_context_per_subq": 4          # 每題取前 N 段拼 context
      }
    """
    q = payload.get("query", "")
    backend = payload.get("backend", "qdrant")
    topk = int(payload.get("topk", 3))
    urls = payload.get("urls", None)
    strategy = payload.get("strategy", "accumulate")
    topn_ctx = int(payload.get("topn_context_per_subq", 4))
    return run_decompose_rag(
        question=q, backend=backend, top_k=topk, urls=urls,
        strategy=strategy, topn_context_per_subq=topn_ctx
    )

# apps/rag-service/service.py （新增）
from .step_back_rag import run_step_back_rag

@app.post("/step_back_search")
def step_back_search(payload: dict):
    """
    payload:
      {
        "query": "你的問題",
        "backend": "qdrant" | "chroma",    # 預設 qdrant
        "topk_normal": 4,
        "topk_step": 4,
        "urls": ["https://..."],           # backend=chroma 時可選
        "model": "gemini-2.5-flash",
        "temperature": 0.0
      }
    """
    q = payload.get("query", "")
    backend = payload.get("backend", "qdrant")
    topk_normal = int(payload.get("topk_normal", 4))
    topk_step   = int(payload.get("topk_step", 4))
    urls = payload.get("urls", None)
    model = payload.get("model", "gemini-2.5-flash")
    temperature = float(payload.get("temperature", 0.0))

    return run_step_back_rag(
        question=q,
        backend=backend,
        topk_normal=topk_normal,
        topk_step=topk_step,
        urls=urls,
        model=model,
        temperature=temperature,
    )
