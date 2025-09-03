# apps/rag-service/service.py
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

import google.genai as genai
from google.genai import types as gtypes

# 策略實作
from multi_query_rag import run_multi_query_rag
from rag_fusion import run_rag_fusion
from decompose_rag import run_decompose_rag
from step_back_rag import run_step_back_rag
from hyde_rag import run_hyde_rag

# baseline / 工具
from retrieval_utils import (
    build_qdrant_retriever,
    join_docs,
    collect_context_items,
)

# ---------------- Init ----------------
load_dotenv()
os.environ.setdefault("USER_AGENT", "kfh-rag/1.0")

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("請在 .env 設定 GENAI_API_KEY")

_gclient = genai.Client(api_key=GENAI_API_KEY)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "kfh_docs_gemini")

app = FastAPI(title="RAG Service (Gemini)")

@app.middleware("http")
async def ensure_utf8(request, call_next):
    response = await call_next(request)
    if isinstance(response, JSONResponse):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

# ---------------- Embedding ----------------
def embed_query(text: str):
    r = _gclient.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=gtypes.EmbedContentConfig(task_type="retrieval_query"),
    )
    return r.embeddings[0].values

@app.get("/health")
def health():
    return {"ok": True, "service": "rag"}

# ---------------- Baseline ----------------
def _baseline_answer(question: str, backend: str = "qdrant", topk: int = 3) -> dict:
    """最小可用：直接向量檢索 + Gemini 生成回答。"""
    try:
        retr = build_qdrant_retriever(top_k=topk)
        docs = retr.invoke(question)
    except Exception as e:
        docs = []
        context_txt = ""
    else:
        context_txt = join_docs(docs, top_k=min(topk, 6))

    prompt = (
        "Use the given context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_txt}\n\nQuestion: {question}"
    )
    try:
        resp = _gclient.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        answer = (resp.text or "").strip()
    except Exception as e:
        answer = f"(baseline generation failed: {e})"

    return {
        "route_used": "baseline_search",
        "backend": backend,
        "answer": answer,
        "contexts": collect_context_items(docs, top_k=min(topk, 6)) if docs else [],
        "meta": {"note": "baseline fallback"},
    }

# ---------------- Strategies ----------------
def _exec_strategy(strategy: str, payload: dict) -> dict:
    """封裝不同策略的呼叫，出錯會 raise，交給上層 fallback"""
    q = payload.get("query", "")
    backend  = payload.get("backend", "qdrant")
    topk     = int(payload.get("topk", 3))
    ctx_topn = int(payload.get("ctx_topn", 6))

    if strategy == "multi_query":
        return run_multi_query_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
    elif strategy == "rag_fusion":
        return run_rag_fusion(q, backend=backend, top_k=topk, top_n_context=ctx_topn)
    elif strategy == "step_back":
        return run_step_back_rag(q, backend=backend, topk_normal=topk, topk_step=topk)
    elif strategy == "hyde":
        return run_hyde_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
    elif strategy == "decompose":
        return run_decompose_rag(
            q, backend=backend, top_k=topk,
            strategy="accumulate", topn_context_per_subq=min(ctx_topn, 4)
        )
    else:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {strategy}")

def _auto_select_strategy(q: str) -> str:
    ql = q.strip()
    low = ql.lower()
    if len(ql) > 48 or any(k in ql for k in ["、", "；", "以及", "與", "與否"]):
        return "rag_fusion"
    if any(k in low for k in ["why", "風險", "來源", "risk", "原因"]):
        return "step_back"
    if any(k in low for k in ["what is", "是什麼", "定義", "explain", "解釋"]):
        return "hyde"
    return "multi_query"

# ---------------- Routes ----------------
@app.post("/auto")
def rag_auto(payload: dict = Body(...)):
    q = payload.get("query", "")
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    chosen = payload.get("strategy") or _auto_select_strategy(q)

    # 先試選定策略
    try:
        result = _exec_strategy(chosen, payload)
        result.setdefault("route_used", chosen)
        result.setdefault("backend", payload.get("backend", "qdrant"))
        return result
    except Exception as e:
        pass

    # 預設 fallback 順序
    for name in ["multi_query", "rag_fusion", "hyde", "step_back", "decompose"]:
        try:
            result = _exec_strategy(name, payload)
            result.setdefault("route_used", name)
            result.setdefault("backend", payload.get("backend", "qdrant"))
            return result
        except Exception:
            continue

    # 全部失敗 → baseline
    return _baseline_answer(
        question=q,
        backend=payload.get("backend", "qdrant"),
        topk=int(payload.get("topk", 3))
    )

@app.post("/search")
def search(payload: dict):
    query = payload.get("query", "")
    topk  = int(payload.get("topk", 3))
    vec = embed_query(query)
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    res = qc.search(collection_name=COLLECTION, query_vector=list(vec), limit=topk)
    return {
        "query": query,
        "hits": [
            {
                "score": float(h.score),
                "source": (h.payload.get("source", "") if h.payload else ""),
                "chunk": (h.payload.get("chunk", "")[:600] if h.payload else "")
            }
            for h in res
        ],
    }
