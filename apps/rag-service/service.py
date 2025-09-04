# apps/rag-service/service.py
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import os, re, time
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
    except Exception:
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

# ---------------- Heuristics ----------------
DEF_PATTERNS = [
    r"什麼是", r"是什麼", r"定義", r"meaning", r"what\s+is\b", r"define\b", r"概念"
]
MULTI_PATTERNS = [
    r"比較", r"差異", r"\bvs\.?\b", r"優缺點", r"pros", r"cons",
    r"步驟", r"流程", r"策略", r"框架", r"如何", r"how\s+to\b"
]

def is_definition_like(q: str) -> bool:
    q = q.lower().strip()
    if len(q) <= 12:
        return True
    return any(re.search(p, q) for p in DEF_PATTERNS)

def is_multifaceted(q: str) -> bool:
    q = q.lower().strip()
    return any(re.search(p, q) for p in MULTI_PATTERNS)

# ---------------- Routes ----------------
@app.post("/auto")
def rag_auto(payload: dict = Body(...)):
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    backend = payload.get("backend", "qdrant")
    topk = int(payload.get("topk", 3))
    ctx_topn = int(payload.get("ctx_topn", 6))
    strategy = payload.get("strategy", "parallel")

    t0 = time.time()

    # Step1: 初步 search
    try:
        retr = build_qdrant_retriever(top_k=max(3, topk))
        docs = retr.invoke(q)
        contexts = collect_context_items(docs, top_k=max(3, ctx_topn))
        top_score = max([c.get("score", 0.0) for c in contexts], default=0.0)
    except Exception as e:
        raise HTTPException(502, f"RAG /search failed: {e}")

    decision_reason = f"top_score={top_score:.2f}, hits={len(contexts)}"

    # Step2: Heuristics 選擇策略
    if top_score >= 0.55 and len(contexts) >= 2:
        chosen = "search"
        decision_reason += " → route=search"
    elif 0.35 <= top_score < 0.55:
        chosen = "multi_query"
        decision_reason += " → route=multi_query"
    else:
        if is_definition_like(q):
            chosen = "hyde"
            decision_reason += " → route=hyde (definition-like)"
        elif is_multifaceted(q):
            chosen = "decompose_parallel" if strategy == "parallel" else "decompose_accumulate"
            decision_reason += f" → route={chosen} (multifaceted)"
        else:
            chosen = "step_back"
            decision_reason += " → route=step_back (short/ambiguous)"

    # Step3: 執行策略
    if chosen == "search":
        result = {
            "route_used": "search",
            "backend": backend,
            "answer": None,
            "contexts": contexts[:ctx_topn],
            "meta": {"top_score": top_score, "decision": decision_reason,
                     "latency_ms": int((time.time()-t0)*1000)},
        }
        return result

    elif chosen == "multi_query":
        result = run_multi_query_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
    elif chosen.startswith("decompose"):
        result = run_decompose_rag(q, backend=backend, top_k=topk,
                                   strategy="parallel" if chosen.endswith("parallel") else "accumulate",
                                   topn_context_per_subq=min(ctx_topn, 4))
    elif chosen == "step_back":
        result = run_step_back_rag(q, backend=backend, topk_normal=topk, topk_step=topk)
    elif chosen == "hyde":
        result = run_hyde_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
    else:
        return _baseline_answer(q, backend=backend, topk=topk)

    # Step4: 補上 meta
    result.setdefault("route_used", chosen)
    result.setdefault("backend", backend)
    result.setdefault("meta", {})["decision"] = decision_reason
    result["meta"]["latency_ms"] = int((time.time()-t0)*1000)

    return result

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

