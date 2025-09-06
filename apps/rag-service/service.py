# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import os, re, time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

import google.genai as genai
from google.genai import types as gtypes

# 策略模組
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

# ---------------- Heuristics ----------------
DEF_PATTERNS = [r"什麼是", r"是什麼", r"定義", r"meaning", r"what\s+is\b", r"define\b", r"概念"]
MULTI_PATTERNS = [r"比較", r"差異", r"\bvs\.?\b", r"優缺點", r"pros", r"cons",
                  r"步驟", r"流程", r"策略", r"框架", r"如何", r"how\s+to\b"]

FEES_KEYS = ["費用", "經理費", "保管費", "手續費", "買回費", "申購", "配息", "RR", "風險等級"]
HOLDINGS_KEYS = ["前十大", "持股", "產業", "國家", "績效", "報酬", "淨值", "資料日期", "月報", "月"]

def is_definition_like(q: str) -> bool:
    q = q.lower().strip()
    if len(q) <= 12:
        return True
    return any(re.search(p, q) for p in DEF_PATTERNS)

def is_multifaceted(q: str) -> bool:
    q = q.lower().strip()
    return any(re.search(p, q) for p in MULTI_PATTERNS)

def decide_doc_type_pref(q: str) -> str | None:
    if any(k in q for k in FEES_KEYS):
        return "prospectus_short"
    if any(k in q for k in HOLDINGS_KEYS):
        return "monthly_report"
    return None

def qdrant_search_with_filter(query_vec, topk: int, doc_type: str | None):
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    flt = None
    if doc_type:
        flt = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))])
    return qc.search(collection_name=COLLECTION, query_vector=list(query_vec), limit=topk, query_filter=flt)

# ---------------- Answer synthesize ----------------
def synthesize_answer(context_txt: str, q: str) -> str:
    prompt = (
        "請只根據提供的脈絡回答用戶問題，並在末行以【來源】列出文件名與頁碼；"
        "若脈絡不足請明確說不知道，不要臆測。\n\n"
        f"脈絡：\n{context_txt}\n\n問題：{q}\n\n回答："
    )
    try:
        resp = _gclient.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"(generation failed: {e})"

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "service": "rag"}

@app.post("/search")
def search(payload: dict):
    query = payload.get("query", "")
    topk  = int(payload.get("topk", 3))
    doc_type = payload.get("doc_type")  # 可選: monthly_report / prospectus_short
    vec = embed_query(query)

    if doc_type:
        res = qdrant_search_with_filter(vec, topk, doc_type)
    else:
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        res = qc.search(collection_name=COLLECTION, query_vector=list(vec), limit=topk)

    return {
        "query": query,
        "hits": [
            {
                "score": float(h.score),
                "source": h.payload.get("source", ""),
                "chunk": h.payload.get("chunk", "")[:800],
                "page": h.payload.get("page"),
                "doc_type": h.payload.get("doc_type"),
                "asof_date": h.payload.get("asof_date"),
            }
            for h in res
        ],
    }

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
    preferred_doc = decide_doc_type_pref(q)
    qvec = embed_query(q)

    try:
        if preferred_doc:
            res = qdrant_search_with_filter(qvec, max(5, topk), preferred_doc)
        else:
            qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            res = qc.search(collection_name=COLLECTION, query_vector=list(qvec), limit=max(5, topk))

        contexts = [{
            "score": float(h.score),
            "source": h.payload.get("source", ""),
            "chunk": h.payload.get("chunk", ""),
            "page": h.payload.get("page"),
            "doc_type": h.payload.get("doc_type"),
            "asof_date": h.payload.get("asof_date"),
        } for h in res]

        contexts.sort(key=lambda c: (c.get("asof_date") or "", c["score"]), reverse=True)
        top_contexts = contexts[:ctx_topn]
        top_score = max([c["score"] for c in top_contexts], default=0.0)
    except Exception as e:
        raise HTTPException(502, f"RAG /search failed: {e}")

    decision_reason = f"pref={preferred_doc}, top_score={top_score:.2f}, hits={len(top_contexts)}"

    if top_score >= 0.55 and len(top_contexts) >= 2:
        chosen = "search+answer"
        ctx_txt = "\n\n".join([f"[{i+1}] {c['chunk']} (來源:{c['source']}, p.{c.get('page')})"
                               for i, c in enumerate(top_contexts)])
        answer = synthesize_answer(ctx_txt, q)
        return {
            "route_used": chosen,
            "backend": backend,
            "answer": answer,
            "contexts": top_contexts,
            "meta": {"decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
        }

    if 0.35 <= top_score < 0.55:
        chosen = "multi_query"
        result = run_multi_query_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
    else:
        if is_definition_like(q):
            chosen = "hyde"
            result = run_hyde_rag(q, backend=backend, topk=topk, ctx_topn=ctx_topn)
        elif is_multifaceted(q):
            chosen = "decompose_parallel" if strategy == "parallel" else "decompose_accumulate"
            result = run_decompose_rag(q, backend=backend, top_k=topk,
                                       strategy="parallel" if chosen.endswith("parallel") else "accumulate",
                                       topn_context_per_subq=min(ctx_topn, 4))
        else:
            chosen = "step_back"
            result = run_step_back_rag(q, backend=backend, topk_normal=topk, topk_step=topk)

    result.setdefault("route_used", chosen)
    result.setdefault("backend", backend)
    result.setdefault("meta", {})["decision"] = decision_reason
    result["meta"]["latency_ms"] = int((time.time()-t0)*1000)
    return result


