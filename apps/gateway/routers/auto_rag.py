# apps/gateway/routers/auto_rag.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException
import os, time, re
import httpx

RAG_URL = os.getenv("RAG_URL", "http://rag:8002")
router = APIRouter(prefix="/rag", tags=["rag-auto"])

# --------- 問題型態判斷（簡易 Heuristics） ---------
DEF_PATTERNS = [
    r"什麼是", r"是什麼", r"定義", r"meaning", r"what\s+is\b", r"define\b", r"概念"
]
MULTI_PATTERNS = [
    r"比較", r"差異", r"\bvs\.?\b", r"優缺點", r"pros", r"cons",
    r"步驟", r"流程", r"策略", r"框架", r"如何", r"how\s+to\b"
]

def is_definition_like(q: str) -> bool:
    q = q.lower().strip()
    if len(q) <= 12:  # 中文很短的查詢多半是定義/概念
        return True
    return any(re.search(p, q) for p in DEF_PATTERNS)

def is_multifaceted(q: str) -> bool:
    q = q.lower().strip()
    return any(re.search(p, q) for p in MULTI_PATTERNS)

# --------- 統一回傳格式 ---------
def unify_response(route_used: str, backend: str, answer, contexts, meta: dict):
    # contexts 統一為 [{source, chunk, score?}]
    norm_ctx = []
    for c in contexts or []:
        norm_ctx.append({
            "source": c.get("source", "unknown"),
            "chunk": c.get("chunk", c.get("page_content", ""))[:800],
            **({"score": c["score"]} if "score" in c else {})
        })
    return {
        "route_used": route_used,
        "backend": backend or "qdrant",
        "answer": answer,
        "contexts": norm_ctx,
        "meta": meta or {}
    }

# --------- 轉換各子系統回應到統一格式的 helper ---------
def from_search(resp: dict):
    backend = resp.get("backend", "qdrant")
    results = resp.get("results") or resp.get("contexts") or []
    # 允許不同 search 回傳形狀：盡力映射
    contexts = []
    for r in results:
        if isinstance(r, dict):
            contexts.append({
                "source": r.get("source") or r.get("metadata", {}).get("source") or "unknown",
                "chunk": r.get("chunk") or r.get("page_content") or "",
                **({"score": r["score"]} if "score" in r else {})
            })
    top_score = max([c.get("score", 0.0) for c in contexts], default=0.0)
    return backend, contexts, top_score

def from_multi_query(resp: dict):
    return resp.get("backend","qdrant"), resp.get("answer"), resp.get("contexts", [])

def from_rff(resp: dict):
    # contexts 內含 score
    return resp.get("backend","qdrant"), resp.get("answer"), resp.get("contexts", [])

def from_decompose(resp: dict):
    backend = resp.get("backend","qdrant")
    answer = resp.get("final_answer")
    # resp["contexts"] 是 list[{"sub_question":..., "contexts":[...]}]
    ctx = []
    for item in resp.get("contexts", []):
        for c in item.get("contexts", []):
            ctx.append(c)
    return backend, answer, ctx

def from_step_back(resp: dict):
    backend = resp.get("backend","qdrant")
    answer = resp.get("answer")
    ctx = (resp.get("normal_contexts", []) or []) + (resp.get("step_back_contexts", []) or [])
    return backend, answer, ctx

def from_hyde(resp: dict):
    return resp.get("backend","qdrant"), resp.get("answer"), resp.get("contexts", [])

# --------- 主路由：/rag/auto ---------
@router.post("/auto")
async def rag_auto(payload: dict):
    """
    payload:
      {
        "query": "使用者問題",
        "backend": "qdrant" | "chroma",      # 預設 qdrant
        "topk": 3,                            # 初步 search 的 topk，以及大多數策略的預設 k
        "urls": ["https://..."],              # backend=chroma 時可選
        "strategy": "parallel|accumulate",    # 針對 decompose 可覆寫
        "ctx_topn": 6                         # 輸出 contexts 最大數，預設 6
      }
    """
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "query 不可為空")

    backend = payload.get("backend", "qdrant")
    topk = int(payload.get("topk", 3))
    urls = payload.get("urls")
    strategy = payload.get("strategy", "parallel")
    ctx_topn = int(payload.get("ctx_topn", 6))

    t0 = time.time()

    # 1) 初步 /search（用來決策）
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            s_resp = await c.post(f"{RAG_URL}/search", json={
                "query": q, "backend": backend, "topk": max(3, topk), "urls": urls
            })
            s_json = s_resp.json()
        except Exception as e:
            raise HTTPException(502, f"呼叫 RAG /search 失敗: {e}")

    s_backend, s_contexts, top_score = from_search(s_json)
    decision_reason = f"top_score={top_score:.2f}, hits={len(s_contexts)}"

    # 2) 規則選擇策略
    route = None
    if top_score >= 0.55 and len(s_contexts) >= 2:
        route = "search"
        decision_reason += " → route=search"
    elif 0.35 <= top_score < 0.55:
        route = "multi_query"
        decision_reason += " → route=multi_query"
    else:
        if is_definition_like(q):
            route = "hyde"
            decision_reason += " → route=hyde (definition-like)"
        elif is_multifaceted(q):
            route = "decompose_parallel" if strategy == "parallel" else "decompose_accumulate"
            decision_reason += f" → route={route} (multifaceted)"
        else:
            route = "step_back"
            decision_reason += " → route=step_back (short/ambiguous)"

    # 3) 依決策呼叫對應子系統，並轉成統一格式
    async with httpx.AsyncClient(timeout=60) as c:
        if route == "search":
            r = await c.post(f"{RAG_URL}/search", json={
                "query": q, "backend": backend, "topk": ctx_topn, "urls": urls
            })
            j = r.json()
            backend_used, contexts, _ = from_search(j)
            ans = None  # search 不生成答案
            meta = {"top_score": top_score, "decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
            return unify_response("search", backend_used, ans, contexts[:ctx_topn], meta)

        elif route == "multi_query":
            r = await c.post(f"{RAG_URL}/multi_query_search", json={
                "query": q, "backend": backend, "topk": topk, "urls": urls
            })
            j = r.json()
            backend_used, ans, ctx = from_multi_query(j)
            meta = {"top_score": top_score, "decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
            return unify_response("multi_query", backend_used, ans, ctx[:ctx_topn], meta)

        elif route == "decompose_parallel" or route == "decompose_accumulate":
            r = await c.post(f"{RAG_URL}/decompose_search", json={
                "query": q, "backend": backend, "topk": topk, "urls": urls,
                "strategy": "parallel" if route.endswith("parallel") else "accumulate",
                "topn_context_per_subq": max(3, min(4, ctx_topn))  # 每題3-4段即可
            })
            j = r.json()
            backend_used, ans, ctx = from_decompose(j)
            meta = {"top_score": top_score, "decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
            return unify_response(route, backend_used, ans, ctx[:ctx_topn], meta)

        elif route == "step_back":
            r = await c.post(f"{RAG_URL}/step_back_search", json={
                "query": q, "backend": backend, "topk_normal": topk, "topk_step": topk, "urls": urls
            })
            j = r.json()
            backend_used, ans, ctx = from_step_back(j)
            meta = {"top_score": top_score, "decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
            return unify_response("step_back", backend_used, ans, ctx[:ctx_topn], meta)

        elif route == "hyde":
            r = await c.post(f"{RAG_URL}/hyde_search", json={
                "query": q, "backend": backend, "topk": topk, "urls": urls, "ctx_topn": ctx_topn
            })
            j = r.json()
            backend_used, ans, ctx = from_hyde(j)
            meta = {"top_score": top_score, "decision": decision_reason, "latency_ms": int((time.time()-t0)*1000)}
            return unify_response("hyde", backend_used, ans, ctx[:ctx_topn], meta)

        else:
            raise HTTPException(500, f"未知的 route: {route}")
