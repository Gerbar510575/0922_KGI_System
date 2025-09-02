from fastapi import FastAPI, HTTPException
import httpx
import os
from routers import rag  # 保留既有 /rag/... 相關路由（非 /rag/auto）

MKT = os.getenv("MKT_URL", "http://market:8001")
RAG = os.getenv("RAG_URL", "http://rag:8002")
ADV = os.getenv("ADV_URL", "http://advisor:8003")
RPT = os.getenv("RPT_URL", "http://report:8004")

app = FastAPI(title="KFH Advisor Gateway")

app.include_router(rag.router)
#app.include_router(auto_rag.router)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rag/auto")
async def rag_auto(payload: dict):
    """
    轉發到 RAG 服務的 /auto，回傳其 JSON 結果。
    payload 範例：
    {
      "query": "...",
      "backend": "qdrant",
      "topk": 3,
      "ctx_topn": 6,
      "strategy": "multi_query" | "rag_fusion" | "step_back" | "hyde" | "decompose"
    }
    """
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{RAG}/auto", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        # 將下游錯誤回拋，保留狀態碼與訊息
        detail = {"downstream": "rag/auto", "status": e.response.status_code, "body": e.response.text}
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gateway /rag/auto error: {e}")


@app.post("/advise")  # 處理「建議」相關請求
async def advise(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        # 從 MKT 服務獲取市場熱度資訊
        m = c.post(f"{MKT}/heat", json={"tickers": payload.get("universe", [])})
        # 從 ADV 服務獲取投資建議
        a = c.post(f"{ADV}/advise", json=payload)
        m_res, a_res = await m, await a
    advice = a_res.json()
    advice["market_heat"] = m_res.json()
    return advice

@app.post("/justify")  # 處理「要求理由」相關請求
async def justify(payload: dict):
    async with httpx.AsyncClient(timeout=30) as c:
        # 轉發到 RAG 服務的 /search
        r = await c.post(f"{RAG}/search", json=payload)
    return r.json()

@app.post("/report")  # 處理「報告」生成相關請求
async def report(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        # 轉發到報告服務
        r = await c.post(f"{RPT}/generate", json=payload)
    return r.json()

