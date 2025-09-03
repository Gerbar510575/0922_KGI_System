from fastapi import FastAPI, HTTPException
import httpx
import os
import asyncio
from routers import rag  # 保留既有 /rag/... 相關路由（非 /rag/auto）

# Service URLs (注意: 要對應 docker-compose.yml 內的容器名稱與 port)
MKT = os.getenv("MKT_URL", "http://market:8005")
RAG = os.getenv("RAG_URL", "http://rag:8002")
ADV = os.getenv("ADV_URL", "http://advisor:8003")
RPT = os.getenv("RPT_URL", "http://report:8004")

app = FastAPI(title="KFH Advisor Gateway")

app.include_router(rag.router)
# app.include_router(auto_rag.router)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rag/auto")
async def rag_auto(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{RAG}/auto", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        detail = {
            "downstream": "rag/auto",
            "status": e.response.status_code,
            "body": e.response.text,
        }
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gateway /rag/auto error: {e}")

@app.post("/advise")
async def advise(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        m_task = c.post(f"{MKT}/heat", json={"tickers": payload.get("universe", [])})
        a_task = c.post(f"{ADV}/advise", json=payload)
        m_res, a_res = await asyncio.gather(m_task, a_task)

    try:
        m_json = m_res.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Market service bad response: {m_res.text}")

    try:
        a_json = a_res.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Advisor service bad response: {a_res.text}")

    a_json["market_heat"] = m_json
    return a_json


@app.post("/justify")
async def justify(payload: dict):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{RAG}/search", json=payload)
    return r.json()

@app.post("/report")
async def report(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RPT}/generate", json=payload)
    return r.json()


