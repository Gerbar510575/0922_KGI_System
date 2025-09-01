from fastapi import FastAPI
import httpx, os
from .routers import rag  # 若放平行資料夾，請依實際相對匯入調整

MKT  = os.getenv("MKT_URL",  "http://market:8001")
RAG  = os.getenv("RAG_URL",  "http://rag:8002")
ADV  = os.getenv("ADV_URL",  "http://advisor:8003")
RPT  = os.getenv("RPT_URL",  "http://report:8004")

app = FastAPI(title="KFH Advisor Gateway")
app.include_router(rag.router)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/advise") #處理「建議」相關請求
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

@app.post("/justify") #處理「要求理由」相關請求
async def justify(payload: dict):
    async with httpx.AsyncClient(timeout=30) as c:
        # 從 RAG 服務獲取理由
        r = await c.post(f"{RAG}/search", json=payload)
    return r.json()

@app.post("/report") #處理「報告」生成相關請求
async def report(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        # 從 RPT 服務獲取生成的報告
        r = await c.post(f"{RPT}/generate", json=payload)
    return r.json()
