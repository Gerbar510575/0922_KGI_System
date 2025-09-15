from fastapi import FastAPI, HTTPException
import httpx
import os
import logging
import traceback

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# Service URLs (注意: 要對應 docker-compose.yml 內的容器名稱與 port)
MKT = os.getenv("MKT_URL", "http://market:8005")
RAG = os.getenv("RAG_URL", "http://rag:8002")
ADV = os.getenv("ADV_URL", "http://advisor:8003")
RPT = os.getenv("RPT_URL", "http://report:8004")

app = FastAPI(title="KFH Advisor Gateway")


@app.get("/health")
def health():
    return {"ok": True}


# ----------------- RAG /auto -----------------
@app.post("/query")
async def rag_auto(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{RAG}/query", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "downstream": "rag/auto",
                "status": e.response.status_code,
                "body": e.response.text,
            },
        )
    except Exception as e:
        logger.error(f"gateway /query error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"gateway /query error: {e}")


# ----------------- Advisor -----------------
@app.post("/advise")
async def advise(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{ADV}/advise", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={
                "downstream": "advisor/advise",
                "status": e.response.status_code,
                "body": e.response.text,
            },
        )
    except Exception as e:
        logger.error(f"gateway /advise error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"gateway /advise error: {e}")


# ----------------- Report -----------------
@app.post("/report")
async def report(payload: dict):
    logger.info(f"收到 /report 請求 payload: {payload}")
    try:
        async with httpx.AsyncClient(timeout=180) as c:  # 報告可能比較久，拉長 timeout
            resp = await c.post(f"{RPT}/report", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"/report 下游錯誤: status={e.response.status_code}, body={e.response.text}"
        )
        raise HTTPException(
            status_code=502,
            detail={
                "downstream": "report/report",
                "status": e.response.status_code,
                "body": e.response.text,
            },
        )
    except Exception as e:
        logger.error(f"gateway /report error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"gateway /report error: {e}")




