from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import os
import uuid
import logging

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("gateway")

# ----------------- Service URLs -----------------
MKT = os.getenv("MKT_URL", "http://market:8005")
RAG = os.getenv("RAG_URL", "http://rag:8002")
ADV = os.getenv("ADV_URL", "http://advisor:8003")
RPT = os.getenv("RPT_URL", "http://report:8004")

# ----------------- 共用 AsyncClient（lifespan 管理） -----------------
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient()
    logger.info("AsyncClient 已建立")
    yield
    await http_client.aclose()
    logger.info("AsyncClient 已關閉")

app = FastAPI(title="KFH Advisor Gateway", lifespan=lifespan)

# ----------------- Correlation ID Middleware -----------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """為每個請求生成（或透傳）X-Request-ID，寫入 log 並附在回應 header。"""
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        logger.info(f"[{req_id}] {request.method} {request.url.path}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

app.add_middleware(RequestIDMiddleware)

# ----------------- Pydantic 請求模型 -----------------
class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query 不得為空")
        if len(v) > 512:
            raise ValueError("query 長度不得超過 512 字元")
        return v

class AdviseRequest(BaseModel):
    kyc: dict

# ----------------- Health Check -----------------
@app.get("/health")
async def health():
    """驗證所有下游服務是否可達，回傳彙整狀態。"""
    checks = {"rag": RAG, "advisor": ADV, "report": RPT, "market": MKT}
    results: dict = {}
    for name, url in checks.items():
        try:
            resp = await http_client.get(f"{url}/health", timeout=3)
            results[name] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception:
            results[name] = "unreachable"
    overall = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    return {"status": overall, "services": results}

# ----------------- RAG -----------------
@app.post("/query")
async def rag_auto(req: QueryRequest, request: Request):
    req_id = request.state.request_id
    try:
        resp = await http_client.post(
            f"{RAG}/query",
            json=req.model_dump(),
            timeout=60,
            headers={"X-Request-ID": req_id},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={"downstream": "rag/query", "status": e.response.status_code},
        )
    except Exception as e:
        logger.exception(f"[{req_id}] gateway /query error")
        raise HTTPException(status_code=500, detail=f"gateway /query error: {e}")

# ----------------- Advisor -----------------
@app.post("/advise")
async def advise(req: AdviseRequest, request: Request):
    req_id = request.state.request_id
    try:
        resp = await http_client.post(
            f"{ADV}/advise",
            json=req.model_dump(),
            timeout=60,
            headers={"X-Request-ID": req_id},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail={"downstream": "advisor/advise", "status": e.response.status_code},
        )
    except Exception as e:
        logger.exception(f"[{req_id}] gateway /advise error")
        raise HTTPException(status_code=500, detail=f"gateway /advise error: {e}")

# ----------------- Report -----------------
@app.post("/report")
async def report(payload: dict, request: Request):
    req_id = request.state.request_id
    logger.info(f"[{req_id}] 收到 /report 請求")
    try:
        resp = await http_client.post(
            f"{RPT}/report",
            json=payload,
            timeout=180,
            headers={"X-Request-ID": req_id},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[{req_id}] /report 下游錯誤: status={e.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail={"downstream": "report/report", "status": e.response.status_code},
        )
    except Exception as e:
        logger.exception(f"[{req_id}] gateway /report error")
        raise HTTPException(status_code=500, detail=f"gateway /report error: {e}")
