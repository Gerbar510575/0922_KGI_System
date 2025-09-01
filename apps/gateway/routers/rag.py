# apps/gateway/routers/rag.py
from fastapi import APIRouter
import httpx, os

RAG_URL = os.getenv("RAG_URL", "http://rag:8002")
router = APIRouter(prefix="/rag", tags=["rag"])

@router.post("/search")
async def search(payload: dict):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{RAG_URL}/search", json=payload)
    return r.json()

@router.post("/multi_query")
async def multi_query(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RAG_URL}/multi_query_search", json=payload)
    return r.json()

@router.post("/fusion")
async def fusion(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RAG_URL}/rag_fusion_search", json=payload)
    return r.json()

@router.post("/decompose")
async def decompose(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RAG_URL}/decompose_search", json=payload)
    return r.json()

@router.post("/step_back")
async def step_back(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RAG_URL}/step_back_search", json=payload)
    return r.json()

@router.post("/hyde")
async def hyde(payload: dict):
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{RAG_URL}/hyde_search", json=payload)
    return r.json()
