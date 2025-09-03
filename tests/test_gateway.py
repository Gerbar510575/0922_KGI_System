import pytest
from fastapi.testclient import TestClient
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "apps", "gateway"))
import main  # 匯入 apps/gateway/main.py

client = TestClient(main.app)

def test_gateway_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True

def test_gateway_rag_auto(monkeypatch):
    # 模擬 httpx AsyncClient.post 回傳固定 JSON
    class FakeResp:
        def json(self): return {"answer": "這是測試回答"}

    async def fake_post(*args, **kwargs): return FakeResp()

    monkeypatch.setattr(main.httpx.AsyncClient, "__aenter__", lambda self: self)
    monkeypatch.setattr(main.httpx.AsyncClient, "__aexit__", lambda self, *args: None)
    monkeypatch.setattr(main.httpx.AsyncClient, "post", fake_post)

    res = client.post("/rag/auto", json={"query": "ETF是什麼？"})
    assert res.status_code == 200
    assert "answer" in res.json()
