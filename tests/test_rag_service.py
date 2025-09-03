import pytest
from fastapi.testclient import TestClient
import sys, os

# 把 rag-service app 加入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "apps", "rag-service"))
import service  # 匯入 apps/rag-service/service.py

client = TestClient(service.app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["service"] == "rag"

def test_search_minimal(monkeypatch):
    # 模擬 embed_query 回傳固定向量
    monkeypatch.setattr(service, "embed_query", lambda x: [0.1]*768)

    # 模擬 QdrantClient.search 回傳假資料
    class FakeHit:
        score = 0.9
        payload = {"source": "fake.pdf", "chunk": "ETF 投資風險"}

    class FakeQC:
        def search(self, *args, **kwargs):
            return [FakeHit()]

    monkeypatch.setattr(service, "QdrantClient", lambda **kwargs: FakeQC())

    payload = {"query": "ETF是什麼？", "topk": 1}
    res = client.post("/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert data["hits"][0]["source"] == "fake.pdf"

def test_baseline_fallback(monkeypatch):
    # 強制 retriever invoke 回傳假文件
    class FakeDoc:
        page_content = "ETF 是一種基金"
        metadata = {"source": "fake.pdf"}
    class FakeRetriever:
        def invoke(self, q): return [FakeDoc()]

    monkeypatch.setattr(service, "build_qdrant_retriever", lambda top_k: FakeRetriever())
    monkeypatch.setattr(service, "_gclient", None)  # 跳過真正 Gemini API

    res = service._baseline_answer("ETF是什麼？")
    assert "answer" in res
    assert "contexts" in res
