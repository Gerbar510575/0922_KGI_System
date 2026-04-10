"""
測試 RAG Service 的 QueryRequest 驗證與 build_where_clause 邏輯。
不啟動服務、不呼叫 Gemini API。
"""
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# mock 掉 chromadb 和 google.genai，讓 app.py 可以 import
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", MagicMock())
sys.modules.setdefault("google.genai.types", MagicMock())
sys.modules.setdefault("google.api_core", MagicMock())
sys.modules.setdefault("google.api_core.retry", MagicMock())

with patch("chromadb.PersistentClient", return_value=MagicMock()):
    from apps.rag_service.app import QueryRequest, build_where_clause


class TestQueryRequestValidation:
    def test_valid(self):
        req = QueryRequest(query="G006 持股？")
        assert req.query == "G006 持股？"

    def test_empty_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_too_long_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 513)

    def test_strips_whitespace(self):
        req = QueryRequest(query="  G006  ")
        assert req.query == "G006"


class TestBuildWhereClause:
    def test_no_keywords_returns_none(self):
        result = build_where_clause("什麼是基金？")
        assert result is None

    def test_fund_code_extracted(self):
        result = build_where_clause("G006 的績效如何？")
        assert result is not None
        assert result.get("fund_code") == {"$eq": "G006"}

    def test_date_extracted(self):
        result = build_where_clause("2024年3月的月報")
        assert result is not None
        assert result.get("asof_date") == {"$contains": "2024-03"}

    def test_doc_type_extracted(self):
        result = build_where_clause("公開說明書的費用說明")
        assert result is not None
        assert result.get("doc_type") == {"$eq": "公開說明書"}
