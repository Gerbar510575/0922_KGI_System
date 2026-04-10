"""
測試 Gateway 的 Pydantic 輸入驗證邏輯。
不啟動服務，只驗證 model 行為。
"""
import pytest
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 直接 import 模型類別（不啟動 FastAPI app）
from apps.gateway.main import QueryRequest, AdviseRequest


class TestQueryRequest:
    def test_valid_query(self):
        req = QueryRequest(query="G006 的費用率是多少？")
        assert req.query == "G006 的費用率是多少？"

    def test_strips_whitespace(self):
        req = QueryRequest(query="  G006  ")
        assert req.query == "G006"

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_exceeds_max_length_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="A" * 513)

    def test_exact_max_length_ok(self):
        req = QueryRequest(query="A" * 512)
        assert len(req.query) == 512


class TestAdviseRequest:
    def test_valid_kyc(self):
        req = AdviseRequest(kyc={"age": 35, "income": 100000})
        assert req.kyc["age"] == 35

    def test_empty_kyc_ok(self):
        req = AdviseRequest(kyc={})
        assert req.kyc == {}

    def test_missing_kyc_raises(self):
        with pytest.raises(ValidationError):
            AdviseRequest()
