"""
測試 Market Service 的工具函式。
不依賴 Redis / Yahoo Finance，只測純邏輯。
"""
import pytest
import datetime
import pandas as pd
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- 替換 Redis / yahoo_fin，讓 service 可以 import ---
sys.modules.setdefault("redis", MagicMock())
sys.modules.setdefault("yahoo_fin", MagicMock())
sys.modules.setdefault("yahoo_fin.stock_info", MagicMock())

from apps.market_service.service import yf_get_histories, load_backup


class TestYfGetHistories:
    def test_empty_tickers_returns_empty(self):
        df = yf_get_histories([], days=10)
        assert df.empty

    def test_parallel_fetch_merges_results(self):
        """各 ticker 各回傳 3 筆資料，合併後應有 6 筆。"""
        def fake_history(ticker, start, end, interval="1d"):
            return pd.DataFrame({
                "Open": [1, 2, 3],
                "Close": [1.1, 2.1, 3.1],
                "Volume": [100, 200, 300],
                "ticker": [ticker] * 3,
            }, index=pd.date_range("2024-01-01", periods=3))

        from apps import market_service
        with patch.object(market_service.service, "yf_get_history", side_effect=fake_history):
            df = yf_get_histories(["AAPL", "MSFT"], days=10)

        assert len(df) == 6
        assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_failed_ticker_skipped(self):
        """某 ticker 失敗（回傳空 DataFrame）時不影響其他 ticker。"""
        def fake_history(ticker, start, end, interval="1d"):
            if ticker == "FAIL":
                return pd.DataFrame()
            return pd.DataFrame({
                "Close": [1.0, 2.0],
                "ticker": [ticker] * 2,
            }, index=pd.date_range("2024-01-01", periods=2))

        from apps import market_service
        with patch.object(market_service.service, "yf_get_history", side_effect=fake_history):
            df = yf_get_histories(["OK", "FAIL"], days=10)

        assert len(df) == 2
        assert "FAIL" not in df["ticker"].values


class TestLoadBackup:
    def test_returns_none_when_file_missing(self, tmp_path):
        with patch.dict(os.environ, {"BACKUP_CSV_PATH": str(tmp_path / "nonexistent.csv")}):
            # reload 讓環境變數生效
            import importlib
            from apps.market_service import service as svc
            importlib.reload(svc)
            result = svc.load_backup()
        assert result is None

    def test_loads_valid_csv(self, tmp_path):
        csv_file = tmp_path / "backup.csv"
        csv_file.write_text("ticker,date,Close,Volume\nAAPL,2024-01-01,100.0,1000\n")
        with patch.dict(os.environ, {"BACKUP_CSV_PATH": str(csv_file)}):
            import importlib
            from apps.market_service import service as svc
            importlib.reload(svc)
            df = svc.load_backup()
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
