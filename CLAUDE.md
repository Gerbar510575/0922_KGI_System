# CLAUDE.md — KGI Advisor 專案指引

本檔案提供 Claude Code 在此專案工作時的關鍵背景知識。

---

## 專案定位

多微服務的 AI 基金投資顧問系統。所有服務由 Docker Compose 管理，透過 Gateway（`:8000`）統一對外。主要語言為 Python（FastAPI），ML 模型為 R（LASSO）。

---

## 服務入口對照

| 服務 | 主程式 | Port |
|------|--------|------|
| gateway | `apps/gateway/main.py` | 8000 |
| rag-service | `apps/rag-service/app.py` | 8002 |
| advisor-service | `apps/advisor-service/app.py` | 8003 |
| report-service | `apps/report-service/service.py` | 8004 |
| market-service | `apps/market-service/service.py` | 8005 |
| ml-bridge | `apps/ml-bridge/service.py` | 7000 |
| ui | `apps/ui/streamlit_app.py` | 8501 |

---

## 關鍵架構決策

### Correlation ID
Gateway 的 `RequestIDMiddleware` 負責生成（或透傳）`X-Request-ID`，並注入每個下游 HTTP 請求的 header。
追蹤跨服務問題時，用 `X-Request-ID` grep log。

### Shared AsyncClient
Gateway 用 `lifespan` 管理單一 `httpx.AsyncClient`，儲存在模組層級的 `http_client`。
不要在 endpoint 內 `async with httpx.AsyncClient()` 新建連線。

### Holdings Map
`apps/advisor-service/app.py` 在模組層級以 `with open(HOLDINGS_PATH)` 載入 `HOLDINGS_MAP`。
修改 holdings 資料後需重啟 advisor 容器才會生效（不是即時重讀）。

### Embedding LRU Cache
`apps/rag-service/app.py` 的 `_cached_embed()` 用 `@lru_cache(maxsize=256)` 快取 Gemini Embedding 結果。
快取以 query 字串為 key，重啟服務後清空。若要清除快取不重啟，可呼叫 `_cached_embed.cache_clear()`。

### ML Bridge 白名單
`apps/ml-bridge/service.py` 的 `ALLOWED_EXECUTABLES` 限制可執行的程式（`Rscript`、`java`、`python` 等）。
新增新語言的模型時，必須同步更新白名單，否則會在啟動時拋出 HTTP 500。

### Backup Data
Market Service 的 `load_backup()` 從 `BACKUP_CSV_PATH` 讀取備援 CSV。
CSV 格式需包含欄位：`ticker, date, Close, Volume`。

---

## 服務間依賴順序

```
redis → rag, market
market → advisor
ml-bridge → advisor
rag + advisor + report → gateway
gateway → ui
```

Docker Compose 以 `condition: service_healthy` 確保依賴就緒後才啟動下游。

---

## 常用指令

```bash
make up          # 啟動所有服務
make down        # 停止
make logs        # 串流所有服務 log
make reindex     # 重建 ChromaDB 向量索引（修改基金資料後執行）
make test        # 執行 tests/ 下的 pytest
make lint        # ruff check apps/
```

---

## 測試慣例

- 測試位於 `tests/`，以 `test_` 前綴命名
- 不啟動任何服務，以 `unittest.mock` 隔離 Redis / Yahoo Finance / Gemini
- Pydantic 驗證測試只 import model class，不依賴 FastAPI app 生命週期
- 執行：`python -m pytest tests/ -v`

---

## 常見陷阱

| 陷阱 | 說明 |
|------|------|
| `HOLDINGS_MAP` 不會即時更新 | 修改 `holdings.json` 後需 `make restart` |
| ChromaDB 啟動慢 | RAG service 啟動時同步初始化 ChromaDB，第一次啟動可能需要數十秒 |
| R 模型需要預先訓練 | `crisis_model_bundle.rds` 必須存在於 `apps/ml-bridge/models/r/`，否則 predict.R 會失敗 |
| Yahoo Finance 不穩定 | `yf_get_history` 失敗時靜默回傳空 DataFrame，fallback 到 `load_backup()` |
| Gemini API Key | 未設定 `GENAI_API_KEY` 時，RAG service 在啟動初始化就會失敗，不是在第一次查詢才報錯 |
| Docker healthcheck 需要 curl | 各 Dockerfile 需確認有安裝 `curl`；若無可改用 `wget -qO- ...` |

---

## 不要做的事

- 不要在 gateway endpoint 內新建 `httpx.AsyncClient()`（已有共用 client）
- 不要在 `/advise` 每次請求重讀 `holdings.json`（已改為模組層級載入）
- 不要在 `ml-bridge/config/models.yaml` 的 `cmd` 加入不在白名單的可執行程式
- 不要把 `GENAI_API_KEY` 寫進程式碼或 commit 進 git
