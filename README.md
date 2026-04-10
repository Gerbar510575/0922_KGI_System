# KGI Advisor — AI 基金投資顧問系統

基於 **FastAPI + ChromaDB + Google Gemini + R + Docker** 的多微服務 AI 架構，整合 RAG 知識檢索、KYC 風險評估、CAPM Monte Carlo 預測與中文報告生成。

---

## 系統架構

```
使用者 (Streamlit UI :8501)
        │
        ▼
  Gateway (:8000)  ── Correlation ID middleware (X-Request-ID)
   ├── /query   ──►  RAG Service (:8002)
   │                  ├── ChromaDB（向量檢索）
   │                  └── Gemini（Embedding LRU快取 + LLM）
   ├── /advise  ──►  Advisor Service (:8003)
   │                  ├── Market Service (:8005)
   │                  │     ├── Yahoo Finance（並發批次查詢）
   │                  │     └── Redis (:6379)（TTL 90s 快取）
   │                  └── ML Bridge (:7000)
   │                        └── R LASSO 風險模型（subprocess 白名單）
   └── /report  ──►  Report Service (:8004)
                       └── Jinja2 中文報告（Markdown / PDF）
```

---

## 服務說明

| 服務 | Port | 主要技術 | 說明 |
|------|------|----------|------|
| `ui` | 8501 | Streamlit | 前端介面（KYC 表單 + RAG 問答 + 報告） |
| `gateway` | 8000 | FastAPI + httpx | API 入口；統一路由；Correlation ID 追蹤 |
| `rag-service` | 8002 | FastAPI + ChromaDB + Gemini | 基金文件 RAG 問答；智慧 metadata 過濾 |
| `advisor-service` | 8003 | FastAPI + CAPM + Monte Carlo | KYC 風險評估 → Beta 篩選 → 預測模擬 |
| `market-service` | 8005 | FastAPI + yahoo_fin + Redis | 個股報價、成交量熱度、基金熱度 |
| `ml-bridge` | 7000 | FastAPI + R subprocess | R LASSO 模型橋接；命令白名單保護 |
| `report-service` | 8004 | FastAPI + Jinja2 + ReportLab | 中文報告生成（Markdown / HTML / PDF） |
| `redis` | 6379 | Redis 7 | 市場資料快取 |

---

## 投資建議流程

```
1. KYC 輸入（年齡、收入、貸款狀況…）
        │
2. ML Bridge → R LASSO 模型 → 風險分數 p ∈ [0,1]
        │
3. p > 0.6 → 積極型 → Beta > 1 基金池
   p ≤ 0.6 → 保守型 → Beta < 1 基金池
        │
4. Market Service /fund_heat → 基金成交量熱度排名
        │
5. 挑選熱度最高的基金 → CAPM + i.i.d. 殘差 Monte Carlo
        │
6. 輸出：P5 / Median / P95 走勢圖 + 中文報告
```

---

## 快速啟動

### 1. 環境設定

```bash
cp .env.example .env
# 填入 GENAI_API_KEY（必填）
```

### 2. 建置與啟動

```bash
make build    # 建置所有 Docker image
make up       # 啟動所有服務（背景執行）
```

### 3. 開啟介面

```
http://localhost:8501
```

### 常用指令

```bash
make logs       # 查看 log（即時串流）
make ps         # 查看服務狀態
make restart    # 重啟所有服務
make down       # 停止所有服務
make clean      # 停止並刪除 volume
make reindex    # 重建 ChromaDB 向量索引
make test       # 執行單元測試（pytest）
make lint       # 靜態分析（ruff）
```

---

## API 說明

所有請求皆經由 Gateway `:8000`。每個回應皆包含 `X-Request-ID` header 可用於跨服務追蹤。

### `POST /query` — RAG 問答

```json
// Request
{ "query": "G006 2025年8月的持股分布？" }

// Response
{
  "answer": "...",
  "passages": [{ "rank": 1, "similarity": 0.92, "metadata": {...}, "snippet": "..." }]
}
```

> `query` 限制 512 字元以內。RAG Service 自動從 query 提取基金代碼（如 `G006`）、月份、文件類型進行過濾。

---

### `POST /advise` — 投資建議

```json
// Request
{
  "kyc": {
    "Gender": "Male",
    "Married": "Yes",
    "ApplicantIncome": 80000,
    "LoanAmount": 200,
    "Credit_History": 1
  }
}

// Response
{
  "selected_fund": {
    "code": "G006", "name": "凱基雲端趨勢基金",
    "beta": 1.14
  },
  "fund_forecast": {
    "fund_name": "凱基雲端趨勢基金",
    "price_scenarios": {
      "P5_10d": 54.2, "Median_10d": 57.1, "P95_10d": 61.8
    },
    "forecast_plot": "data:image/png;base64,..."
  },
  "debug_info": {
    "risk_type": "高風險承受度族群",
    "risk_score": 0.73,
    "qualified_funds": ["G006_USD", "G013_USD"]
  }
}
```

---

### `POST /report` — 報告生成

將 `/advise` 的回應直接轉傳：

```bash
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d @advise_response.json
```

---

## 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `GENAI_API_KEY` | Google Generative AI 金鑰（必填） | — |
| `GEMINI_MODEL` | LLM 模型 | `gemini-2.5-flash` |
| `EMBED_MODEL` | Embedding 模型 | `models/gemini-embedding-001` |
| `CHROMA_DIR` | ChromaDB 路徑 | `/app/chroma_db` |
| `DB_NAME` | ChromaDB collection 名稱 | `funds` |
| `REDIS_HOST` / `REDIS_PORT` | Redis 連線 | `redis` / `6379` |
| `MARKET_TTL_SEC` | 市場資料快取秒數 | `90` |
| `BACKUP_CSV_PATH` | 備援歷史資料 CSV | `/app/data/backup.csv` |
| `ML_BRIDGE_URL` | ML Bridge URL | `http://ml-bridge:7000` |
| `MKT_URL` | Market Service URL | `http://market:8005` |

---

## 專案結構

```
0922_KGI/
├── apps/
│   ├── advisor-service/     # CAPM + Monte Carlo + KYC 風險評估
│   │   ├── app.py
│   │   └── data/            # funds.csv, holdings.json
│   ├── gateway/             # API 路由 + Correlation ID middleware
│   │   └── main.py
│   ├── market-service/      # 報價 / 熱度（並發 Yahoo Finance + Redis 快取）
│   │   └── service.py
│   ├── ml-bridge/           # R subprocess 橋接（命令白名單）
│   │   ├── service.py
│   │   ├── config/models.yaml
│   │   └── models/r/        # LASSO 訓練腳本 + crisis_model_bundle.rds
│   ├── rag-service/         # ChromaDB + Gemini Embedding（LRU 快取）
│   │   └── app.py
│   ├── report-service/      # Jinja2 中文報告
│   │   └── service.py
│   ├── ui/                  # Streamlit 前端
│   └── embed_funds.py       # 建立 / 更新基金向量索引
├── tests/                   # 單元測試（pytest）
│   ├── test_gateway_validation.py
│   ├── test_market_utils.py
│   └── test_rag_validation.py
├── chroma_db/               # ChromaDB 向量資料庫（本地持久化）
├── configs/                 # rag.yaml 等服務設定
├── data/                    # 基金元資料 JSON
├── infra/
│   ├── docker-compose.yml   # 含 healthcheck + restart policy
│   └── docker/              # 各服務 Dockerfile
├── scripts/                 # 初始化 / Demo 腳本
├── .env.example
└── Makefile
```

---

## 技術棧

| 層次 | 技術 |
|------|------|
| API Framework | FastAPI + uvicorn |
| LLM / Embedding | Google Gemini 2.5 Flash + gemini-embedding-001 |
| 向量資料庫 | ChromaDB（PersistentClient） |
| 統計模型 | CAPM、i.i.d. 殘差 Monte Carlo（NumPy + Statsmodels） |
| ML 模型 | R glmnet LASSO（tidymodels workflow） |
| 快取 | Redis 7（TTL 90s 市場資料；LRU 256 Embedding） |
| 前端 | Streamlit 1.37+ |
| 容器化 | Docker Compose（healthcheck + restart: unless-stopped） |
| 測試 / Lint | pytest + ruff |
