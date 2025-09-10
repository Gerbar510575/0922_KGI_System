# Market Service

`market_service` 提供市場數據 API，包含 **個股報價**、**成交量熱度** 與 **基金熱度**，可供 `advisor_service` 與 `strategy_service` 使用。

---

## 📂 專案架構

```
market_service/
│
├── app.py                   # FastAPI 主程式，定義 API 路由
├── data/
│   └── sample_prices.csv     # 備援樣本數據 (包含 ticker, date, open, high, low, close, volume)
├── services/
│   ├── cache.py              # Redis 快取工具
│   ├── yahoo.py              # Yahoo Finance 抓取工具
│   ├── backup.py             # 備援資料讀取
│   ├── quotes.py             # /quotes API
│   ├── heat.py               # /heat API
│   └── fund_heat.py          # /fund_heat API
├── requirements.txt
└── README.md
```

---

## 🚀 安裝與啟動

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 啟動服務

```bash
uvicorn app:app --reload --port 8005
```

啟動後可透過 [http://localhost:8005/docs](http://localhost:8005/docs) 查看 API 文件 (Swagger UI)。

---

## 📘 API 說明

### ▶ `/quotes`

* **功能**: 回傳個股最新報價與前收盤價。
* **輸入**:

```json
{ "tickers": ["AAPL", "MSFT"] }
```

* **輸出**:

```json
{
  "source": "live",
  "data": {
    "AAPL": {"price": 176.2, "prev_close": 175.1, "currency": "", "time": 1694332800},
    "MSFT": {"price": 321.5, "prev_close": 319.7, "currency": "", "time": 1694332800}
  }
}
```

### ▶ `/heat`

* **功能**: 用成交量的 Z-score 衡量個股交易熱度。
* **計算公式**:

$$
  score = \frac{V_{last} - \mu_{30}}{\sigma_{30} + 1e{-9}}
$$

* **輸入**:

```json
{ "tickers": ["AAPL", "MSFT"] }
```

* **輸出**:

```json
{
  "source": "live",
  "data": {
    "AAPL": {"rel_volume_score": 1.82},
    "MSFT": {"rel_volume_score": -0.25}
  }
}
```

### ▶ `/fund_heat`

* **功能**: 聚合基金持股的量能熱度 (加權平均)。
* **輸入**:

```json
{ "fund_codes": ["G006_USD"] }
```

* **輸出**:

```json
{
  "source": "live",
  "data": {
    "G006_USD": {"rel_volume_score": 0.75, "coverage": 8, "missing": ["ANET"]}
  }
}
```

---

## 📦 requirements.txt

```txt
fastapi==0.115.0
uvicorn==0.30.1
pandas==2.2.2
numpy==1.26.4
redis==5.0.4
yahoo_fin==0.8.9.1
requests==2.32.3
```

### 額外建議 (選用)

* `matplotlib` → 若需繪圖
* `scipy` → 若需更多統計方法

---

## 🧩 系統整合建議

* **market\_service** → 提供基礎數據 (價格/成交量/基金熱度)
* **strategy\_service** → 提供策略與訊號
* **advisor\_service** → 綜合風險屬性、基金持股、預測與策略，最終給投資建議

---

## 📝 範例呼叫

```bash
curl -X POST http://localhost:8005/heat \
  -H "Content-Type: application/json" \
  -d '{"tickers":["AAPL","MSFT"]}'
```

回傳:

```json
{
  "source": "live",
  "data": {
    "AAPL": {"rel_volume_score": 1.5},
    "MSFT": {"rel_volume_score": -0.3}
  }
}
```
