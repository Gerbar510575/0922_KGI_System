# Advisor Service

這是一個基於 **FastAPI** 的投資顧問服務，結合 **CAPM** 與 **GARCH** 模型，提供：

* 基金推薦 (`/advise`)
* 單檔股票價格預測 (`/forecast_stock`)

---

## 📂 專案架構

```
advisor_service/
├── app.py              # FastAPI 主程式
├── data/
│   ├── funds.csv       # 基金基本資料
│   └── holdings.json   # 基金持股資料
├── services/
│   ├── risk.py         # 風險屬性推斷 (ML Bridge)
│   ├── data_loader.py  # 抓取股價日報酬
│   ├── capm.py         # CAPM 模型
│   ├── garch.py        # GARCH 模型
│   ├── simulation.py   # 模擬工具 (Monte Carlo)
│   ├── fund.py         # 基金預測 (加權股票模擬)
│   └── stock.py        # 單檔股票預測
├── requirements.txt
└── README.md
```

---

## 🚀 使用方式

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 啟動服務

```bash
uvicorn app:app --reload --port 8000
```

### 3. API 說明

#### `/advise` (POST)

**輸入：**

```json
{
  "kyc": {
    "age": 30,
    "income": 1000000
  }
}
```

**回傳範例：**

```json
{
  "risk_inferred": "穩健",
  "risk_probability": 0.48,
  "selected_fund": {
    "code": "G006",
    "name": "凱基雲端趨勢基金",
    "category": "科技型",
    "currency": "USD",
    "aum": 500000000,
    "manager": "張經理",
    "fee": 1.5,
    "beta": 1.12
  },
  "fund_forecast": {
    "fund_code": "G006",
    "P5_20d": 95.2,
    "Median_20d": 101.3,
    "P95_20d": 109.7,
    "forecast_plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
  },
  "explanation": [
    "系統推斷您屬於「穩健」型投資人 (p=0.48)。",
    "因此推薦 凱基雲端趨勢基金 (Beta=1.12)。",
    "推薦邏輯：p>0.5 → 挑 Beta>1 的基金；否則挑 Beta<1 的基金。",
    "基金預測使用 CAPM+GARCH 模型模擬股票報酬，並加權為基金價格分布。"
  ],
  "market_heat": {
    "rel_volume_score": 0.72
  }
}
```

---

#### `/forecast_stock` (POST)

**輸入：**

```json
{
  "ticker": "AAPL"
}
```

**回傳範例：**

```json
{
  "stock_forecast": {
    "ticker": "AAPL",
    "P5_20d": 172.3,
    "Median_20d": 181.7,
    "P95_20d": 194.5,
    "forecast_plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
  }
}
```

---

## 🧩 模型邏輯

* **CAPM**：計算單檔股票 Beta 與殘差
* **GARCH/EGARCH**：對殘差建模，捕捉波動聚集效應
* **Monte Carlo**：模擬未來價格分布，輸出 P5, Median, P95
* **基金預測**：對基金持股做加權，得到基金層級的價格預測
