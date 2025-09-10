from fastapi import FastAPI, HTTPException
import pandas as pd, os, json, numpy as np, requests

from services.risk import infer_risk_via_bridge
from services.data_loader import get_daily_returns
from services.capm import fit_capm
from services.fund import forecast_fund_with_plot
from services.stock import forecast_stock_with_plot

import math

def sanitize_for_json(obj):
    """遞迴清理 NaN / Inf，確保能轉成合法 JSON"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # 你也可以改成 None
        return float(obj)
    elif isinstance(obj, pd.Series):
        return [sanitize_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return obj.applymap(sanitize_for_json).to_dict(orient="list")
    return obj


app = FastAPI(title="Advisor Service")

# === 檔案路徑 ===
FUNDS_PATH = "data/funds.csv"
HOLDINGS_PATH = "data/holdings.json"

funds = pd.read_csv(FUNDS_PATH)
MKT = os.getenv("MKT_URL", "http://market:8005")


def compute_stock_betas(daily_return: pd.DataFrame) -> dict:
    """逐檔股票跑 CAPM，計算個股 Beta"""
    market_returns = daily_return.mean(axis=1)
    stock_betas = {}
    for t in daily_return.columns:
        try:
            res = fit_capm(daily_return[t], market_returns)
            stock_betas[t] = float(res["beta"])
        except Exception:
            stock_betas[t] = 0.0
    return stock_betas

def compute_fund_betas(holdings_map: dict, stock_betas: dict) -> dict:
    """基金 Beta = 持股股票 Beta 加權平均"""
    fund_betas = {}
    for fc, comps in holdings_map.items():
        weights = np.array(list(comps.values()), dtype=float)
        weights /= weights.sum()
        tickers = list(comps.keys())
        bvals = [stock_betas.get(t, 0.0) for t in tickers]
        fund_betas[fc] = float(np.dot(weights, bvals))
    return fund_betas


# === API: 基金推薦 ===
@app.post("/advise")
def advise(payload: dict):
    kyc = payload.get("kyc", {})

    # Step 1: ML Bridge 判斷風險屬性 (取得 type 與 p)
    risk_info = infer_risk_via_bridge(kyc)
    inferred_type = risk_info["type"]
    p_value = risk_info["p"]

    # Step 2: 載入基金持股與日報酬
    holdings_map = json.load(open(HOLDINGS_PATH, "r", encoding="utf-8"))
    tickers = list({t for fc in holdings_map for t in holdings_map[fc]})
    daily_return = get_daily_returns(tickers)

    # Step 3: 個股 Beta 與基金 Beta
    stock_betas = compute_stock_betas(daily_return)
    fund_betas = compute_fund_betas(holdings_map, stock_betas)

    # Step 4: 根據 p 值邏輯挑選基金
    target_high = (p_value > 0.5)
    selected_fund, selected_beta = None, None
    for fc, b in fund_betas.items():
        if target_high and b > 1:
            selected_fund, selected_beta = fc, b
            break
        if not target_high and b < 1:
            selected_fund, selected_beta = fc, b
            break

    if not selected_fund:
        raise HTTPException(status_code=404, detail="沒有符合風險邏輯的基金")

    # Step 5: 市場熱度
    fund_heat = {}
    try:
        resp = requests.post(f"{MKT}/fund_heat",
                             json={"fund_codes": [selected_fund]}, timeout=20)
        fund_heat = resp.json().get("data", {}).get(selected_fund, {})
    except:
        pass

    # Step 6: 基金基本資料
    fmeta = funds[funds["code"] == selected_fund].iloc[0].to_dict()
    fmeta["beta"] = selected_beta

    # Step 7: 預測基金價格分布 + 圖表
    fund_forecast = forecast_fund_with_plot(selected_fund, holdings_map, daily_return)

    # Step 8: 預測該基金持股的個股價格分布
    stock_forecasts = {}
    for t in holdings_map[selected_fund].keys():
        try:
            stock_forecasts[t] = forecast_stock_with_plot(t, daily_return)
        except Exception:
            stock_forecasts[t] = {}

    result = {
    "selected_fund": fmeta,
    "fund_forecast": fund_forecast,
    "stock_betas": {t: stock_betas.get(t, 0.0) for t in holdings_map[selected_fund].keys()},
    "stock_forecasts": stock_forecasts,
    "market_heat": fund_heat
}
    return sanitize_for_json(result)

# === API: 單檔股票預測 ===
@app.post("/forecast_stock")
def forecast_stock(payload: dict):
    ticker = payload.get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="需要提供股票代碼 ticker")

    daily_return = get_daily_returns([ticker])
    if ticker not in daily_return.columns:
        raise HTTPException(status_code=404, detail=f"找不到 {ticker} 的日報酬資料")

    stock_forecast = forecast_stock_with_plot(ticker, daily_return)

    result = {
    "stock_forecast": stock_forecast
}
    return sanitize_for_json(result)




