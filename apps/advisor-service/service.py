from fastapi import FastAPI, HTTPException
import pandas as pd, requests, os, json
from typing import Dict, Any, List
import statsmodels.api as sm
import numpy as np

app = FastAPI(title="Advisor Service")

# === 檔案路徑 ===
FUNDS_PATH = "data/funds.csv"
HOLDINGS_PATH = "data/holdings.json"

# === 載入資料 ===
funds = pd.read_csv(FUNDS_PATH)

MKT = os.getenv("MKT_URL", "http://market:8005")
MLB = os.getenv("ML_BRIDGE_URL", "http://ml-bridge:7000")  # ML Bridge

# === 1、風險屬性推斷 ===
def infer_risk_via_bridge(kyc: Dict[str, Any]) -> str:
    """
    呼叫 ML Bridge 的 R 模型 (risk_r) 推斷風險。
    約定：
      - 若回傳 prob 則用閾值映射到 保守/穩健/積極
      - 若沒有 prob，則依 predictions 值做最簡單映射
    """
    try:
        resp = requests.post(
            f"{MLB}/predict",
            json={"model_id": "risk_r", "input": kyc},
            timeout=20
        )
        resp.raise_for_status()
        out = resp.json().get("output", {})

        prob = out.get("prob")
        if prob:
            # 取第一個機率
            p = float(prob[0]) if isinstance(prob, list) else float(prob)
            if p > 0.7:
                return "積極"
            if p > 0.4:
                return "穩健"
            return "保守"

        preds = out.get("predictions")
        if preds:
            v = str(preds[0])
            return "積極" if v in ["1", "Y", "Yes", "Positive", "TRUE"] else "保守"

    except Exception:
        # 發生錯誤時 fallback 到穩健
        pass

    return "穩健"


# === 2、抓取每日報酬 ===
def get_daily_returns(tickers: List[str]) -> pd.DataFrame:
    try:
        resp = requests.post(
            f"{MKT}/history",
            json={"tickers": tickers, "days": 180},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        frames = []
        for t, rows in data.items():
            df = pd.DataFrame(rows)
            if "date" in df.columns and "close" in df.columns and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                df = df.set_index("date")
                df[t] = df["close"].pct_change()
                frames.append(df[[t]])
        if frames:
            return pd.concat(frames, axis=1).dropna()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_daily_returns failed: {e}")
    return pd.DataFrame()


# === 3、計算個股 beta ===
def compute_stock_betas(daily_return: pd.DataFrame) -> dict:
    market_returns = daily_return.mean(axis=1)
    excess_market = market_returns - (0.045/252)
    betas = {}
    for ticker in daily_return.columns:
        y = daily_return[ticker] - (0.045/252)
        X = sm.add_constant(excess_market)
        try:
            model = sm.OLS(y, X).fit()
            # 改成用名稱取值，不要用 index
            beta_val = model.params.iloc[1] if len(model.params) > 1 else 0.0
            betas[ticker] = float(beta_val)
        except Exception:
            betas[ticker] = 0.0
    return betas


# === 4、計算基金 beta (持股加權平均) ===
def compute_fund_betas(holdings_map: dict, stock_betas: dict) -> dict:
    fund_betas = {}
    for fc, comps in holdings_map.items():
        weights = np.array(list(comps.values()))
        weights = weights / weights.sum()
        tickers = list(comps.keys())
        bvals = [stock_betas.get(t, 0.0) for t in tickers]
        fund_betas[fc] = float(np.dot(weights, bvals))
    return fund_betas





# === API ===
@app.post("/advise")
def advise(payload: dict):
    kyc = payload.get("kyc", {})
    beta_pref = kyc.get("beta_pref", [0.8, 1.2])  # 使用者 Beta 偏好範圍

    # Step 1: 模型推斷風險屬性
    inferred = infer_risk_via_bridge(kyc)

    # Step 2: 定義風險屬性對應合理 Beta 區間
    risk_beta_ranges = {
        "保守": (0.0, 0.9),
        "穩健": (0.9, 1.2),
        "積極": (1.2, 3.0)
    }
    expected_range = risk_beta_ranges.get(inferred, (0.9, 1.2))

    # Step 3: 載入 holdings & 日報酬
    holdings_map = json.load(open(HOLDINGS_PATH, "r", encoding="utf-8"))
    tickers = list({t for fc in holdings_map for t in holdings_map[fc]})
    try:
        daily_return = get_daily_returns(tickers)
    except NotImplementedError:
        raise HTTPException(status_code=500, detail="尚未實作 get_daily_returns()")

    # Step 4: 計算股票 beta 與基金 beta
    stock_betas = compute_stock_betas(daily_return)
    fund_betas = compute_fund_betas(holdings_map, stock_betas)

    # Step 5: 挑選符合 Beta 偏好的基金
    selected_fund, selected_beta = None, None
    for fc, b in fund_betas.items():
        if beta_pref[0] <= b <= beta_pref[1]:
            if selected_beta is None or abs(b - np.mean(beta_pref)) < abs(selected_beta - np.mean(beta_pref)):
                selected_fund, selected_beta = fc, b

    if not selected_fund:
        raise HTTPException(status_code=404, detail="沒有基金符合指定的 Beta 區間")

    # Step 6: 聚合基金熱度
    resp = requests.post(f"{MKT}/fund_heat", json={"fund_codes": [selected_fund]}, timeout=20)
    fund_heat = resp.json().get("data", {}).get(selected_fund, {})

    # Step 7: 基金基本資料
    fmeta = funds[funds["code"] == selected_fund].iloc[0].to_dict()
    pick = {
        "name": fmeta["name"],
        "code": fmeta["code"],
        "category": fmeta["category"],
        "fee": fmeta["fee"],
        "currency": fmeta["currency"],
        "aum": fmeta["aum (NTD million)"],
        "manager": fmeta["manager"],
        "beta": selected_beta,
        "heat": fund_heat.get("rel_volume_score", 0.0)
    }

    # Step 8: 適合度檢查
    suitability_flags = []
    if not (expected_range[0] <= np.mean(beta_pref) <= expected_range[1]):
        suitability_flags.append({
            "code": selected_fund,
            "issue": f"您的 Beta 偏好範圍 {beta_pref} 與系統推斷風險屬性「{inferred}」不符 (建議範圍 {expected_range})"
        })

    # Step 9: 回傳
    explanation = [
        f"根據您輸入的 Beta 偏好範圍 {beta_pref}，系統挑選出 {pick['name']} (Beta={selected_beta:.2f})",
        f"系統模型推斷您屬於「{inferred}」型投資人 (建議 Beta 範圍 {expected_range})",
        "基金 Beta 來自持股股票 Beta 的加權平均，並參考基金市場熱度。"
    ]

    return {
        "risk_inferred": inferred,
        "selected_fund": pick,
        "explanation": explanation,
        "suitability_flags": suitability_flags,
        "market_heat": fund_heat
    }

@app.get("/debug_betas")
def debug_betas():
    try:
        holdings_map = json.load(open(HOLDINGS_PATH, "r", encoding="utf-8"))
        tickers = list({t for fc in holdings_map for t in holdings_map[fc]})
        daily_return = get_daily_returns(tickers)

        stock_betas = {}
        market_returns = daily_return.mean(axis=1)
        excess_market = market_returns - (0.02/252)

        for ticker in daily_return.columns:
            y = daily_return[ticker] - (0.02/252)
            X = sm.add_constant(excess_market)
            model = sm.OLS(y, X).fit()
            
            # Debug：印出 summary
            print(f"=== {ticker} ===")
            print(model.summary())
            
            beta_val = model.params.iloc[1] if len(model.params) > 1 else 0.0
            stock_betas[ticker] = float(beta_val)

        fund_betas = compute_fund_betas(holdings_map, stock_betas)

        return {
            "stock_betas": stock_betas,
            "fund_betas": fund_betas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"debug_betas failed: {e}")


