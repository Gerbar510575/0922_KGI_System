# app.py
from fastapi import FastAPI, HTTPException
import pandas as pd, os, json, numpy as np, requests, math
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io, base64

# === 初始化 FastAPI ===
app = FastAPI(title="Advisor Service")

# === 檔案路徑 ===
FUNDS_PATH = "data/funds.csv"
HOLDINGS_PATH = "data/holdings.json"

funds = pd.read_csv(FUNDS_PATH)
MKT = os.getenv("MKT_URL", "http://market:8005")
MLB = os.getenv("ML_BRIDGE_URL", "http://ml-bridge:7000")

# ---------------------------------------------------
# Service: Risk
# ---------------------------------------------------
def infer_risk_via_bridge(kyc: dict) -> dict:
    try:
        resp = requests.post(
            f"{MLB}/predict",
            json={"model_id": "risk_r", "input": kyc},
            timeout=20
        )
        resp.raise_for_status()
        out = resp.json().get("output", {})
        prob = out.get("prob")
        if prob is not None:
            p = float(prob[0]) if isinstance(prob, list) else float(prob)
            if p > 0.7: return {"type": "積極", "p": p}
            if p > 0.4: return {"type": "穩健", "p": p}
            return {"type": "保守", "p": p}
    except Exception:
        pass
    return {"type": "穩健", "p": 0.5}

# ---------------------------------------------------
# Service: Data Loader
# ---------------------------------------------------
def get_daily_returns(tickers: list) -> pd.DataFrame:
    try:
        resp = requests.post(
            f"{MKT}/history",
            json={"tickers": tickers, "days": 90},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        frames = []
        for t, rows in data.items():
            df = pd.DataFrame(rows)
            if "date" in df.columns and "close" in df.columns and not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")
                df[t] = df["close"].pct_change()
                frames.append(df[[t]])
        if frames:
            return pd.concat(frames, axis=1).dropna()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_daily_returns failed: {e}")
    return pd.DataFrame()

# ---------------------------------------------------
# Service: CAPM
# ---------------------------------------------------
def fit_capm(asset_returns: pd.Series, market_returns: pd.Series, rf: float = 0.045/252):
    """以 OLS 估 alpha / beta，回傳殘差做之後的 i.i.d. 模擬"""
    y = asset_returns - rf
    X = sm.add_constant(market_returns - rf)
    try:
        model = sm.OLS(y, X, missing="drop").fit()
        return {
            "alpha": float(model.params.get("const", 0.0)),
            "beta": float(model.params.iloc[1] if len(model.params) > 1 else 0.0),
            "r2": float(model.rsquared),
            "residuals": model.resid.dropna()
        }
    except Exception:
        # 保持簡潔：回傳零殘差與零參數
        return {
            "alpha": 0.0, "beta": 0.0, "r2": 0.0,
            "residuals": pd.Series(index=asset_returns.index, dtype=float)
        }

def compute_stock_betas(daily_return: pd.DataFrame) -> dict:
    market_returns = daily_return.mean(axis=1)
    out = {}
    for t in daily_return.columns:
        res = fit_capm(daily_return[t], market_returns)
        out[t] = res["beta"]
    return out

def compute_fund_betas(holdings_map: dict, stock_betas: dict) -> dict:
    fund_betas = {}
    for fc, comps in holdings_map.items():
        weights = np.array(list(comps.values()), dtype=float)
        weights /= weights.sum()
        tickers = list(comps.keys())
        bvals = [stock_betas.get(t, 0.0) for t in tickers]
        fund_betas[fc] = float(np.dot(weights, bvals))
    return fund_betas

# ---------------------------------------------------
# Service: CAPM + i.i.d. 殘差 模擬
# ---------------------------------------------------
def simulate_stock_paths_iid(alpha, beta, expected_market_excess, residuals: np.ndarray,
                             last_price: float, horizon: int = 10, n_sim: int = 1000, seed: int = 42):
    """
    使用 CAPM 漂移 + i.i.d. 殘差 (歷史抽樣 bootstrap) 的 Monte Carlo 模擬。
    r_t = (alpha + beta * E[Rm - Rf]) + eps_t,  eps_t ~ i.i.d. (from CAPM residuals)
    price_{t+1} = price_t * exp(r_t)
    """
    if last_price <= 0 or not np.isfinite(last_price):
        raise ValueError("last_price 非法")

    # 殘差樣本
    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size < 10:
        raise ValueError("CAPM 殘差不足以進行 i.i.d. 模擬")

    mu = float(alpha + beta * expected_market_excess)
    rng = np.random.default_rng(seed)

    paths = np.empty((n_sim, horizon + 1), dtype=float)
    for i in range(n_sim):
        price = last_price
        path = [price]
        # 以「歷史殘差」i.i.d. 抽樣（保留厚尾/偏態）
        eps = rng.choice(resid, size=horizon, replace=True)
        for t in range(horizon):
            r = mu + eps[t]
            price *= np.exp(r)
            path.append(price)
        paths[i, :] = path
    return paths  # (n_sim, horizon+1)

# ---------------------------------------------------
# 視覺化 & 預測包裝
# ---------------------------------------------------
def plot_forecast_with_paths(title: str, horizon: int, simulated_paths: np.ndarray) -> str:
    plt.figure(figsize=(12, 6))
    k = min(50, simulated_paths.shape[0])
    plt.plot(simulated_paths[:k].T, color="grey", alpha=0.3)
    plt.plot(range(horizon+1), np.median(simulated_paths, axis=0), lw=2, label="Median")
    plt.plot(range(horizon+1), np.percentile(simulated_paths, 5, axis=0), lw=2, ls="--", label="5%")
    plt.plot(range(horizon+1), np.percentile(simulated_paths, 95, axis=0), lw=2, ls="--", label="95%")
    plt.title(title); plt.xlabel("Days Ahead"); plt.ylabel("Price"); plt.legend()
    buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def forecast_stock_with_plot(ticker: str, daily_return: pd.DataFrame,
                             horizon: int = 10, n_sim: int = 1000):
    series = daily_return[ticker].dropna()
    if series.empty:
        return {"ticker": ticker, "error": "無報酬資料"}

    # 市場與 CAPM 參數
    market_returns = daily_return.mean(axis=1).loc[series.index]
    expected_mkt_excess = float((market_returns - 0.045/252).mean())
    res = fit_capm(series, market_returns)

    # 基準價（把最後累積乘回 100 的基準）
    last_price = float((1.0 + series).cumprod().iloc[-1] * 100.0)

    # Monte Carlo（CAPM 漂移 + i.i.d. 殘差）
    try:
        paths = simulate_stock_paths_iid(
            res["alpha"], res["beta"], expected_mkt_excess,
            res["residuals"].values, last_price,
            horizon=horizon, n_sim=n_sim
        )
    except Exception as e:
        print(f"[警告] 股票 {ticker} 模擬失敗：{e}")
        return {"ticker": ticker, "error": f"simulate failed: {e}"}

    # 圖與分位數
    img = plot_forecast_with_paths(f"Stock {ticker} Forecast (CAPM+i.i.d., {horizon}d)", horizon, paths)
    last_prices = paths[:, -1]

        # ===== Debug 印出 =====
    print(f"[DEBUG] {ticker} 模擬結果前10筆: {last_prices[:10]}")
    print(f"[DEBUG] {ticker} 分位數 -> "
          f"P5={np.percentile(last_prices,5)}, "
          f"Median={np.percentile(last_prices,50)}, "
          f"P95={np.percentile(last_prices,95)}")

    return {
        "ticker": ticker,
        f"P5_{horizon}d": float(np.percentile(last_prices, 5)),
        f"Median_{horizon}d": float(np.percentile(last_prices, 50)),
        f"P95_{horizon}d": float(np.percentile(last_prices, 95)),
        "forecast_plot": f"data:image/png;base64,{img}"
    }

def forecast_fund_with_plot(fund_code: str, holdings_map: dict, daily_return: pd.DataFrame,
                            horizon: int = 10, n_sim: int = 1000):
    tickers = list(holdings_map[fund_code].keys())
    weights = np.array(list(holdings_map[fund_code].values()), dtype=float)
    weights /= weights.sum()

    market_returns_all = daily_return.mean(axis=1)
    stock_paths = []
    for i, t in enumerate(tickers):
        series = daily_return[t].dropna()
        if series.empty:
            print(f"[警告] {t} 無報酬資料，跳過")
            continue

        # CAPM 參數（用各該股票的對齊市場報酬）
        market_returns = market_returns_all.loc[series.index]
        expected_mkt_excess = float((market_returns - 0.045/252).mean())
        res = fit_capm(series, market_returns)
        last_price = float((1.0 + series).cumprod().iloc[-1] * 100.0)

        try:
            paths = simulate_stock_paths_iid(
                res["alpha"], res["beta"], expected_mkt_excess,
                res["residuals"].values, last_price,
                horizon=horizon, n_sim=n_sim
            )
            stock_paths.append(weights[i] * paths)
        except Exception as e:
            print(f"[警告] 股票 {t} 模擬失敗：{e}")
            continue

    if not stock_paths:
        return {"fund_code": fund_code, "error": "基金沒有可用股票模擬"}

    # 基金層級價格路徑 = 權重 * 個股模擬價格 的加總
    fund_paths = np.sum(stock_paths, axis=0)

    img = plot_forecast_with_paths(f"Fund {fund_code} Forecast (CAPM+i.i.d., {horizon}d)", horizon, fund_paths)
    last_prices = fund_paths[:, -1]

    print(f"[DEBUG] Fund {fund_code} 模擬最後價 last_prices: {last_prices[:10]}")
    print(f"[DEBUG] Fund {fund_code} 分位數: "
          f"P5={np.percentile(last_prices, 5)}, "
          f"Median={np.percentile(last_prices, 50)}, "
          f"P95={np.percentile(last_prices, 95)}")


    return {
        "fund_code": fund_code,
        "price_scenarios": {
            f"P5_{horizon}d": float(np.percentile(last_prices, 5)),
            f"Median_{horizon}d": float(np.percentile(last_prices, 50)),
            f"P95_{horizon}d": float(np.percentile(last_prices, 95)),
        },
        "forecast_plot": f"data:image/png;base64,{img}"
    }

# ---------------------------------------------------
# API
# ---------------------------------------------------
@app.post("/advise")
def advise(payload: dict):
    kyc = payload.get("kyc", {})
    risk_info = infer_risk_via_bridge(kyc)
    p_value = risk_info["p"]

    holdings_map = json.load(open(HOLDINGS_PATH, "r", encoding="utf-8"))
    tickers = list({t for fc in holdings_map for t in holdings_map[fc]})
    daily_return = get_daily_returns(tickers)

    # Beta & 基金挑選
    stock_betas = compute_stock_betas(daily_return)
    fund_betas = compute_fund_betas(holdings_map, stock_betas)

    target_high = (p_value > 0.5)
    selected_fund, selected_beta = None, None
    for fc, b in fund_betas.items():
        if target_high and b > 1:
            selected_fund, selected_beta = fc, b; break
        if not target_high and b < 1:
            selected_fund, selected_beta = fc, b; break
    if not selected_fund:
        raise HTTPException(status_code=404, detail="沒有符合風險邏輯的基金")

    # 市場熱度
    fund_heat = {}
    try:
        resp = requests.post(f"{MKT}/fund_heat",
                             json={"fund_codes": [selected_fund]},
                             timeout=20)
        fund_heat = resp.json().get("data", {}).get(selected_fund, {})
    except:
        pass

    # 基金基本資料
    fmeta = funds[funds["code"] == selected_fund].iloc[0].to_dict()
    fmeta["beta"] = selected_beta

    # 預測（CAPM + i.i.d. 殘差）
    fund_forecast = forecast_fund_with_plot(selected_fund, holdings_map, daily_return)
    stock_forecasts = {t: forecast_stock_with_plot(t, daily_return)
                       for t in holdings_map[selected_fund].keys()}

    result = {
        "selected_fund": fmeta,
        "fund_forecast": fund_forecast,
        "stock_betas": {t: stock_betas.get(t, 0.0) for t in holdings_map[selected_fund].keys()},
        "stock_forecasts": stock_forecasts,
        "market_heat": fund_heat
    }
    return result

@app.post("/forecast_stock")
def forecast_stock(payload: dict):
    ticker = payload.get("ticker")
    if not ticker:
        raise HTTPException(status_code=400, detail="需要提供股票代碼 ticker")
    daily_return = get_daily_returns([ticker])
    if ticker not in daily_return.columns:
        raise HTTPException(status_code=404, detail=f"找不到 {ticker} 的日報酬資料")
    stock_forecast = forecast_stock_with_plot(ticker, daily_return)
    return {"stock_forecast": stock_forecast}





