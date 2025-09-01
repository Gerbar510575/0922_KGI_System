from fastapi import FastAPI
from typing import List
import os, json, time
import pandas as pd
import redis
try:
    import yfinance as yf
    YF_OK = True
except:
    YF_OK = False

app = FastAPI(title="Market Service")
rds = redis.Redis(host=os.getenv("REDIS_HOST","redis"), port=int(os.getenv("REDIS_PORT",6379)), decode_responses=True)
TTL = int(os.getenv("MARKET_TTL_SEC", "90"))

def cache_get(key): 
    v = rds.get(key)
    return json.loads(v) if v else None

def cache_set(key, val): 
    rds.setex(key, TTL, json.dumps(val))

def load_backup():
    p = os.path.join(os.path.dirname(__file__), "data", "sample_prices.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["date"])
        return df
    return None

@app.post("/quotes") # get latest price/prev_close for given tickers
def quotes(payload: dict):
    tickers: List[str] = payload.get("tickers", [])
    key = f"quotes:{','.join(sorted(tickers))}"
    cached = cache_get(key)
    if cached: return {"source":"cache", "data": cached}

    data = {}
    if YF_OK and tickers:
        try:
            for t in tickers:
                y = yf.Ticker(t).fast_info
                data[t] = {"price": float(y.get("last_price", 0)), "prev_close": float(y.get("previous_close", 0)),
                           "currency": y.get("currency",""), "time": int(time.time())}
        except Exception:
            pass

    if not data:
        # fallback to backup sample
        df = load_backup()
        if df is not None:
            last = df.sort_values("date").groupby("ticker").tail(1)
            data = {row["ticker"]: {"price": float(row["close"]), "currency": "TWD", "time": int(time.time())}
                    for _,row in last.iterrows()}

    cache_set(key, data)
    return {"source":"live" if YF_OK else "backup", "data": data}

@app.post("/heat") # get relative volume heat score for given tickers
def heat(payload: dict):
    tickers: List[str] = payload.get("tickers", [])
    key = f"heat:{','.join(sorted(tickers))}"
    cached = cache_get(key)
    if cached: return {"source":"cache", "data": cached}

    # 簡易熱度：近一日成交量 / 近30日均量 的 z-like 分數（無 yfinance 時用備援）
    heat = {}
    df = load_backup()
    if YF_OK and tickers:
        try:
            hist = yf.download(tickers=tickers, period="2mo", interval="1d", group_by="ticker", threads=True, auto_adjust=True)
            for t in tickers:
                sub = hist[t] if isinstance(hist.columns, pd.MultiIndex) else hist
                vol = sub["Volume"].dropna()
                if len(vol) >= 5:
                    score = float((vol.iloc[-1] - vol.rolling(30).mean().iloc[-1]) / (vol.rolling(30).std().iloc[-1] + 1e-9))
                    heat[t] = {"rel_volume_score": score}
        except Exception:
            pass

    if not heat and df is not None:
        for t, g in df.groupby("ticker"):
            vol = g.sort_values("date")["volume"]
            if len(vol) >= 5:
                score = float((vol.iloc[-1] - vol.rolling(30).mean().iloc[-1]) / (vol.rolling(30).std().iloc[-1] + 1e-9))
                heat[t] = {"rel_volume_score": score}

    cache_set(key, heat)
    return {"source":"live" if YF_OK else "backup", "data": heat}
