from fastapi import FastAPI
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import os, json, time, datetime
import pandas as pd
import redis
import logging
import json
HOLDINGS_PATH = os.getenv("HOLDINGS_PATH", "/app/data/holdings.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Yahoo Finance -----------------------------------------------------------
YF_OK = True
try:
    # 你提供的寫法：以 headers 參數穩定抓資料
    import yahoo_fin.stock_info as si
except Exception:
    YF_OK = False


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# --- FastAPI / Redis ---------------------------------------------------------
app = FastAPI(title="Market Service")
rds = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)
TTL = int(os.getenv("MARKET_TTL_SEC", "90"))

def cache_get(key: str):
    v = rds.get(key)
    return json.loads(v) if v else None

def cache_set(key: str, val: Any):
    rds.setex(key, TTL, json.dumps(val))



# --- Yahoo 工具：抓一檔 ticker 的歷史資料，統一欄位 ------------------------------
def yf_get_history(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    從 yahoo_fin 取得單一 ticker 在區間內的歷史資料。
    - 會統一欄位名稱為：Open / High / Low / Close / Adj Close / Volume
    - 加上 'ticker' 欄位
    - 若抓不到、或為空，回傳空 DataFrame
    """
    try:
        df = si.get_data(
            ticker,
            start_date=str(start_date),
            end_date=str(end_date),
            index_as_date=True,
            interval=interval,
            headers=HEADERS,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # 你範例的欄位命名映射
        rename_map = {
            "adjclose": "Adj Close",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "open": "Open",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        # 補 ticker 欄位
        df["ticker"] = ticker
        # 確保只留必要欄位（若多了其他欄位也無妨）
        keep = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()

def yf_get_histories(
    tickers: List[str],
    days: int,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    並發抓多檔 ticker 的最近 N 天資料，合併成一張表。
    使用 ThreadPoolExecutor 減少串行等待時間。
    """
    if not tickers:
        return pd.DataFrame()

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=max(days + 5, 7))

    def fetch(t: str) -> pd.DataFrame:
        return yf_get_history(t, start_date, end_date, interval=interval)

    max_workers = min(len(tickers), 8)
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for df in pool.map(fetch, tickers):
            if not df.empty:
                frames.append(df)

    if frames:
        return pd.concat(frames).reset_index()  # date 變回欄位
    return pd.DataFrame()


BACKUP_CSV_PATH = os.getenv("BACKUP_CSV_PATH", "/app/data/backup.csv")

def load_backup() -> pd.DataFrame | None:
    """
    載入備援歷史資料 CSV（欄位需含 ticker / date / Close / Volume）。
    檔案不存在時回傳 None，呼叫端視為無備援資料。
    """
    if not os.path.exists(BACKUP_CSV_PATH):
        logging.warning(f"[load_backup] 備援檔案不存在：{BACKUP_CSV_PATH}")
        return None
    try:
        df = pd.read_csv(BACKUP_CSV_PATH, parse_dates=["date"])
        return df
    except Exception:
        logging.exception(f"[load_backup] 讀取備援檔案失敗：{BACKUP_CSV_PATH}")
        return None


@app.post("/quotes")
def quotes(payload: dict):
    """
    回傳每一檔：
      {
        "price": <最新 Close>,
        "prev_close": <前一日 Close>,
        "currency": "",       # Yahoo 無保證，暫留空
        "time": <unix ts>
      }
    """
    tickers: List[str] = payload.get("tickers", [])
    key = f"quotes:{','.join(sorted(tickers))}"
    cached = cache_get(key)
    if cached:
        return {"source": "cache", "data": cached}

    now_ts = int(time.time())
    data: Dict[str, Dict[str, Any]] = {}

    # 盡量先用 Yahoo，抓 10 天內資料
    live_ok = False
    if YF_OK and tickers:
        df_live = yf_get_histories(tickers, days=10, interval="1d")
        if not df_live.empty:
            live_ok = True
            for t, g in df_live.groupby("ticker"):
                g = g.sort_values("date")
                if "Close" not in g.columns or g["Close"].dropna().empty:
                    continue
                last_close = float(g["Close"].dropna().iloc[-1])
                prev_close = float(g["Close"].dropna().iloc[-2]) if g["Close"].dropna().shape[0] >= 2 else last_close
                data[t] = {
                    "price": last_close,
                    "prev_close": prev_close,
                    "currency": "",  # 不可靠來源，先留空
                    "time": now_ts,
                }

    # 若 live 沒抓到任何一檔 → fallback 到備援樣本
    if not data:
        df = load_backup()
        if df is not None:
            for t in tickers:
                g = df[df["ticker"] == t].sort_values("date")
                if "Close" not in g.columns or g["Close"].dropna().empty:
                    # 如果備援沒有 Close，就嘗試 'close'
                    base_col = "Close" if "Close" in g.columns else ("close" if "close" in g.columns else None)
                    if base_col is None or g[base_col].dropna().empty:
                        continue
                    closes = g[base_col].dropna()
                else:
                    closes = g["Close"].dropna()

                last_close = float(closes.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_close
                data[t] = {
                    "price": last_close,
                    "prev_close": prev_close,
                    "currency": "TWD",
                    "time": now_ts,
                }

    cache_set(key, data)
    return {"source": "live" if live_ok else "backup", "data": data}

@app.get("/health")
def health():
    """驗證 Redis 連線是否正常。"""
    try:
        rds.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        logging.exception("health check Redis 連線失敗")
        return {"status": "degraded", "redis": f"error: {e}"}


def load_holdings_map() -> dict:
    if os.path.exists(HOLDINGS_PATH):
        with open(HOLDINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- /heat：用成交量算相對熱度 ---------------------------------------------------
@app.post("/heat")
def heat(payload: dict):
    """
    以「近一日成交量相對近 30 日均量的 z-like 分數」當作熱度：
      score = (vol_last - mean_30) / (std_30 + 1e-9)

    回傳：
      { "source": "...", "data": { "TICKER": {"rel_volume_score": <float>} } }
    """
    tickers: List[str] = payload.get("tickers", [])
    logging.info(f"[heat] input tickers={tickers}")
    key = f"heat:{','.join(sorted(tickers))}"
    cached = cache_get(key)
    if cached:
        return {"source": "cache", "data": cached}

    scores: Dict[str, Dict[str, float]] = {}
    live_ok = False

    # 先嘗試 Yahoo：抓 90 天資料以計算 30 日均量 / 波動
    if YF_OK and tickers:
        df_live = yf_get_histories(tickers, days=90, interval="1d")
        if not df_live.empty:
            live_ok = True
            for t, g in df_live.groupby("ticker"):
                g = g.sort_values("date")
                if "Volume" not in g.columns:
                    continue
                vol = g["Volume"].dropna()
                # 至少要有一定長度，避免 rolling 全是 NaN
                if len(vol) >= 35:
                    mean_30 = vol.rolling(30).mean().iloc[-1]
                    std_30 = vol.rolling(30).std(ddof=0).iloc[-1]  # population-like
                    last = vol.iloc[-1]
                    if pd.notna(mean_30) and pd.notna(std_30):
                        score = float((last - mean_30) / (std_30 + 1e-9))
                        scores[t] = {"rel_volume_score": score}

    # 如果還是空 → 備援樣本
    if not scores:
        df = load_backup()
        if df is not None:
            for t, g in df.groupby("ticker"):
                g = g.sort_values("date")
                base_col = "Volume" if "Volume" in g.columns else ("volume" if "volume" in g.columns else None)
                if base_col is None:
                    continue
                vol = g[base_col].dropna()
                if len(vol) >= 35:
                    mean_30 = vol.rolling(30).mean().iloc[-1]
                    std_30 = vol.rolling(30).std(ddof=0).iloc[-1]
                    last = vol.iloc[-1]
                    if pd.notna(mean_30) and pd.notna(std_30):
                        score = float((last - mean_30) / (std_30 + 1e-9))
                        scores[t] = {"rel_volume_score": score}

    cache_set(key, scores)
    if live_ok:
        logging.info(f"[heat] live computed scores={scores}")
    else:
        logging.info(f"[heat] fallback scores={scores}")

    return {"source": "live" if live_ok else "backup", "data": scores}
# === 新增：基金熱度（依持股的相對量能分數做均值） ==============================
@app.post("/fund_heat")
def fund_heat(payload: dict):
    """
    參數: {"fund_codes": ["G006_USD", "G011_USD", ...]}
    回傳: 
    {
      "source": "live|backup|mixed",
      "data": {
        "G006_USD": {
          "rel_volume_score": 0.42,
          "coverage": 8,
          "missing": ["AAPL", "ANET"]
        }
      }
    }
    """
    fund_codes: List[str] = payload.get("fund_codes", [])
    logging.info(f"[fund_heat] fund_codes={fund_codes}")
    if not fund_codes:
        return {"source": "mixed", "data": {}}

    holdings_map = load_holdings_map()  # fund_code -> {ticker: weight}
    logging.info(f"[fund_heat] holdings_map keys={list(holdings_map.keys())[:5]}")

    # 收集所有需要的股票
    all_tickers = sorted({t for fc in fund_codes for t in holdings_map.get(fc, {}).keys()})
    logging.info(f"[fund_heat] all_tickers={all_tickers}")

    if not all_tickers:
        return {
            "source": "mixed",
            "data": {fc: {"rel_volume_score": 0.0, "coverage": 0, "missing": []} for fc in fund_codes}
        }

    # 個股熱度
    heat_resp = heat({"tickers": all_tickers})
    per_ticker = heat_resp.get("data", {})
    source = heat_resp.get("source", "mixed")

    # 聚合基金熱度（加權平均）
    result = {}
    for fc in fund_codes:
        ts = holdings_map.get(fc, {})
        scores = []
        weights = []
        missing = []

        for t, w in ts.items():
            s = per_ticker.get(t, {}).get("rel_volume_score")
            if s is None:
                missing.append(t)
            else:
                scores.append(float(s) * float(w))
                weights.append(float(w))

        agg = float(sum(scores) / sum(weights)) if weights else 0.0
        result[fc] = {"rel_volume_score": agg, "coverage": len(weights), "missing": missing}

    logging.info(f"[fund_heat] result={result}")
    return {"source": source, "data": result}
# --- /history：抓多檔股票的歷史收盤價 -------------------------------------------
@app.post("/history")
def history(payload: dict):
    """
    參數: {"tickers": ["MSFT","NVDA"], "days": 180}
    回傳: {"data": {ticker: [{"date": "...", "close": ...}, ...]}}
    """
    tickers: List[str] = payload.get("tickers", [])
    days: int = payload.get("days", 180)

    df = yf_get_histories(tickers, days=days, interval="1d")
    if df.empty:
        return {"source": "empty", "data": {}}

    result = {}
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        result[t] = [
            {"date": str(d), "close": float(c)}
            for d, c in zip(g["date"], g["Close"])
            if not pd.isna(c)
        ]
    return {"source": "live", "data": result}


