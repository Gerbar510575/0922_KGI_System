import pandas as pd, requests
from fastapi import HTTPException
import os

MKT = os.getenv("MKT_URL", "http://market:8005")

def get_daily_returns(tickers: list) -> pd.DataFrame:
    try:
        resp = requests.post(f"{MKT}/history",
                             json={"tickers": tickers, "days": 180},
                             timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data",{})
        frames=[]
        for t, rows in data.items():
            df=pd.DataFrame(rows)
            if "date" in df.columns and "close" in df.columns and not df.empty:
                df["date"]=pd.to_datetime(df["date"])
                df=df.sort_values("date").set_index("date")
                df[t]=df["close"].pct_change()
                frames.append(df[[t]])
        if frames:
            return pd.concat(frames,axis=1).dropna()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_daily_returns failed: {e}")
    return pd.DataFrame()
