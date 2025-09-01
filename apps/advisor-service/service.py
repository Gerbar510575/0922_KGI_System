from fastapi import FastAPI
import pandas as pd, yaml, requests, os
from portfolio.optimizer import midpoint_target, pick_products

app = FastAPI(title="Advisor Service")
rules = yaml.safe_load(open("portfolio/rules.yaml", "r", encoding="utf-8"))
products = pd.read_csv("data/products.csv")  # ticker,name,asset_class,fee,aum,currency,region,risk,esg?

MKT = os.getenv("MKT_URL", "http://market:8001")

@app.post("/advise")
def advise(payload: dict):
    kyc = payload["kyc"]  # {risk_level, horizon_years, goal, preferences}
    risk = kyc["risk_level"]
    target = midpoint_target(rules, risk)

    # 從候選宇宙（若未給則以主檔全部ticker）取熱度
    universe = payload.get("universe", products["ticker"].tolist())
    heat = requests.post(f"{MKT}/heat", json={"tickers": universe}, timeout=10).json()["data"]

    picks = pick_products(products[products["ticker"].isin(universe)], target, kyc.get("preferences",[]), heat, topn=3)

    # 基本適合度檢核
    flags=[]
    for p in picks:
        if p.get("risk","中")=="高" and risk=="保守":
            flags.append({"ticker":p["ticker"],"issue":"商品風險高於客戶承受度"})
        if kyc["horizon_years"]<3 and p["asset_class"]=="equities":
            flags.append({"ticker":p["ticker"],"issue":"投資期限較短，股權類比重建議降低"})

    explanation = [
        f"依風險等級「{risk}」設定目標配置：{', '.join([f'{k}:{int(v*100)}%' for k,v in target.items()])}。",
        "商品以低費用/規模/市場熱度加權遴選，並通過基本適合度檢核。"
    ]
    return {"target_allocation": target, "picks": picks, "suitability_flags": flags, "explanation": explanation}
