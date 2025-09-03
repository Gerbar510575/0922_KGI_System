from fastapi import FastAPI
import pandas as pd, requests, os, joblib
from typing import Dict, Any, List

app = FastAPI(title="Advisor Service")

# === 檔案路徑 ===
FUNDS_PATH = "data/funds.csv"
MODEL_PATH = "models/risk_model.pkl"

# === 載入資料 ===
funds = pd.read_csv(FUNDS_PATH)
clf = joblib.load(MODEL_PATH)

MKT = os.getenv("MKT_URL", "http://market:8005")

# === 工具：風險屬性推斷（用 risk_model.pkl） ===
def infer_risk(kyc: Dict[str, Any]) -> str:
    X = pd.DataFrame([kyc])
    prob = clf.predict_proba(X)[0][1]
    if prob > 0.7: return "積極"
    if prob > 0.4: return "穩健"
    return "保守"

# === 工具：配置目標 ===
def target_by_risk(risk: str) -> Dict[str, float]:
    if risk == "積極":
        return {"equity_fund": 0.75, "bond_fund": 0.20, "cash": 0.05}
    if risk == "穩健":
        return {"equity_fund": 0.45, "bond_fund": 0.45, "cash": 0.10}
    return {"equity_fund": 0.25, "bond_fund": 0.65, "cash": 0.10}

# === 核心：挑選基金 ===
def pick_funds(funds_df: pd.DataFrame, target: Dict[str, float], prefs: List[str], fund_heat: Dict[str, Any], topn=2):
    picks = []
    base = funds_df[funds_df["kgi"].astype(str).str.lower().isin(["yes","true","1"])]
    for cat, w in target.items():
        subgroup = base[base["category"] == cat].copy()
        if subgroup.empty or w <= 0:
            continue
        subgroup["heat"] = subgroup["code"].map(lambda c: fund_heat.get(c, {}).get("rel_volume_score", 0.0))
        subgroup = subgroup.sort_values(by=["fee","aum","heat"], ascending=[True, False, False])
        top = subgroup.head(topn)
        if len(top) == 0:
            picks.append({"category": cat, "note": "此類別無可用基金"})
            continue
        picks += top.assign(weight=w/len(top)).to_dict("records")
    return picks

# === API: /advise ===
@app.post("/advise")
def advise(payload: dict):
    kyc = payload.get("kyc", {})

    # 1. 模型推斷風險
    inferred = infer_risk(kyc)

    # 2. 配置目標
    target = target_by_risk(inferred)

    # 3. 市場熱度 (fund_heat)
    fund_codes = funds["code"].tolist()
    resp = requests.post(f"{MKT}/fund_heat", json={"fund_codes": fund_codes}, timeout=20).json()
    fund_heat = resp.get("data", {})

    # 4. 推薦基金
    picks = pick_funds(funds, target, kyc.get("preferences", []), fund_heat, topn=2)

    explanation = [
        f"模型推斷您的風險屬性為「{inferred}」，配置目標："
        + "、".join([f"{k}:{int(v*100)}%" for k,v in target.items()]),
        "候選僅限凱基自家基金，並依費用低、AUM大、熱度高進行挑選。"
    ]

    return {
        "risk_inferred": inferred,
        "target_allocation": target,
        "picks": picks,
        "explanation": explanation,
        "market_heat": {"source": resp.get("source","mixed"), "data": fund_heat}
    }

