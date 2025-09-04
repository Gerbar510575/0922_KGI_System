from fastapi import FastAPI
import pandas as pd, requests, os
from typing import Dict, Any, List

app = FastAPI(title="Advisor Service")

# === 檔案路徑 ===
FUNDS_PATH = "data/funds.csv"
#MODEL_PATH = "models/risk_model.pkl"

# === 載入資料 ===
funds = pd.read_csv(FUNDS_PATH)
#clf = joblib.load(MODEL_PATH)

MKT = os.getenv("MKT_URL", "http://market:8005")
MLB = os.getenv("ML_BRIDGE_URL", "http://ml-bridge:7000")  # ML Bridge

# === 風險屬性推斷 ===
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

# === 配置規則 ===
def target_by_risk(risk: str) -> Dict[str, float]:
    if risk == "積極":
        return {"equity_fund": 0.75, "bond_fund": 0.20, "cash": 0.05}
    if risk == "穩健":
        return {"equity_fund": 0.45, "bond_fund": 0.45, "cash": 0.10}
    return {"equity_fund": 0.25, "bond_fund": 0.65, "cash": 0.10}

# === 核心：挑選基金 ===
def pick_funds(
    funds_df: pd.DataFrame,
    target: Dict[str, float],
    prefs: List[str],
    fund_heat: Dict[str, Any],
    topn=2
):
    picks = []
    # 僅挑選凱基自家基金
    base = funds_df[funds_df["kgi"].astype(str).str.lower().isin(["yes", "true", "1"])]

    for cat, w in target.items():
        subgroup = base[base["category"] == cat].copy()
        if subgroup.empty or w <= 0:
            continue

        # 加上熱度分數
        subgroup["heat"] = subgroup["code"].map(
            lambda c: fund_heat.get(c, {}).get("rel_volume_score", 0.0)
        )

        # 排序：費用低 > AUM 大 > 熱度高
        subgroup = subgroup.sort_values(by=["fee", "aum", "heat"], ascending=[True, False, False])
        top = subgroup.head(topn)

        if len(top) == 0:
            picks.append({"category": cat, "note": "此類別無可用基金"})
            continue

        picks += top.assign(weight=w/len(top)).to_dict("records")

    return picks

# === API ===
@app.post("/advise")
def advise(payload: dict):
    kyc = payload.get("kyc", {})

    # 1) 透過 ML Bridge + 外部語言模型推斷風險
    inferred = infer_risk_via_bridge(kyc)

    # 2) 配置目標
    target = target_by_risk(inferred)

    # 3) 聚合基金熱度
    fund_codes = funds["code"].tolist()
    resp = requests.post(
        f"{MKT}/fund_heat",
        json={"fund_codes": fund_codes},
        timeout=20
    )
    fund_heat = resp.json().get("data", {}) if resp.status_code == 200 else {}

    # 4) 挑基金
    picks = pick_funds(funds, target, kyc.get("preferences", []), fund_heat, topn=2)

    explanation = [
        f"系統透過跨語言模型推斷您的風險屬性為「{inferred}」，配置目標：" +
        "、".join([f"{k}:{int(v*100)}%" for k, v in target.items()]),
        "候選僅限凱基自家基金，並依費用低、AUM 大、熱度高進行挑選。"
    ]

    return {
        "risk_inferred": inferred,
        "target_allocation": target,
        "picks": picks,
        "explanation": explanation,
        "market_heat": {"source": resp.json().get("source", "mixed"), "data": fund_heat}
    }


