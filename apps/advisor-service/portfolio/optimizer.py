import numpy as np, pandas as pd

def midpoint_target(rules, risk_level):
    lohi = rules["risk_buckets"][risk_level]
    w = {k: (v[0]+v[1])/2 for k,v in lohi.items()}
    s = sum(w.values())
    return {k: v/s for k,v in w.items()}

def pick_products(products, target, preferences, heat, topn=3):
    picks = []
    for asset_class, w in target.items():
        # 過濾符合該資產類別的商品
        subset = products[products["asset_class"] == asset_class].copy()
        if subset.empty:
            continue

        # 偏好篩選 (例如 ESG、低費用)
        if "低費用" in preferences:
            subset = subset.sort_values("fee", ascending=True)
        if "ESG" in preferences and "esg" in subset.columns:
            subset = subset[subset["esg"] == "Yes"]

        # 加入市場熱度分數
        subset["score"] = subset["ticker"].map(
            lambda t: heat.get(t, {}).get("rel_volume_score", 0.0)
        )

        # 選前 topn
        top = subset.head(topn)
        if len(top) == 0:
            # 防呆：避免 ZeroDivisionError
            picks.append({
                "asset_class": asset_class,
                "note": "無符合條件的商品"
            })
            continue

        picks += top.assign(weight=w / len(top)).to_dict("records")

    return picks
