import numpy as np, pandas as pd

def midpoint_target(rules, risk_level):
    lohi = rules["risk_buckets"][risk_level]
    w = {k: (v[0]+v[1])/2 for k,v in lohi.items()}
    s = sum(w.values())
    return {k: v/s for k,v in w.items()}

def pick_products(products: pd.DataFrame, target: dict, prefs:list, heat:dict, topn=3):
    picks = []
    for ac, w in target.items():
        pool = products[products["asset_class"]==ac].copy()
        # 偏好加分：低費用/ESG/美元資產等
        score = -pool["fee"].rank(pct=True)
        if "ESG" in prefs and "esg" in pool.columns:
            score += 0.2*pool["esg"].fillna(0)
        # 熱度加分：相對量能分數
        score += pool["ticker"].map(lambda t: 0.1*heat.get(t, {}).get("rel_volume_score", 0))
        pool["score"] = score
        top = pool.sort_values("score", ascending=False).head(topn)
        picks += top.assign(weight=w/len(top)).to_dict("records")
    return picks
