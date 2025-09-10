from arch import arch_model
import numpy as np

def fit_best_garch(resid, max_ar_lag=2, lags_vol=(1,1)):
    resid = resid.dropna()
    if len(resid) < 30:  # 樣本太少直接跳過
        return None
    
    best = {"aic": np.inf, "res": None}
    vol_list = ["GARCH", "EGARCH"]
    dist_list = ["t", "skewt"]
    p, q = lags_vol
    
    for mean_lag in range(0, max_ar_lag+1):
        mean_kw = {"mean": "AR", "lags": mean_lag} if mean_lag > 0 else {"mean": "Zero"}
        for vol in vol_list:
            for dist in dist_list:
                try:
                    am = arch_model(resid, vol=vol, p=p, q=q,
                                    o=1 if vol == "EGARCH" else 0,
                                    dist=dist, **mean_kw)
                    res = am.fit(disp="off")
                    if np.isfinite(res.aic) and res.aic < best["aic"]:
                        best = {"aic": res.aic, "res": res}
                except Exception:
                    continue
    return best["res"]
