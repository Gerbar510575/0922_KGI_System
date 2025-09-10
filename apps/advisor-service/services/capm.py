import statsmodels.api as sm
import pandas as pd
import math

def safe_float(x, default=0.0):
    """確保數值可以轉成 JSON 合法 float"""
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return default
    return float(x)

def fit_capm(asset_returns: pd.Series, market_returns: pd.Series, rf: float = 0.045/252):
    y = asset_returns - rf
    X = sm.add_constant(market_returns - rf)
    
    try:
        model = sm.OLS(y, X, missing="drop").fit()
        alpha = safe_float(model.params.get("const", 0.0))
        beta = safe_float(model.params.iloc[1] if len(model.params) > 1 else 0.0)
        r2 = safe_float(model.rsquared)
        return {
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "residuals": model.resid.fillna(0.0)  # 保證沒有 NaN
        }
    except Exception:
        # OLS 爆炸時回傳預設值
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "r2": 0.0,
            "residuals": pd.Series([0.0] * len(asset_returns), index=asset_returns.index)
        }
