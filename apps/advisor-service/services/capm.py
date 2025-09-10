import statsmodels.api as sm
import pandas as pd

def fit_capm(asset_returns: pd.Series, market_returns: pd.Series, rf: float=0.045/252):
    y=asset_returns - rf
    X=sm.add_constant(market_returns - rf)
    model=sm.OLS(y,X).fit()
    return {
        "alpha": model.params["const"],
        "beta": model.params.iloc[1],
        "r2": model.rsquared,
        "residuals": model.resid
    }
