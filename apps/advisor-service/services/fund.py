import numpy as np, matplotlib.pyplot as plt, io, base64, pandas as pd
from services.capm import fit_capm
from services.garch import fit_best_garch
from services.simulation import simulate_stock_paths

def forecast_fund_with_plot(fund_code: str, holdings_map: dict,
                            daily_return: pd.DataFrame,
                            horizon: int=20, n_sim: int=1000):
    tickers=list(holdings_map[fund_code].keys())
    weights=np.array(list(holdings_map[fund_code].values()),dtype=float)
    weights/=weights.sum()

    market_returns=daily_return.mean(axis=1)
    expected_mkt_excess=(market_returns - 0.045/252).mean()

    stock_forecasts={}
    for t in tickers:
        try:
            res=fit_capm(daily_return[t], market_returns)
            garch=fit_best_garch(res["residuals"])
            last_price=(1+daily_return[t]).cumprod().iloc[-1]*100
            paths=simulate_stock_paths(res["alpha"], res["beta"],
                                       expected_mkt_excess, garch,
                                       last_price, horizon=horizon, n_sim=n_sim)
            stock_forecasts[t]=np.percentile(paths[:,-1],[5,50,95])
        except:
            stock_forecasts[t]=[np.nan,np.nan,np.nan]

    fund_p5=fund_median=fund_p95=0
    for i,t in enumerate(tickers):
        vals=stock_forecasts.get(t,[0,0,0])
        fund_p5+=weights[i]*vals[0]
        fund_median+=weights[i]*vals[1]
        fund_p95+=weights[i]*vals[2]

    # 繪圖
    plt.figure(figsize=(10,5))
    plt.plot([0,horizon],[fund_median,fund_median],'b--',label="Median")
    plt.plot([0,horizon],[fund_p5,fund_p5],'r--',label="P5")
    plt.plot([0,horizon],[fund_p95,fund_p95],'g--',label="P95")
    plt.title(f"Fund {fund_code} Forecast (CAPM+GARCH, {horizon} days)")
    plt.xlabel("Days Ahead"); plt.ylabel("Forecasted Price"); plt.legend()
    buf=io.BytesIO(); plt.savefig(buf,format="png",bbox_inches="tight"); plt.close()
    img_base64=base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "fund_code": fund_code,
        f"P5_{horizon}d": fund_p5,
        f"Median_{horizon}d": fund_median,
        f"P95_{horizon}d": fund_p95,
        "forecast_plot": f"data:image/png;base64,{img_base64}"
    }
