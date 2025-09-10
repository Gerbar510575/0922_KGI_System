# services/stock.py
import numpy as np, matplotlib.pyplot as plt, io, base64, pandas as pd
from services.capm import fit_capm
from services.garch import fit_best_garch
from services.simulation import simulate_stock_paths

def forecast_stock_with_plot(ticker: str, daily_return: pd.DataFrame,
                             horizon: int=20, n_sim: int=1000):
    market_returns = daily_return.mean(axis=1)
    expected_mkt_excess = (market_returns - 0.045/252).mean()

    try:
        res = fit_capm(daily_return[ticker], market_returns)
        garch = fit_best_garch(res["residuals"])
        last_price = (1+daily_return[ticker]).cumprod().iloc[-1]*100
        paths = simulate_stock_paths(res["alpha"], res["beta"],
                                     expected_mkt_excess, garch,
                                     last_price, horizon=horizon, n_sim=n_sim)
        p5, median, p95 = np.percentile(paths[:,-1],[5,50,95])
    except:
        p5=median=p95=np.nan

    # 繪圖
    plt.figure(figsize=(10,5))
    plt.plot([0,horizon],[median,median],'b--',label="Median")
    plt.plot([0,horizon],[p5,p5],'r--',label="P5")
    plt.plot([0,horizon],[p95,p95],'g--',label="P95")
    plt.title(f"Stock {ticker} Forecast (CAPM+GARCH, {horizon} days)")
    plt.xlabel("Days Ahead"); plt.ylabel("Forecasted Price"); plt.legend()
    buf=io.BytesIO(); plt.savefig(buf,format="png",bbox_inches="tight"); plt.close()
    img_base64=base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "ticker": ticker,
        f"P5_{horizon}d": p5,
        f"Median_{horizon}d": median,
        f"P95_{horizon}d": p95,
        "forecast_plot": f"data:image/png;base64,{img_base64}"
    }
