import numpy as np

def simulate_stock_paths(alpha, beta, expected_market_return, garch_model,
                         last_price, horizon=20, n_sim=1000, seed=42):
    forecast=garch_model.forecast(horizon=horizon)
    cond_vol=np.sqrt(forecast.variance.values[-1,:])
    np.random.seed(seed)
    paths=[]
    for _ in range(n_sim):
        price=last_price
        path=[price]
        for t in range(horizon):
            mu=alpha + beta*expected_market_return
            sigma=cond_vol[t]
            shock=np.random.normal(0, sigma)
            r=mu+shock
            price*=np.exp(r)
            path.append(price)
        paths.append(path)
    return np.array(paths)
