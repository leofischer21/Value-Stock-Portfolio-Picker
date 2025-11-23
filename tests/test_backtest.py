import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from scripts.backtest import run_backtest_from_weights


def make_synthetic_prices(tickers=['A','B','C'], days=252*2, start_pd=None):
    rng = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')
    np.random.seed(42)
    prices = pd.DataFrame(index=rng)
    for t in tickers:
        # geometric random walk
        returns = np.random.normal(loc=0.0005, scale=0.01, size=days)
        prices[t] = 100 * (1 + returns).cumprod()
    return prices


def test_backtest_basic_metrics():
    prices = make_synthetic_prices(['A','B','C'], days=252*2)
    weights = {'A': 0.4, 'B': 0.4, 'C': 0.2}
    result = run_backtest_from_weights(weights, prices, rebalance='M', capital=1000.0)
    metrics = result['metrics']
    # basic sanity checks
    assert 'cagr' in metrics
    assert 'annual_vol' in metrics
    assert 'max_drawdown' in metrics
    assert result['wealth_index'].iloc[0] == 1000.0
    assert result['wealth_index'].shape[0] == prices.shape[0]
    # returns should be finite
    assert result['returns'].notna().all()
