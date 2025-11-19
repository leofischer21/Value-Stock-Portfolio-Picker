import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os
import subprocess


def _annualize_return(cum_return, years):
    if years <= 0:
        return 0.0
    return (cum_return ** (1.0 / years)) - 1.0


def max_drawdown(wealth_index: pd.Series) -> float:
    roll_max = wealth_index.cummax()
    drawdown = (wealth_index - roll_max) / roll_max
    return float(drawdown.min())


def run_backtest_from_weights(weights: Dict[str, float], prices: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, rebalance: str = 'M', capital: float = 1.0):
    """Run a simple backtest given `weights` mapping ticker->target weight and a `prices` DataFrame (Adj Close).

    - prices: DataFrame indexed by date, columns are tickers.
    - rebalance: pandas offset alias for rebalancing frequency (default 'M' monthly)
    - capital: starting capital (float)
    Returns a dict with metrics and time series.
    """
    px = prices.copy()
    if start:
        px = px[px.index >= pd.to_datetime(start)]
    if end:
        px = px[px.index <= pd.to_datetime(end)]

    # forward fill missing prices
    px = px.ffill().dropna(axis=1, how='all')

    tickers = [t for t in weights.keys() if t in px.columns]
    if not tickers:
        raise ValueError('No tickers with price data available for backtest')

    px = px[tickers]

    # Compute returns
    rets = px.pct_change().fillna(0.0)

    # Determine rebalance dates
    rebalance_dates = px.resample(rebalance).last().index

    # Initialize positions according to weights at first rebalance date
    wealth = pd.Series(index=px.index, dtype=float)
    cash = 0.0
    # track current weights
    current_weights = {t: 0.0 for t in tickers}
    # initial allocation at first available date
    current_weights.update({t: float(weights.get(t, 0.0)) for t in tickers})

    # compute daily portfolio return
    weighted_returns = (rets * pd.Series(current_weights)).sum(axis=1)
    # but need to apply rebalancing: at each rebalance date reset weights
    portfolio_returns = pd.Series(index=px.index, dtype=float)
    current_weights = {t: float(weights.get(t, 0.0)) for t in tickers}
    for i, date in enumerate(px.index):
        if date in rebalance_dates:
            # reset to target weights (no transaction costs)
            current_weights = {t: float(weights.get(t, 0.0)) for t in tickers}
        portfolio_returns.loc[date] = (rets.loc[date] * pd.Series(current_weights)).sum()

    wealth_index = (1 + portfolio_returns).cumprod() * capital

    total_years = (wealth_index.index[-1] - wealth_index.index[0]).days / 365.25
    cum_return = float(wealth_index.iloc[-1] / wealth_index.iloc[0])
    cagr = _annualize_return(cum_return, total_years)
    ann_vol = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = float(cagr / ann_vol) if ann_vol > 0 else 0.0
    mdd = max_drawdown(wealth_index)

    metrics = {
        'cagr': cagr,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'cum_return': cum_return,
        'start': str(wealth_index.index[0].date()),
        'end': str(wealth_index.index[-1].date()),
    }

    return {
        'metrics': metrics,
        'wealth_index': wealth_index,
        'returns': portfolio_returns,
    }


def save_backtest_result(result: Dict[str, Any], out_dir: str = 'examples', name_prefix: str = 'backtest'):
    Path = os.path
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = os.path.join(out_dir, f"{name_prefix}_{ts}.json")
    # add metadata: git commit if available
    meta = result.get('metrics', {}).copy()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        meta['git_commit'] = commit
    except Exception:
        meta['git_commit'] = None

    to_save = {'metrics': meta}
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return filename
