# src/run_picker.py
"""Main starter script: loads a universe, fetches basics via yfinance,
computes simple scores and writes top N portfolio.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from src.portfolio import construct_portfolio, save_portfolio
from src.fetch_dataroma import fetch_dataroma_signal

MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20
UNIVERSE_FILE = "data/sp500_tickers.csv"  # add your own file


def load_universe(path=UNIVERSE_FILE):
    try:
        return pd.read_csv(path)['ticker'].str.strip().tolist()
    except Exception:
        # fallback example set
        return ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V","MA","UNH","PG","HD","KO","PEP","ADBE","CRM","ORCL","CSCO"]


def fetch_basic_info(ticker):
    t = yf.Ticker(ticker)
    info = t.info
    return {
        'ticker': ticker,
        'marketCap': info.get('marketCap', np.nan),
        'sector': info.get('sector', 'Unknown'),
        'trailingPE': info.get('trailingPE', np.nan),
        'forwardPE': info.get('forwardPE', np.nan),
        'beta': info.get('beta', np.nan),
        'price': info.get('currentPrice', np.nan) or info.get('regularMarketPrice', np.nan),
        'returnOnEquity': info.get('returnOnEquity', np.nan),
        'grossMargins': info.get('grossMargins', np.nan),
        'debtToEquity': info.get('debtToEquity', np.nan),
    }


def gather(universe):
    rows = []
    for tk in universe:
        try:
            rows.append(fetch_basic_info(tk))
        except Exception as e:
            print('skip', tk, e)
    return pd.DataFrame(rows)


def compute_scores(df):
    df = df[df['marketCap'] >= MIN_MARKET_CAP].copy()
    if df.empty:
        raise SystemExit('Universe empty after marketcap filter')

    # value proxies
    df['pe_rank'] = df['trailingPE'].rank(ascending=True, pct=True).fillna(0)
    df['fpe_rank'] = df['forwardPE'].rank(ascending=True, pct=True).fillna(0)
    # quality
    df['roe_rank'] = df['returnOnEquity'].rank(ascending=False, pct=True).fillna(0)
    df['gm_rank'] = df['grossMargins'].rank(ascending=False, pct=True).fillna(0)
    df['debt_rank'] = df['debtToEquity'].rank(ascending=True, pct=True).fillna(0)

    # dataroma signal (0/1) - boost if superinvestor recently bought
    df['dataroma_buy'] = df['ticker'].apply(lambda t: fetch_dataroma_signal(t))

    # combine
    df['value_score'] = 0.6*df['pe_rank'] + 0.4*df['fpe_rank']
    df['quality_score'] = 0.5*df['roe_rank'] + 0.4*df['gm_rank'] + 0.1*df['debt_rank']
    df['composite'] = 0.45*df['value_score'] + 0.45*df['quality_score'] + 0.10*df['dataroma_buy']
    df['final_score'] = df['composite'].rank(ascending=False, pct=True)

    return df


def main():
    universe = load_universe()
    print('universe size', len(universe))
    df = gather(universe)
    scored = compute_scores(df)
    portfolio = construct_portfolio(scored, n=PORTFOLIO_SIZE)
    save_portfolio(portfolio, 'examples/selected_portfolio.csv')
    print('done. saved examples/selected_portfolio.csv')


if __name__ == '__main__':
    main()