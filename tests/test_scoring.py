import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from scripts.run_picker_2 import compute_scores
from scripts.portfolio import construct_portfolio


def make_sample_df():
    # three tickers with varying metrics
    df = pd.DataFrame([
        {
            'ticker': 'AAA',
            'marketCap': 1e9,
            'sector': 'Tech',
            'trailingPE': 10.0,
            'forwardPE': 9.0,
            'beta': 1.0,
            'returnOnEquity': 0.2,
            'debtToEquity': 10.0,
            'superinvestor_score': 0.8,
            'reddit_score': 0.1,
            'x_score': 0.2,
            'ki_moat_score': 0.5,
            'pe_vs_history_score': 0.6,
            'insider_score': 0.0,
            'analyst_score': 0.5,
        },
        {
            'ticker': 'BBB',
            'marketCap': 2e9,
            'sector': 'Finance',
            'trailingPE': 20.0,
            'forwardPE': 18.0,
            'beta': 0.8,
            'returnOnEquity': 0.1,
            'debtToEquity': 5.0,
            'superinvestor_score': 0.2,
            'reddit_score': 0.9,
            'x_score': 0.6,
            'ki_moat_score': 0.2,
            'pe_vs_history_score': 0.4,
            'insider_score': 0.0,
            'analyst_score': 0.6,
        },
        {
            'ticker': 'CCC',
            'marketCap': 3e9,
            'sector': 'Tech',
            'trailingPE': 5.0,
            'forwardPE': 4.5,
            'beta': 1.2,
            'returnOnEquity': 0.3,
            'debtToEquity': 20.0,
            'superinvestor_score': 0.6,
            'reddit_score': 0.4,
            'x_score': 0.4,
            'ki_moat_score': 0.7,
            'pe_vs_history_score': 0.8,
            'insider_score': 0.1,
            'analyst_score': 0.4,
        },
    ])
    return df


def test_compute_scores_ordering():
    df = make_sample_df()
    scored = compute_scores(df, min_market_cap=0)
    # highest final_score should be for the lowest PE and decent quality (CCC expected)
    top = scored.iloc[0]['ticker']
    assert top in ('CCC', 'AAA')
    # ensure final_score column exists and is finite
    assert 'final_score' in scored.columns
    assert scored['final_score'].notna().all()


def test_construct_portfolio_weights():
    df = make_sample_df()
    scored = compute_scores(df, min_market_cap=0)
    port = construct_portfolio(scored, n=3, max_weight=0.6)
    # weights should sum to ~1
    s = port['weight'].sum()
    assert abs(s - 1.0) < 1e-6
    # no weight should exceed max_weight
    assert (port['weight'] <= 0.6 + 1e-9).all()
