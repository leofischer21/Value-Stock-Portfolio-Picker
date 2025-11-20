import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from scripts.run_picker_2 import gather, compute_scores


class DummyCommunity:
    @staticmethod
    def load_community_signals():
        # superinvestor, reddit, x
        return ({'AAA': 0.8, 'BBB': 0.6}, {'AAA': 0.5, 'BBB': 0.4}, {'AAA': 0.6, 'BBB': 0.3})


@pytest.fixture(autouse=True)
def patch_community(monkeypatch):
    import scripts.community as community
    monkeypatch.setattr(community, 'load_community_signals', DummyCommunity.load_community_signals)
    monkeypatch.setattr(community, 'load_ai_moat', lambda: {'AAA': 0.2, 'BBB': 0.1})
    yield


@pytest.fixture
def stub_fetch_fundamentals(monkeypatch):
    def _fake(universe):
        return {
            'AAA': {'marketCap': 1e9, 'sector': 'Tech', 'trailingPE': 10, 'forwardPE': 9, 'beta': 1.0, 'returnOnEquity': 0.2, 'debtToEquity': 10, 'recommendationMean': 3.0},
            'BBB': {'marketCap': 2e9, 'sector': 'Finance', 'trailingPE': 20, 'forwardPE': 18, 'beta': 0.8, 'returnOnEquity': 0.1, 'debtToEquity': 5, 'recommendationMean': 2.5},
        }
    import scripts.data_providers as dp
    monkeypatch.setattr(dp, 'fetch_fundamentals', _fake)
    yield


def test_gather_and_compute(stub_fetch_fundamentals):
    df = gather(['AAA','BBB'])
    assert not df.empty
    scored = compute_scores(df, min_market_cap=0)
    assert 'final_score' in scored.columns
    assert scored.shape[0] == 2
