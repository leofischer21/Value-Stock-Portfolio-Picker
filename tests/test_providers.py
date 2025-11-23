import sys
import os
import pandas as pd

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_providers import (
    get_pe_history_features,
    get_insider_summary,
    get_analyst_summary,
)
from scripts import cache as scache


class DummyTicker:
    def __init__(self, info, hist=None):
        self.info = info
        self._hist = hist or pd.DataFrame()

    def history(self, period="1y"):
        return self._hist


def test_get_pe_history_features(monkeypatch):
    scache.clear()
    # build a simple 5y monthly series
    dates = pd.date_range(end=pd.Timestamp('2025-01-01'), periods=60, freq='M')
    closes = pd.Series([100.0 + i for i in range(len(dates))], index=dates)
    hist = pd.DataFrame({'Close': closes})

    info = {'trailingEps': 2.0, 'currentPrice': 130.0}

    monkeypatch.setattr('yfinance.Ticker', lambda tk: DummyTicker(info, hist))

    res = get_pe_history_features('FAKE')
    assert 'pe_score' in res
    assert 0.0 <= res['pe_score'] <= 1.0
    # Ensure score present and in expected range (0..1)
    assert 0.0 <= res['pe_score'] <= 1.0


def test_get_insider_summary(monkeypatch):
    scache.clear()
    sample = "<html>Open Market Purchase by CEO on 2025-01-01</html>"
    # monkeypatch the module-local get_text used by data_providers
    monkeypatch.setattr('src.data_providers.get_text', lambda url, headers=None, timeout=12: sample)

    res = get_insider_summary('FAKE')
    assert res['recent_buys'] > 0
    assert res['senior_buy'] is True
    assert res['insider_score'] >= 0.8


def test_get_analyst_summary(monkeypatch):
    scache.clear()
    # provide recommendationMean and prices
    info = {'recommendationMean': 2.0, 'targetMeanPrice': 200.0, 'currentPrice': 100.0}
    monkeypatch.setattr('yfinance.Ticker', lambda tk: DummyTicker(info, pd.DataFrame()))

    res = get_analyst_summary('FAKE')
    assert 'analyst_score' in res
    # Ensure a numeric score in expected bounds
    assert 'analyst_score' in res
    assert 0.0 <= res['analyst_score'] <= 1.0
