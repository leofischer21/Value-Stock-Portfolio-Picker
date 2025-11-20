# src/fetch_dataroma.py
"""Lightweight scraper for dataroma.com to detect whether a ticker
appears in recent insider/superinvestor buys lists.

Note: Respect ToS; this is a minimal demo. In production add caching, rate-limit handling
and robust parsing.
"""
from bs4 import BeautifulSoup
from scripts.httpx import get_text

BASE = 'https://www.dataroma.com'


def fetch_dataroma_signal(ticker):
    """Return 1 if ticker appears among recent top holdings additions (simple heuristic), else 0."""
    try:
        url = f'{BASE}/m/stock.php?s={ticker}'
        text = get_text(url, timeout=10)
        if not text:
            return 0
        soup = BeautifulSoup(text, 'lxml')
        if soup.find('table'):
            return 1
        return 0
    except Exception:
        return 0