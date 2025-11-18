# src/fetch_dataroma.py
"""Lightweight scraper for dataroma.com to detect whether a ticker
appears in recent insider/superinvestor buys lists.

Note: Respect ToS; this is a minimal demo. In production add caching, rate-limit handling
and robust parsing.
"""
import requests
from bs4 import BeautifulSoup

BASE = 'https://www.dataroma.com'


def fetch_dataroma_signal(ticker):
    """Return 1 if ticker appears among recent top holdings additions (simple heuristic), else 0."""
    try:
        # dataroma has pages listing recent buys; here we search the site for ticker pages
        # Simple approach: query "site:dataroma.com TICKER" is not available; instead try holdings page
        url = f'{BASE}/m/stock.php?s={ticker}'
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return 0
        soup = BeautifulSoup(r.text, 'lxml')
        # If the ticker page exists and shows 'Recent Picks' or holdings, treat as signal
        # This is a heuristic: look for a header or table on page
        if soup.find('table'):
            # further heuristics could inspect the table contents
            return 1
        return 0
    except Exception:
        return 0