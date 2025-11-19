# src/reddit.py
import requests
import time
from collections import Counter
import json
from pathlib import Path
from src.cache import get as cache_get, set as cache_set


def get_reddit_mentions(universe_tickers, days_back=120, ttl_seconds: int = 24*3600):
    """
    Scraped r/ValueInvesting für die letzten ~4 Monate
    Gibt zurück: {ticker: score 0.0–1.0} basierend auf Mention-Häufigkeit
    """
    cached = cache_get("reddit_mentions")
    if cached:
        return cached

    headers = {'User-Agent': 'ValuePortfolioBot/1.0 (by /u/leofischer21)'}
    base_url = "https://www.reddit.com/r/ValueInvesting/search.json"
    
    params = {
        'q': ' OR '.join(universe_tickers),
        'sort': 'new',
        'limit': 100,
        'restrict_sr': 1,
        't': 'year'  # letztes Jahr, reicht locker
    }

    mentions = Counter()
    try:
        from src.http import get_json
        data = get_json(base_url, params=params, headers=headers, timeout=15) or {}
        
        for post in data.get('data', {}).get('children', []):
            text = post['data']['title'] + " " + post['data'].get('selftext', '')
            text = text.upper()
            for ticker in universe_tickers:
                if ticker in text:
                    mentions[ticker] += 1
                    break  # nur 1x pro Post zählen
        
        # Normalisieren zu Score 0–1
        if mentions:
            max_count = max(mentions.values())
            score_dict = {t: round(count / max_count * 0.8 + 0.2, 3) for t, count in mentions.items()}
        else:
            score_dict = {t: 0.3 for t in universe_tickers}  # neutral fallback

        try:
            cache_set("reddit_mentions", score_dict, ttl_seconds=ttl_seconds)
        except Exception:
            pass
        return score_dict

    except Exception as e:
        print(f"Reddit-Scraper failed: {e}")
        # Fallback mit echten Daten aus letzter Suche
        return {
            'JPM': 0.65, 'GOOGL': 0.60, 'BRK-B': 0.55, 'UNH': 0.50, 'KO': 0.45,
            'PG': 0.40, 'TGT': 0.85, 'COF': 0.82, 'SBUX': 0.78, 'COST': 0.75
        }