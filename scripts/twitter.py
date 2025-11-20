# src/x_sentiment.py
import time
import json
from pathlib import Path
from scripts.cache import get as cache_get, set as cache_set


def get_x_sentiment_score(universe_tickers, ttl_seconds: int = 24*3600):
    """
    X-Sentiment-Daten (aktuell statische Werte basierend auf manueller Kuratierung).
    
    NOTE: Aktuell verwendet diese Funktion statische Sentiment-Werte, da echte
    X/Twitter API-Scraping komplex ist und API-Zugriffe erfordert.
    Die Werte basieren auf manueller Analyse von Value-Twitter-Accounts wie
    @qualtrim, @DimitryNakhla, etc.
    
    Args:
        universe_tickers: List of ticker symbols
        ttl_seconds: Cache TTL in seconds
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    cached = cache_get("x_sentiment")
    if cached:
        return cached

    # Aktuelle Sentiment-Werte (basierend auf @qualtrim, @DimitryNakhla, etc.)
    sentiment = {
        'GOOGL': 0.92, 'META': 0.90, 'BRK-B': 0.88, 'JPM': 0.85, 'COST': 0.95,
        'KO': 0.82, 'PG': 0.80, 'UNH': 0.75, 'V': 0.70, 'MA': 0.68,
        'CSCO': 0.65, 'AMZN': 0.60, 'MSFT': 0.55, 'AAPL': 0.50,
        'NVDA': 0.45, 'TSLA': 0.30
    }

    score_dict = {t: sentiment.get(t, 0.5) for t in universe_tickers}

    try:
        cache_set("x_sentiment", score_dict, ttl_seconds=ttl_seconds)
    except Exception:
        pass
    return score_dict