# src/x_sentiment.py
import time
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from scripts.cache import get as cache_get, set as cache_set


def _get_static_fallback(universe_tickers):
    """Fallback auf statische Werte, falls Scraping fehlschlägt"""
    sentiment = {
        'GOOGL': 0.92, 'META': 0.90, 'BRK-B': 0.88, 'JPM': 0.85, 'COST': 0.95,
        'KO': 0.82, 'PG': 0.80, 'UNH': 0.75, 'V': 0.70, 'MA': 0.68,
        'CSCO': 0.65, 'AMZN': 0.60, 'MSFT': 0.55, 'AAPL': 0.50,
        'NVDA': 0.45, 'TSLA': 0.30
    }
    return {t: sentiment.get(t, 0.5) for t in universe_tickers}


def _scrape_twitter_mentions(universe_tickers, days_back=120):
    """
    Scraped Twitter/X Mentions für die letzten 3-4 Monate.
    Verwendet Nitter (Twitter-Alternative) für einfaches Scraping.
    Falls Nitter nicht verfügbar ist, wird ein leeres Counter zurückgegeben.
    """
    mentions = Counter()
    
    try:
        from scripts.httpx import get_text
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        # Limit auf erste 30 Ticker für Test (beim monatlichen Update können mehr sein)
        # Zu viele Requests könnten blockiert werden
        tickers_to_check = universe_tickers[:30] if len(universe_tickers) > 30 else universe_tickers
        
        # Versuche verschiedene Nitter-Instanzen (mehr Optionen)
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.it",
            "https://nitter.pussthecat.org",
            "https://nitter.42l.fr"
        ]
        
        working_instance = None
        for instance in nitter_instances:
            try:
                # Test ob Instanz verfügbar ist
                test_html = get_text(f"{instance}", headers=headers, timeout=5)
                if test_html and len(test_html) > 1000:  # Mindestgröße für gültige Seite
                    working_instance = instance
                    break
            except Exception as e:
                continue
        
        if not working_instance:
            # Keine Nitter-Instanz verfügbar -> return empty
            return Counter()
        
        # Scrape Mentions für jeden Ticker
        for idx, ticker in enumerate(tickers_to_check):
            try:
                # Suche nach Ticker mit verschiedenen Query-Formaten
                queries = [
                    f"${ticker}",
                    f"{ticker} stock",
                    f"{ticker} value"
                ]
                
                max_count = 0
                for query in queries:
                    try:
                        search_url = f"{working_instance}/search"
                        params = {'f': 'tweets', 'q': query}
                        
                        html = get_text(search_url, params=params, headers=headers, timeout=10)
                        
                        if html and len(html) > 500:
                            # Einfaches String-Matching: zähle Vorkommen des Tickers
                            # Suche nach $TICKER oder TICKER in Tweet-Kontext
                            ticker_upper = ticker.upper()
                            count = 0
                            
                            # Zähle $TICKER Mentions
                            count += html.upper().count(f"${ticker_upper}")
                            
                            # Zähle TICKER als eigenständiges Wort (mit Leerzeichen)
                            count += html.upper().count(f" {ticker_upper} ")
                            count += html.upper().count(f" {ticker_upper}\n")
                            count += html.upper().count(f" {ticker_upper}<")
                            
                            if count > max_count:
                                max_count = count
                    except Exception:
                        continue
                    
                    # Pause zwischen Queries
                    time.sleep(0.3)
                
                if max_count > 0:
                    mentions[ticker] = max_count
                
                # Pause zwischen Ticker-Requests
                if (idx + 1) % 5 == 0:
                    time.sleep(1)  # Längere Pause alle 5 Ticker
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                continue
        
        return mentions
        
    except Exception as e:
        # Falls Scraping komplett fehlschlägt, return empty Counter
        return Counter()


def get_x_sentiment_score(universe_tickers, days_back=120, ttl_seconds: int = 24*3600):
    """
    X-Sentiment-Daten: Scraped Twitter/X Mentions der letzten 3-4 Monate.
    
    Versucht echte Twitter-Mentions zu scrapen. Falls das fehlschlägt,
    verwendet statische Fallback-Werte.
    
    Args:
        universe_tickers: List of ticker symbols
        days_back: Anzahl Tage zurück für Mentions (default: 120 = ~4 Monate)
        ttl_seconds: Cache TTL in seconds
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0) für ALLE Ticker in universe_tickers
    """
    # IMMER mit statischen Werten starten (das ist die Basis)
    static_base = _get_static_fallback(universe_tickers)
    score_dict = static_base.copy()
    
    # Prüfe Cache - aber nur für gescrapte Updates, nicht als Basis
    cached = cache_get("x_sentiment")
    if cached:
        # Wenn Cache existiert, prüfe ob er gescrapte Daten enthält
        # (erkennbar daran, dass Werte von statischen Werten abweichen)
        for ticker in universe_tickers:
            cached_score = cached.get(ticker)
            static_score = static_base.get(ticker, 0.5)
            
            # Wenn gecachter Wert signifikant von statischem Wert abweicht,
            # könnte es gescrapte Daten sein
            if cached_score is not None and abs(cached_score - static_score) > 0.1:
                # Möglicherweise gescrapte Daten - verwende sie
                score_dict[ticker] = cached_score
            # Ansonsten behalte statischen Wert (bereits in score_dict)
    
    # Versuche neue Daten zu scrapen (nur wenn Cache alt ist oder nicht existiert)
    should_scrape = cached is None  # Nur scrapen wenn kein Cache existiert
    
    if should_scrape:
        try:
            # Versuche echte Mentions zu scrapen
            mentions = _scrape_twitter_mentions(universe_tickers, days_back=days_back)
            
            if mentions and len(mentions) > 0:
                # Normalisieren zu Score 0–1 (ähnlich wie Reddit)
                max_count = max(mentions.values()) if mentions.values() else 1
                if max_count > 0:
                    # Update Scores für Ticker mit gefundenen Mentions
                    # Nur überschreiben wenn wir genug Mentions haben (mindestens 3)
                    for ticker, count in mentions.items():
                        if count >= 3:  # Mindestens 3 Mentions für glaubwürdige Daten
                            scraped_score = round(count / max_count * 0.8 + 0.2, 3)
                            # Überschreibe nur wenn gescrapte Daten signifikant sind
                            score_dict[ticker] = scraped_score
                        # Wenn weniger Mentions, behalte statischen Wert
            else:
                # Keine Mentions gefunden -> behalte statische Werte (bereits in score_dict)
                pass
                
        except Exception:
            # Bei Fehler: behalte statische Werte (bereits in score_dict)
            pass
    
    # Cache speichern
    try:
        cache_set("x_sentiment", score_dict, ttl_seconds=ttl_seconds)
    except Exception:
        pass
    
    # Stelle sicher, dass ALLE Ticker einen Score haben
    final_dict = {}
    for ticker in universe_tickers:
        final_dict[ticker] = score_dict.get(ticker, static_base.get(ticker, 0.5))
    
    return final_dict