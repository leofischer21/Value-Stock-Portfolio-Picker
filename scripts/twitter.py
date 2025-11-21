# src/x_sentiment.py
import time
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
from cache import get as cache_get, set as cache_set


def _generate_fallback_scores(universe_tickers, financials_df=None):
    """
    Generiert realistische Fallback-Scores für X/Twitter basierend auf:
    - Marktkapitalisierung (größer = mehr Diskussion = höherer Score)
    - Sektor (Technology bekommt höheren Bonus als bei Reddit)
    - Bekanntheit (FAANG, Blue Chips = höhere Scores)
    - Growth vs. Value (Growth-Stocks bekommen niedrigere Scores)
    
    Args:
        universe_tickers: List of ticker symbols
        financials_df: Optional DataFrame with ticker, marketCap, sector columns
    
    Returns:
        Dict mapping ticker -> score (0.0-0.9)
    """
    # Base score (neutral, höher als Reddit da X/Twitter mehr Tech-fokussiert ist)
    base_score = 0.5
    
    # Sektor-Gewichtungen (X/Twitter Präferenzen - Technology höher als Reddit)
    sector_bonuses = {
        'Technology': 0.15,  # Höher als Reddit (0.05)
        'Financial Services': 0.20,
        'Consumer Defensive': 0.18,
        'Healthcare': 0.15,
        'Communication Services': 0.12,  # Höher als Reddit (0.05)
        'Consumer Cyclical': 0.10,
        'Energy': 0.10,
        'Industrials': 0.10,
        'Utilities': 0.12,
        'Real Estate': 0.10,
        'Basic Materials': 0.10,
    }
    
    # Growth-Stocks mit niedrigeren Scores (Value-Investing weniger relevant)
    growth_stocks = {
        'NVDA': -0.15, 'TSLA': -0.20, 'AMD': -0.10, 'PLTR': -0.10,
        'NFLX': -0.05, 'META': 0.0,  # META ist bekannt, aber Growth
    }
    
    # Bekannte Ticker mit zusätzlichem Bonus
    known_tickers = {
        # FAANG/MAANG
        'AAPL': 0.10, 'MSFT': 0.10, 'AMZN': 0.10, 'GOOGL': 0.15, 'GOOG': 0.15,
        'META': 0.10, 'NFLX': 0.05,
        # Value Investing Favorites
        'BRK-B': 0.15, 'BRK.B': 0.15, 'JPM': 0.15, 'BAC': 0.10, 'WFC': 0.10,
        'COST': 0.15, 'WMT': 0.12, 'KO': 0.12, 'PG': 0.12,
        # Other Blue Chips
        'JNJ': 0.12, 'UNH': 0.12, 'V': 0.12, 'MA': 0.12, 'HD': 0.10,
    }
    
    scores = {}
    
    # If financials_df is available, use it for dynamic calculation
    if financials_df is not None and not financials_df.empty:
        # Create lookup dictionaries
        ticker_to_mcap = financials_df.set_index('ticker')['marketCap'].to_dict()
        ticker_to_sector = financials_df.set_index('ticker')['sector'].to_dict()
        
        for ticker in universe_tickers:
            score = base_score
            
            # Market Cap Bonus
            mcap = ticker_to_mcap.get(ticker, 0)
            if mcap and mcap > 0:
                if mcap > 500_000_000_000:  # > 500B
                    score += 0.4
                elif mcap > 200_000_000_000:  # 200B-500B
                    score += 0.3
                elif mcap > 100_000_000_000:  # 100B-200B
                    score += 0.2
                elif mcap > 50_000_000_000:  # 50B-100B
                    score += 0.1
            
            # Sektor Bonus
            sector = ticker_to_sector.get(ticker, 'Unknown')
            if sector and sector in sector_bonuses:
                score += sector_bonuses[sector]
            
            # Growth-Stock Penalty
            if ticker in growth_stocks:
                score += growth_stocks[ticker]
            
            # Bekanntheits-Bonus
            if ticker in known_tickers:
                score += known_tickers[ticker]
            
            # Cap at 0.9 (leave room for scraped data)
            scores[ticker] = round(max(0.2, min(0.9, score)), 3)
    else:
        # Fallback: use known_tickers and default base_score
        for ticker in universe_tickers:
            score = base_score
            if ticker in known_tickers:
                score += known_tickers[ticker]
            if ticker in growth_stocks:
                score += growth_stocks[ticker]
            scores[ticker] = round(max(0.2, min(0.9, score)), 3)
    
    return scores


def _get_static_fallback(universe_tickers, financials_df=None):
    """Fallback auf statische Werte, falls Scraping fehlschlägt"""
    # Manual overrides for specific tickers (these take priority)
    manual_overrides = {
        # Very high sentiment (frequently discussed by value investors on X)
        'COST': 0.95, 'GOOGL': 0.92, 'META': 0.90, 'BRK-B': 0.88,
        'JPM': 0.85, 'KO': 0.82, 'PG': 0.80, 'UNH': 0.75,
        # High sentiment
        'V': 0.70, 'MA': 0.68, 'WMT': 0.75, 'HD': 0.72,
        'TGT': 0.78, 'SBUX': 0.70, 'NKE': 0.65, 'LOW': 0.68,
        # Medium-high sentiment
        'CSCO': 0.65, 'AMZN': 0.60, 'MSFT': 0.55, 'AAPL': 0.50,
        'BAC': 0.65, 'WFC': 0.60, 'GS': 0.68, 'MS': 0.65,
        'JNJ': 0.70, 'LLY': 0.65, 'ABBV': 0.60, 'MRK': 0.65,
        'PFE': 0.55, 'TMO': 0.60, 'AVGO': 0.58, 'ORCL': 0.55,
        # Medium sentiment
        'NVDA': 0.45, 'TSLA': 0.30, 'XOM': 0.50, 'CVX': 0.45,
        'DIS': 0.55, 'NFLX': 0.50, 'CMCSA': 0.45, 'VZ': 0.40,
        'TM': 0.50, 'BABA': 0.45, 'ASML': 0.60, 'TSM': 0.65,
        'INTC': 0.40, 'AMD': 0.35
    }
    
    # Generate dynamic fallback scores for all tickers
    generated_fallbacks = _generate_fallback_scores(universe_tickers, financials_df=financials_df)
    
    # Combine: manual overrides take priority, then generated scores
    result = {}
    for ticker in universe_tickers:
        if ticker in manual_overrides:
            result[ticker] = manual_overrides[ticker]
        else:
            result[ticker] = generated_fallbacks.get(ticker, 0.5)
    
    return result


def _scrape_twitter_mentions(universe_tickers, days_back=120):
    """
    Scraped Twitter/X Mentions für die letzten 3-4 Monate.
    Verwendet Nitter (Twitter-Alternative) für einfaches Scraping.
    Falls Nitter nicht verfügbar ist, wird ein leeres Counter zurückgegeben.
    """
    mentions = Counter()
    
    try:
        from httpx import get_text
        
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


def get_x_sentiment_score(universe_tickers, days_back=120, ttl_seconds: int = 24*3600, financials_df=None):
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
    static_base = _get_static_fallback(universe_tickers, financials_df=financials_df)
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