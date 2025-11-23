# src/x_sentiment.py
import time
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
from cache import get as cache_get, set as cache_set
from community import load_community_signals


def _generate_fallback_scores(universe_tickers, financials_df=None):
    """
    Generiert abgestrafte Fallback-Scores für X/Twitter.
    Da diese simuliert sind (keine echten Twitter-Daten), werden sie niedrig gehalten.
    
    Verwendet Normalverteilung mit Mittelwert ~0.35-0.4 und kleinen Anpassungen
    basierend auf Market Cap und Sektor.
    
    Args:
        universe_tickers: List of ticker symbols
        financials_df: Optional DataFrame with ticker, marketCap, sector columns
    
    Returns:
        Dict mapping ticker -> score (0.0-0.7, meist 0.25-0.5)
    """
    # Setze Seed für Reproduzierbarkeit (basierend auf Ticker-Liste)
    random.seed(hash(tuple(sorted(universe_tickers))) % 2**32)
    np.random.seed(hash(tuple(sorted(universe_tickers))) % 2**32)
    
    # Base Mittelwert für simulierte Daten (niedrig, da abgestraft)
    # Weiter reduziert, um niedriger als Reddit zu sein
    base_mean = 0.30  # Niedriger als Reddit (0.35), damit Mittelwert < 0.35
    base_std = 0.08  # Standardabweichung für Normalverteilung
    
    # Weitere reduzierte Sektor-Anpassungen (noch kleiner)
    sector_adjustments = {
        'Technology': 0.03,  # Reduziert
        'Financial Services': 0.03,
        'Consumer Defensive': 0.02,
        'Healthcare': 0.02,
        'Communication Services': 0.02,
        'Consumer Cyclical': 0.01,
        'Energy': 0.01,
        'Industrials': 0.01,
        'Utilities': 0.02,
        'Real Estate': 0.01,
        'Basic Materials': 0.01,
    }
    
    # Weitere reduzierte Market Cap Anpassungen
    mcap_adjustments = {
        'mega': 0.04,      # > 500B (reduziert von 0.06)
        'large': 0.03,      # 200B-500B (reduziert von 0.04)
        'mid_large': 0.02,  # 100B-200B
        'mid': 0.01,       # 50B-100B
        'small': 0.0,      # < 50B
    }
    
    # Growth-Stocks mit kleinen Penalties (nur kleine Anpassungen)
    growth_stocks = {
        'NVDA': -0.05, 'TSLA': -0.08, 'AMD': -0.03, 'PLTR': -0.03,
        'NFLX': -0.02,
    }
    
    scores = {}
    
    # If financials_df is available, use it for dynamic calculation
    if financials_df is not None and not financials_df.empty:
        # Create lookup dictionaries
        ticker_to_mcap = financials_df.set_index('ticker')['marketCap'].to_dict()
        ticker_to_sector = financials_df.set_index('ticker')['sector'].to_dict()
        
        for ticker in universe_tickers:
            # Starte mit Normalverteilung
            score = np.random.normal(base_mean, base_std)
            
            # Kleine Market Cap Anpassung
            mcap = ticker_to_mcap.get(ticker, 0)
            if mcap and mcap > 0:
                if mcap > 500_000_000_000:
                    score += mcap_adjustments['mega']
                elif mcap > 200_000_000_000:
                    score += mcap_adjustments['large']
                elif mcap > 100_000_000_000:
                    score += mcap_adjustments['mid_large']
                elif mcap > 50_000_000_000:
                    score += mcap_adjustments['mid']
                else:
                    score += mcap_adjustments['small']
            
            # Kleine Sektor-Anpassung
            sector = ticker_to_sector.get(ticker, 'Unknown')
            if sector and sector in sector_adjustments:
                score += sector_adjustments[sector]
            
            # Kleine Growth-Stock Penalty
            if ticker in growth_stocks:
                score += growth_stocks[ticker]
            
            # Cap zwischen 0.15 und 0.60 (niedrige Werte für simulierte Daten, reduziert von 0.65)
            scores[ticker] = round(max(0.15, min(0.60, score)), 3)
    else:
        # Fallback: einfache Normalverteilung ohne Financials
        for ticker in universe_tickers:
            score = np.random.normal(base_mean, base_std)
            if ticker in growth_stocks:
                score += growth_stocks[ticker]
            scores[ticker] = round(max(0.15, min(0.60, score)), 3)
    
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
            result[ticker] = generated_fallbacks.get(ticker, 0.30)  # Niedriger Default
    
    return result


def _scrape_twitter_mentions(universe_tickers, days_back=120):
    """
    Scraped Twitter/X Mentions für die letzten 3-4 Monate.
    Verwendet Nitter (Twitter-Alternative) für einfaches Scraping.
    Falls Nitter nicht verfügbar ist, wird ein leeres Counter zurückgegeben.
    """
    mentions = Counter()
    
    try:
        from http_utils import get_text
        
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
    Lädt X/Twitter-Scores aus community_signals.json (echte Daten) und generiert
    simulierte Daten für fehlende Ticker.
    
    Gibt zurück: {ticker: score 0.0–1.0}
    - Echte Daten aus community_signals.json für vorhandene Ticker
    - Simulierte Daten (Normalverteilung, Mittelwert 0.30) für fehlende Ticker
    """
    # Lade echte Daten aus community_signals.json
    try:
        _, _, x_scores_from_file = load_community_signals()
    except Exception as e:
        print(f"Warning: Could not load community_signals.json: {e}")
        x_scores_from_file = {}
    
    # Generiere simulierte Scores für alle Ticker (wird für fehlende verwendet)
    simulated_scores = _generate_fallback_scores(universe_tickers, financials_df=financials_df)
    
    # Kombiniere: Echte Daten haben Priorität, dann simulierte Daten
    result = {}
    for ticker in universe_tickers:
        if ticker in x_scores_from_file and x_scores_from_file[ticker] is not None:
            # Verwende echte Daten aus community_signals.json
            result[ticker] = float(x_scores_from_file[ticker])
        else:
            # Verwende simulierte Daten für fehlende Ticker
            result[ticker] = simulated_scores.get(ticker, 0.30)
    
    return result