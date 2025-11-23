# src/reddit.py
import requests
import time
from collections import Counter
import json
from pathlib import Path
import pandas as pd
import numpy as np
import random
from cache import get as cache_get, set as cache_set
from community import load_community_signals


def _generate_fallback_scores(universe_tickers, financials_df=None):
    """
    Generiert abgestrafte Fallback-Scores für alle Ticker.
    Da diese simuliert sind (keine echten Reddit-Daten), werden sie niedrig gehalten.
    
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
    base_mean = 0.35
    base_std = 0.08  # Standardabweichung für Normalverteilung
    
    # Reduzierte Sektor-Anpassungen (nur kleine Unterschiede)
    sector_adjustments = {
        'Financial Services': 0.05,
        'Consumer Defensive': 0.04,
        'Healthcare': 0.03,
        'Utilities': 0.03,
        'Consumer Cyclical': 0.02,
        'Energy': 0.02,
        'Industrials': 0.02,
        'Real Estate': 0.02,
        'Basic Materials': 0.02,
        'Technology': 0.01,
        'Communication Services': 0.01,
    }
    
    # Reduzierte Market Cap Anpassungen
    mcap_adjustments = {
        'mega': 0.06,      # > 500B
        'large': 0.04,      # 200B-500B
        'mid_large': 0.02,  # 100B-200B
        'mid': 0.01,       # 50B-100B
        'small': 0.0,      # < 50B
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
            
            # Cap zwischen 0.15 und 0.65 (niedrige Werte für simulierte Daten)
            scores[ticker] = round(max(0.15, min(0.65, score)), 3)
    else:
        # Fallback: einfache Normalverteilung ohne Financials
        for ticker in universe_tickers:
            score = np.random.normal(base_mean, base_std)
            scores[ticker] = round(max(0.15, min(0.65, score)), 3)
    
    return scores


def get_reddit_mentions(universe_tickers, days_back=120, ttl_seconds: int = 24*3600, financials_df=None):
    """
    Lädt Reddit-Scores aus community_signals.json (echte Daten) und generiert
    simulierte Daten für fehlende Ticker.
    
    Gibt zurück: {ticker: score 0.0–1.0}
    - Echte Daten aus community_signals.json für vorhandene Ticker
    - Simulierte Daten (Normalverteilung, Mittelwert 0.35) für fehlende Ticker
    """
    # Lade echte Daten aus community_signals.json
    try:
        _, reddit_scores_from_file, _ = load_community_signals()
    except Exception as e:
        print(f"Warning: Could not load community_signals.json: {e}")
        reddit_scores_from_file = {}
    
    # Generiere simulierte Scores für alle Ticker (wird für fehlende verwendet)
    simulated_scores = _generate_fallback_scores(universe_tickers, financials_df=financials_df)
    
    # Kombiniere: Echte Daten haben Priorität, dann simulierte Daten
    result = {}
    for ticker in universe_tickers:
        if ticker in reddit_scores_from_file and reddit_scores_from_file[ticker] is not None:
            # Verwende echte Daten aus community_signals.json
            result[ticker] = float(reddit_scores_from_file[ticker])
        else:
            # Verwende simulierte Daten für fehlende Ticker
            result[ticker] = simulated_scores.get(ticker, 0.35)
    
    return result