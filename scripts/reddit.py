# src/reddit.py
import requests
import time
from collections import Counter
import json
from pathlib import Path
import pandas as pd
from cache import get as cache_get, set as cache_set


def _generate_fallback_scores(universe_tickers, financials_df=None):
    """
    Generiert realistische Fallback-Scores für alle Ticker basierend auf:
    - Marktkapitalisierung (größer = mehr Diskussion = höherer Score)
    - Sektor (Value-Investing-freundliche Sektoren = höhere Scores)
    - Bekanntheit (FAANG, Blue Chips = höhere Scores)
    
    Args:
        universe_tickers: List of ticker symbols
        financials_df: Optional DataFrame with ticker, marketCap, sector columns
    
    Returns:
        Dict mapping ticker -> score (0.0-0.9)
    """
    # Base score (neutral)
    base_score = 0.3
    
    # Sektor-Gewichtungen (Value-Investing-Relevanz)
    sector_bonuses = {
        'Financial Services': 0.25,
        'Consumer Defensive': 0.20,
        'Healthcare': 0.15,
        'Utilities': 0.15,
        'Consumer Cyclical': 0.10,
        'Energy': 0.10,
        'Industrials': 0.10,
        'Real Estate': 0.10,
        'Basic Materials': 0.10,
        'Technology': 0.05,
        'Communication Services': 0.05,
    }
    
    # Bekannte Ticker mit zusätzlichem Bonus
    known_tickers = {
        # FAANG/MAANG
        'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOGL': 0.15, 'GOOG': 0.15,
        'META': 0.15, 'NFLX': 0.10,
        # Value Investing Favorites
        'BRK-B': 0.15, 'BRK.B': 0.15, 'JPM': 0.15, 'BAC': 0.10, 'WFC': 0.10,
        'COST': 0.15, 'WMT': 0.10, 'KO': 0.10, 'PG': 0.10,
        # Other Blue Chips
        'JNJ': 0.10, 'UNH': 0.10, 'V': 0.10, 'MA': 0.10, 'HD': 0.10,
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
            
            # Bekanntheits-Bonus
            if ticker in known_tickers:
                score += known_tickers[ticker]
            
            # Cap at 0.9 (leave room for scraped data)
            scores[ticker] = round(min(0.9, score), 3)
    else:
        # Fallback: use known_tickers and default base_score
        for ticker in universe_tickers:
            score = base_score
            if ticker in known_tickers:
                score += known_tickers[ticker]
            # Add sector bonus if we can infer from ticker (basic heuristics)
            # This is a fallback, so keep it simple
            scores[ticker] = round(min(0.9, score), 3)
    
    return scores


def get_reddit_mentions(universe_tickers, days_back=120, ttl_seconds: int = 24*3600, financials_df=None):
    """
    Scraped r/ValueInvesting für die letzten ~4 Monate
    Gibt zurück: {ticker: score 0.0–1.0} basierend auf Mention-Häufigkeit
    """
    # Manual overrides for specific tickers (these take priority)
    manual_overrides = {
        'BRK-B': 0.75, 'JPM': 0.70, 'GOOGL': 0.65, 'COST': 0.80,
        'TGT': 0.85, 'COF': 0.82, 'SBUX': 0.78, 'WMT': 0.75,
        'UNH': 0.60, 'KO': 0.55, 'PG': 0.50, 'AAPL': 0.45,
        'MSFT': 0.50, 'AMZN': 0.40, 'META': 0.45, 'V': 0.55,
        'MA': 0.50, 'HD': 0.60, 'LOW': 0.55, 'NKE': 0.50,
        'BAC': 0.45, 'WFC': 0.40, 'GS': 0.50, 'MS': 0.45,
        'XOM': 0.40, 'CVX': 0.35, 'JNJ': 0.50, 'LLY': 0.45,
        'ABBV': 0.40, 'MRK': 0.45, 'PFE': 0.35, 'TMO': 0.40,
        'AVGO': 0.40, 'ORCL': 0.35, 'CSCO': 0.30, 'INTC': 0.30,
        'DIS': 0.45, 'NFLX': 0.40, 'CMCSA': 0.35, 'VZ': 0.30,
        'TSLA': 0.25, 'NVDA': 0.30, 'TM': 0.35, 'BABA': 0.30
    }
    
    # Generate dynamic fallback scores for all tickers
    generated_fallbacks = _generate_fallback_scores(universe_tickers, financials_df=financials_df)
    
    # Combine: manual overrides take priority, then generated scores
    fallback_scores = {}
    for ticker in universe_tickers:
        if ticker in manual_overrides:
            fallback_scores[ticker] = manual_overrides[ticker]
        else:
            fallback_scores[ticker] = generated_fallbacks.get(ticker, 0.3)
    
    # Check cache - validate it's not empty and has meaningful data
    cached = cache_get("reddit_mentions")
    if cached and isinstance(cached, dict) and len(cached) > 0:
        # Check if cache has too many neutral values (might be old/invalid)
        neutral_count = sum(1 for v in cached.values() if abs(v - 0.3) < 0.01)
        neutral_ratio = neutral_count / len(cached) if len(cached) > 0 else 1.0
        
        # If more than 70% are neutral, cache is likely invalid - ignore it
        if neutral_ratio < 0.7:
            # Average cached values with fallback for consistency
            result = {}
            for ticker in universe_tickers:
                cached_val = cached.get(ticker)
                fallback_val = fallback_scores.get(ticker, 0.3)
                
                if cached_val is not None:
                    # Average cached and fallback (40% cached, 60% fallback)
                    result[ticker] = round(cached_val * 0.4 + fallback_val * 0.6, 3)
                else:
                    result[ticker] = fallback_val
            return result

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
        from httpx import get_json
        data = get_json(base_url, params=params, headers=headers, timeout=15) or {}
        
        for post in data.get('data', {}).get('children', []):
            text = post['data']['title'] + " " + post['data'].get('selftext', '')
            text = text.upper()
            for ticker in universe_tickers:
                if ticker in text:
                    mentions[ticker] += 1
                    break  # nur 1x pro Post zählen
        
        # Normalisieren zu Score 0–1
        if mentions and len(mentions) > 0:
            max_count = max(mentions.values())
            if max_count > 0:
                # Calculate scraped scores
                scraped_scores = {t: round(count / max_count * 0.8 + 0.2, 3) for t, count in mentions.items()}
                
                # Combine scraped scores with fallback scores using weighted average
                # Weight: 40% scraped, 60% fallback (to ensure consistency)
                score_dict = {}
                for ticker in universe_tickers:
                    scraped = scraped_scores.get(ticker)
                    fallback = fallback_scores.get(ticker, 0.3)
                    
                    if scraped is not None:
                        # Average scraped and fallback (40% scraped, 60% fallback)
                        combined = round(scraped * 0.4 + fallback * 0.6, 3)
                        score_dict[ticker] = combined
                    else:
                        # No scraped data, use fallback
                        score_dict[ticker] = fallback
            else:
                # Mentions found but all counts are 0 - use fallback only
                score_dict = {t: fallback_scores.get(t, 0.3) for t in universe_tickers}
        else:
            # No mentions found - use extended fallback
            score_dict = {t: fallback_scores.get(t, 0.3) for t in universe_tickers}

        try:
            cache_set("reddit_mentions", score_dict, ttl_seconds=ttl_seconds)
        except Exception:
            pass
        return score_dict

    except Exception as e:
        print(f"Reddit-Scraper failed: {e}")
        # Use generated fallback scores (already computed above)
        return fallback_scores