# twitter_api.py
"""
X/Twitter-Sentiment via twitterapi.io mit Groq-basierter Sentiment-Analyse.
Bei Fehlern wird automatisch auf twitter.py zurückgegriffen.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

# Root-Verzeichnis bestimmen
try:
    ROOT_DIR = Path(__file__).parent.parent
except:
    ROOT_DIR = Path.cwd()

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass
except Exception:
    pass

logger = logging.getLogger(__name__)

# Import Retry-Logik
try:
    from api_utils import fetch_with_retry
except ImportError:
    # Fallback wenn api_utils nicht verfügbar
    def fetch_with_retry(func, *args, **kwargs):
        return func(*args, **kwargs)

# API-Konfiguration
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")  # Apify für Twitter
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")

# X/Twitter-Accounts, die stark gewichtet werden sollen (wie vom Benutzer spezifiziert)
PREFERRED_ACCOUNTS = [
    "compounding quality",
    "oguz o.",  # x capitalist
    "shay boloor",
    "patient investor",
    "fiscal.ai",
    "data driven investing",
    "qualtrim",
    "bourbon capital",
    "mindset for money",
    "dimitry nakhla",
    "quality equities",
    "sam badawi",
    "antonio linares",
]

# Account-Gewichtungen: Preferred Accounts bekommen höheres Gewicht
PREFERRED_ACCOUNT_WEIGHT = 3.0  # Starke Gewichtung für Preferred Accounts
DEFAULT_ACCOUNT_WEIGHT = 1.0     # Normale Gewichtung für andere Accounts

# Value-Investor-Keywords für Qualität statt Quantität
VALUE_KEYWORDS = [
    'undervalued',      # Sehr wichtig - stark gewichtet
    'quality',
    'moat',
    'value investing',
    'intrinsic value',
    'margin of safety',
    'earnings yield',
    'book value',
    'dividend yield',
    'free cash flow',
    'ROE',
    'ROIC',
    'DCF',
    'P/B ratio',
]


def _analyze_sentiment_groq(tweets: List[Dict], ticker: str) -> float:
    """
    Analysiert Tweets mit Groq und gibt einen Sentiment-Score zurück.
    Berücksichtigt speziell 'undervalued' (+0.3) und 'overvalued' (-0.3).
    
    Args:
        tweets: List of dicts with 'text', 'author', 'likes', 'retweets', 'created_at'
        ticker: Ticker symbol für Kontext
    
    Returns:
        Sentiment score (0.0-1.0)
    """
    if not tweets or not GROQ_API_KEY:
        return 0.5  # Neutral wenn keine Tweets oder kein API-Key
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        # Kombiniere Tweets zu einem Text (erste 5000 Zeichen)
        combined_text = "\n\n".join([
            f"Author: {t.get('author', 'Unknown')}\nText: {t.get('text', '')[:500]}"
            for t in tweets[:30]  # Max 30 Tweets
        ])[:5000]
        
        prompt = f"""Analyze these Tweets about stock ${ticker}. Return a JSON with a single 'sentiment_score' between 0.0 (very bearish) and 1.0 (very bullish).

**CRITICAL**: The keyword 'undervalued' is a VERY STRONG positive signal (add +0.3 to score). The keyword 'overvalued' is a VERY STRONG negative signal (subtract -0.3 from score).

Also consider: positive keywords (buy, bullish, value, cheap, attractive), negative keywords (sell, bearish, crash, expensive, avoid), and overall tone.

Tweets:
{combined_text}

Return only valid JSON: {{"sentiment_score": 0.0-1.0}}"""
        
        response = client.chat.completions.create(
            model=AI_MODEL if "llama" in AI_MODEL.lower() or "mixtral" in AI_MODEL.lower() else "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        # Entferne Markdown-Code-Blöcke falls vorhanden
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        score = float(result.get('sentiment_score', 0.5))
        
        # Clamp auf 0.0-1.0
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.warning(f"Groq sentiment analysis failed for {ticker}: {e}")
        return 0.5  # Neutral bei Fehler


def _is_preferred_account(author: str) -> bool:
    """
    Prüft ob ein Account zu den Preferred Accounts gehört (case-insensitive).
    """
    author_lower = author.lower()
    for preferred in PREFERRED_ACCOUNTS:
        if preferred.lower() in author_lower or author_lower in preferred.lower():
            return True
    return False


def _fetch_tweets_apify(ticker: str, days_back: int = 120) -> List[Dict]:
    """
    Holt Tweets via Apify API für einen Ticker.
    Verwendet kaitoeasyapi/tweet-scraper Actor.
    
    Args:
        ticker: Ticker symbol
        days_back: Anzahl Tage zurück (default: 120 = ~4 Monate)
    
    Returns:
        List of tweets with 'text', 'author', 'likes', 'retweets', 'created_at'
    """
    if not APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN not set in .env")
    
    try:
        from apify_client import ApifyClient
        import time
        
        client = ApifyClient(APIFY_API_TOKEN)
        
        # Berechne Start- und End-Datum für Actor
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        start_date = datetime.now() - timedelta(days=days_back)
        
        # Value-Investor-Keywords mit Ticker kombinieren (Qualität statt Quantität)
        # Kombiniere Ticker mit Value-Keywords für bessere, relevantere Ergebnisse
        search_terms = []
        for keyword in VALUE_KEYWORDS[:8]:  # Top 8 Keywords (undervalued, quality, moat, etc.)
            search_terms.append(f"${ticker} {keyword}")
            search_terms.append(f"{ticker} {keyword}")
        # Füge auch reine Ticker-Suche hinzu (aber weniger wichtig)
        search_terms.extend([f"${ticker}", ticker])
        
        all_tweets = []
        
        try:
            # kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest Actor - Optimiert für Qualität
            run_input = {
                "searchTerms": search_terms,  # Value-Investor-Keywords mit Ticker
                "maxTweets": 15,  # Qualität vor Quantität (statt 500)
                "fromDate": start_date_str,
                "toDate": end_date,
                "minLikes": 3,  # Mindestens 3 Likes (filtert Spam, Actor filtert dann auf Top 15)
                "proxyConfig": {"useApifyProxy": True}
            }
            
            # Starte Actor Run mit Retry-Logik
            def _call_actor():
                return client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input)
            
            run = fetch_with_retry(
                _call_actor,
                max_retries=3,
                initial_delay=2.0,
                check_result=lambda r: r is not None
            )
            
            if not run:
                logger.warning(f"Apify actor call failed for {ticker} after retries")
                return all_tweets
            
            # Warte auf Completion und hole Ergebnisse
            dataset_id = run["defaultDatasetId"]
            items = list(client.dataset(dataset_id).iterate_items())
            
            # Parse Tweets mit korrekten Feldnamen und Filterung für Qualität
            tweets_with_scores = []
            
            for item in items:
                # Korrekte Feldnamen für kaitoeasyapi/twitter-x-data-tweet-scraper
                text = item.get("full_text", "") or item.get("text", "")
                likes = item.get("favorite_count", 0) or item.get("likes", 0)
                retweets = item.get("retweet_count", 0) or item.get("retweets", 0)
                created_at = item.get("created_at") or item.get("created") or item.get("date")
                
                if not text:
                    continue
                
                # QUALITÄTSFILTER: Nur Tweets mit mindestens 3 Likes (wird dann auf Top 15 limitiert)
                if likes < 3:
                    continue
                
                # Prüfe ob Value-Keywords im Text vorkommen (Bonus für Sortierung)
                text_lower = text.lower()
                has_value_keyword = any(keyword.lower() in text_lower for keyword in VALUE_KEYWORDS[:8])  # Top 8 Keywords
                
                # Score für Sortierung: Likes + Bonus für Value-Keywords (priorisiert relevante Tweets)
                sort_score = likes + (50 if has_value_keyword else 0)  # +50 Bonus für Value-Keywords
                
                # Extrahiere Author-Info (verschiedene Formate möglich)
                author = 'Unknown'
                if isinstance(item.get('author'), dict):
                    author = item.get('author', {}).get('username') or item.get('author', {}).get('name') or item.get('author', {}).get('screen_name') or 'Unknown'
                elif isinstance(item.get('author'), str):
                    author = item.get('author')
                elif item.get('user'):
                    if isinstance(item.get('user'), dict):
                        author = item.get('user', {}).get('username') or item.get('user', {}).get('screen_name') or item.get('user', {}).get('name') or 'Unknown'
                    else:
                        author = str(item.get('user'))
                
                # Parse Datum (optional, da bereits durch fromDate/toDate gefiltert)
                tweet_date = None
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            # Versuche verschiedene Datumsformate
                            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%a %b %d %H:%M:%S %z %Y"]:
                                try:
                                    tweet_date = datetime.strptime(created_at[:19].replace('T', ' '), fmt.replace('T', ' '))
                                    break
                                except:
                                    continue
                            # Fallback: versuche ISO-Format
                            if not tweet_date:
                                try:
                                    tweet_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                except:
                                    pass
                        elif isinstance(created_at, (int, float)):
                            # Unix Timestamp
                            tweet_date = datetime.fromtimestamp(created_at)
                    except Exception as e:
                        logger.debug(f"Failed to parse date for tweet: {e}")
                
                # Filtere nach Datum (zusätzlich zu fromDate/toDate)
                if tweet_date and tweet_date < start_date:
                    continue
                
                tweet_dict = {
                    'text': text,
                    'author': author,
                    'likes': likes,
                    'retweets': retweets,
                    'created_at': created_at,
                }
                tweets_with_scores.append((tweet_dict, sort_score))
            
            # Sortiere nach Score (Likes + Value-Keyword-Bonus) und nehme nur Top 15 (Qualität vor Quantität)
            tweets_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Limitiere auf 15 Tweets (Qualität vor Quantität)
            for tweet_dict, _ in tweets_with_scores[:15]:
                all_tweets.append(tweet_dict)
            
            logger.debug(f"Twitter API: {len(all_tweets)} Tweets gefunden für {ticker}")
            
        except Exception as e:
            logger.debug(f"Apify query failed for {ticker}: {e}")
            # Bei Fehler: leere Liste zurückgeben, Fallback wird verwendet
        
        return all_tweets
        
    except ImportError:
        raise ImportError("apify-client not installed. Install with: pip install apify-client")
    except Exception as e:
        logger.error(f"Apify API call failed for {ticker}: {e}")
        raise


def get_x_sentiment_score_api(tickers: List[str], financials_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Hauptfunktion: Holt X/Twitter-Sentiment via Apify API für alle Ticker.
    Verwendet kaitoeasyapi/tweet-scraper Actor.
    Bei Fehlern wird auf twitter.py zurückgegriffen.
    
    Args:
        tickers: List of ticker symbols
        financials_df: Optional DataFrame für Fallback
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    if not APIFY_API_TOKEN:
        logger.warning("APIFY_API_TOKEN not set, falling back to legacy method")
        return _fallback_to_legacy(tickers, financials_df)
    
    results = {}
    failed_tickers = []
    total_tweets_found = 0  # Zähle gefundene Tweets für Qualitätsprüfung
    
    for i, ticker in enumerate(tickers):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"Processing X/Twitter API: {i + 1}/{len(tickers)}")
            
            # Hole Tweets via Apify
            tweets = _fetch_tweets_apify(ticker)
            
            # Prüfe ob wirklich Tweets gefunden wurden
            total_tweets_found += len(tweets) if tweets else 0
            
            if not tweets:
                # Keine Tweets gefunden → wahrscheinlich blockiert
                logger.debug(f"No tweets found for {ticker} - likely blocked, will use fallback")
                failed_tickers.append(ticker)
                continue
            
            # Trenne Preferred Accounts von anderen
            preferred_tweets = [t for t in tweets if _is_preferred_account(t.get('author', ''))]
            other_tweets = [t for t in tweets if not _is_preferred_account(t.get('author', ''))]
            
            # Analysiere Sentiment für beide Gruppen
            preferred_score = _analyze_sentiment_groq(preferred_tweets, ticker) if preferred_tweets else None
            other_score = _analyze_sentiment_groq(other_tweets, ticker) if other_tweets else None
            
            # Berechne gewichteten Score
            if preferred_score is not None and other_score is not None:
                # Beide Gruppen vorhanden: gewichteter Durchschnitt
                total_weight = PREFERRED_ACCOUNT_WEIGHT + DEFAULT_ACCOUNT_WEIGHT
                final_score = (preferred_score * PREFERRED_ACCOUNT_WEIGHT + other_score * DEFAULT_ACCOUNT_WEIGHT) / total_weight
            elif preferred_score is not None:
                # Nur Preferred Accounts
                final_score = preferred_score
            elif other_score is not None:
                # Nur andere Accounts
                final_score = other_score
            else:
                # Keine Tweets analysierbar -> neutral
                final_score = 0.5
            
            results[ticker] = max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.warning(f"X/Twitter API failed for {ticker}: {e}, will use fallback")
            failed_tickers.append(ticker)
    
    # Qualitätsprüfung: Wenn zu viele Ticker fehlgeschlagen sind (>50%), verwende Fallback für alle
    if len(failed_tickers) > len(tickers) * 0.5:
        logger.warning(f"Too many failed tickers ({len(failed_tickers)}/{len(tickers)} = {len(failed_tickers)/len(tickers)*100:.1f}%), falling back to legacy method for all")
        return _fallback_to_legacy(tickers, financials_df)
    
    # Qualitätsprüfung: Wenn zu wenige Tweets insgesamt gefunden wurden, verwende Fallback
    avg_tweets_per_ticker = total_tweets_found / len(tickers) if tickers else 0
    if avg_tweets_per_ticker < 1.0:  # Weniger als 1 Tweet pro Ticker im Durchschnitt
        logger.warning(f"Too few tweets found ({total_tweets_found} total, {avg_tweets_per_ticker:.1f} per ticker), falling back to legacy method")
        return _fallback_to_legacy(tickers, financials_df)
    
    # Für fehlgeschlagene Ticker: Fallback auf Legacy
    if failed_tickers:
        logger.info(f"Falling back to legacy method for {len(failed_tickers)} tickers")
        legacy_scores = _fallback_to_legacy(failed_tickers, financials_df)
        results.update(legacy_scores)
    
    # Für Ticker ohne Ergebnisse: Fallback
    missing_tickers = [t for t in tickers if t not in results]
    if missing_tickers:
        logger.info(f"Using legacy method for {len(missing_tickers)} tickers without API results")
        legacy_scores = _fallback_to_legacy(missing_tickers, financials_df)
        results.update(legacy_scores)
    
    return results


def _fallback_to_legacy(tickers: List[str], financials_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Fallback: Verwendet die Legacy-Methode aus twitter.py
    """
    try:
        from twitter import get_x_sentiment_score
        logger.info("Using legacy X/Twitter method (twitter.py)")
        return get_x_sentiment_score(tickers, financials_df=financials_df)
    except Exception as e:
        logger.error(f"Legacy fallback also failed: {e}")
        # Letzter Fallback: Neutrale Scores
        return {ticker: 0.5 for ticker in tickers}

