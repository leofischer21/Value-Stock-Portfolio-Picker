# reddit_api.py
"""
Reddit-Sentiment via Apify API mit Groq-basierter Sentiment-Analyse.
Bei Fehlern wird automatisch auf reddit.py zurückgegriffen.
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

# API-Konfiguration
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")

# Subreddit-Gewichtungen (wie im Plan spezifiziert)
SUBREDDIT_WEIGHTS = {
    'ValueInvesting': 3.0,      # Am stärksten
    'stocks': 1.5,
    'investing': 1.5,
    'stockmarket': 1.0,
    'wallstreetbets': 0.3,      # Niedrig, da Ansichten nicht bevorzugt
}

# Top-Subs only für Qualität (reduziert Kosten)
TOP_SUBREDDITS = ['ValueInvesting', 'stocks', 'investing']  # Nur die wichtigsten

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


def _analyze_sentiment_groq(posts: List[Dict], ticker: str) -> float:
    """
    Analysiert Reddit-Posts mit Groq und gibt einen Sentiment-Score zurück.
    Berücksichtigt speziell 'undervalued' (+0.3) und 'overvalued' (-0.3).
    
    Args:
        posts: List of dicts with 'title', 'text', 'upvotes', 'created_utc'
        ticker: Ticker symbol für Kontext
    
    Returns:
        Sentiment score (0.0-1.0)
    """
    if not posts or not GROQ_API_KEY:
        return 0.5  # Neutral wenn keine Posts oder kein API-Key
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        # Kombiniere Posts zu einem Text (erste 5000 Zeichen)
        combined_text = "\n\n".join([
            f"Title: {p.get('title', '')}\nText: {p.get('text', '')[:500]}"
            for p in posts[:20]  # Max 20 Posts
        ])[:5000]
        
        prompt = f"""Analyze these Reddit posts about stock ${ticker}. Return a JSON with a single 'sentiment_score' between 0.0 (very bearish) and 1.0 (very bullish).

**CRITICAL**: The keyword 'undervalued' is a VERY STRONG positive signal (add +0.3 to score). The keyword 'overvalued' is a VERY STRONG negative signal (subtract -0.3 from score).

Also consider: positive keywords (buy, bullish, value, cheap, attractive), negative keywords (sell, bearish, crash, expensive, avoid), and overall tone.

Posts:
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


def _fetch_reddit_posts_apify(ticker: str, days_back: int = 120) -> Dict[str, List[Dict]]:
    """
    Holt Reddit-Posts via Apify API für einen Ticker.
    Verwendet comchat/reddit-api-scraper Actor.
    
    Args:
        ticker: Ticker symbol
        days_back: Anzahl Tage zurück (default: 120 = ~4 Monate)
    
    Returns:
        Dict mapping subreddit -> list of posts
    """
    if not APIFY_API_TOKEN:
        raise ValueError("APIFY_API_TOKEN not set in .env")
    
    try:
        from apify_client import ApifyClient
        import time
        
        client = ApifyClient(APIFY_API_TOKEN)
        
        # Berechne Start-Datum (Unix Timestamp)
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        all_posts = {subreddit: [] for subreddit in TOP_SUBREDDITS}
        
        # Berechne Start- und End-Datum für Actor
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Value-Investor-Keywords mit Ticker kombinieren (Qualität statt Quantität)
        # Kombiniere Ticker mit Value-Keywords für bessere, relevantere Ergebnisse
        search_terms = []
        for keyword in VALUE_KEYWORDS[:8]:  # Top 8 Keywords (undervalued, quality, moat, etc.)
            search_terms.append(f"{ticker} {keyword}")
            search_terms.append(f"${ticker} {keyword}")
        # Füge auch reine Ticker-Suche hinzu (aber weniger wichtig)
        search_terms.extend([ticker, f"${ticker}"])
        
        # Suche in Top-Subreddits nur (effizienter, bessere Qualität)
        try:
            # comchat/reddit-api-scraper Actor - Optimiert für Qualität
            run_input = {
                "searchList": search_terms,  # Value-Investor-Keywords mit Ticker
                "subRedditList": TOP_SUBREDDITS,  # Nur Top-Subs (reduziert Kosten)
                "startDate": start_date,
                "endDate": end_date,
                "resultsLimit": 100,  # Hole mehr, filtere dann auf Top 20 (Actor unterstützt min_score nicht direkt)
                "over18": True,
                "proxy": {"useApifyProxy": True}
            }
            
            # Starte Actor Run
            run = client.actor("comchat/reddit-api-scraper").call(run_input=run_input)
            
            # Hole Ergebnisse aus defaultDatasetId (korrekt)
            dataset_id = run["defaultDatasetId"]
            items = list(client.dataset(dataset_id).iterate_items())
            
            # Parse Posts mit korrekten Feldnamen
            for item in items:
                # Korrekte Feldnamen für comchat/reddit-api-scraper
                title = item.get("title", "")
                body = item.get("selftext", "") or item.get("body", "") or item.get("selftext_html", "")
                # Score kann verschiedene Namen haben
                score = item.get("score") or item.get("ups") or item.get("upvote_ratio", 0) * 100 if item.get("upvote_ratio") else 0
                if not isinstance(score, (int, float)):
                    score = 0
                # created_utc kann verschiedene Namen haben
                created_utc = item.get("created_utc") or item.get("created") or item.get("created_at")
                url = item.get("url", "") or item.get("permalink", "")
                subreddit_name = item.get("subreddit", "").lower() or item.get("subreddit_name_prefixed", "").replace("r/", "").lower()
                
                # Kombiniere title und body zu text
                text = f"{title} {body}".strip()
                
                if not text:
                    continue
                
                # QUALITÄTSFILTER: Da wir bereits nach Value-Keywords suchen, ist die Qualität gewährleistet
                # Wir limitieren später auf 20 Posts, also keine zusätzliche Filterung nötig
                
                # Konvertiere created_utc zu Unix Timestamp (optional, da bereits durch startDate/endDate gefiltert)
                post_timestamp = None
                if created_utc:
                    if isinstance(created_utc, (int, float)):
                        post_timestamp = int(created_utc)
                    elif isinstance(created_utc, str):
                        try:
                            # Versuche Unix Timestamp direkt
                            post_timestamp = int(float(created_utc))
                        except:
                            try:
                                # Versuche ISO-Format
                                post_dt = datetime.fromisoformat(created_utc.replace('Z', '+00:00'))
                                post_timestamp = int(post_dt.timestamp())
                            except:
                                pass  # Wenn Parsing fehlschlägt, verwende Post trotzdem (wurde bereits gefiltert)
                
                # Datumsfilterung: Wenn created_utc vorhanden, prüfe es; sonst vertraue auf startDate/endDate Filter
                if post_timestamp and post_timestamp < start_timestamp:
                    continue
                # Wenn kein post_timestamp, verwende Post trotzdem (wurde bereits durch startDate/endDate gefiltert)
                
                # Finde passendes Subreddit (normalisiere Namen)
                matched_subreddit = None
                for subreddit in TOP_SUBREDDITS:
                    if subreddit.lower() == subreddit_name:
                        matched_subreddit = subreddit
                        break
                
                # Falls kein Match, überspringe (nur Top-Subs für Qualität)
                if not matched_subreddit:
                    continue
                
                post = {
                    'title': title,
                    'text': text,
                    'upvotes': score,
                    'created_utc': post_timestamp or int(datetime.now().timestamp()),  # Fallback auf jetzt
                    'subreddit': matched_subreddit,
                    'url': url,
                }
                all_posts[matched_subreddit].append(post)
            
            # Limitiere auf 20 Posts insgesamt (Qualität vor Quantität)
            # Verteile gleichmäßig auf Subreddits
            total_posts = sum(len(p) for p in all_posts.values())
            if total_posts > 20:
                # Reduziere auf Top 20, gleichmäßig verteilt
                posts_per_sub = max(1, 20 // len(TOP_SUBREDDITS))
                for subreddit in TOP_SUBREDDITS:
                    if len(all_posts[subreddit]) > posts_per_sub:
                        all_posts[subreddit] = all_posts[subreddit][:posts_per_sub]
            
            logger.debug(f"Reddit API: {sum(len(p) for p in all_posts.values())} Posts gefunden für {ticker} (limit: 20)")
            
        except Exception as e:
            logger.debug(f"Apify query failed for {ticker}: {e}")
            # Bei Fehler: leere Dict zurückgeben, Fallback wird verwendet
        
        return all_posts
        
    except ImportError:
        raise ImportError("apify-client not installed. Install with: pip install apify-client")
    except Exception as e:
        logger.error(f"Apify API call failed for {ticker}: {e}")
        raise


def get_reddit_mentions_api(tickers: List[str], days_back: int = 120, financials_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Hauptfunktion: Holt Reddit-Sentiment via Apify API für alle Ticker.
    Bei Fehlern wird auf reddit.py zurückgegriffen.
    
    Args:
        tickers: List of ticker symbols
        days_back: Anzahl Tage zurück (default: 120 = ~4 Monate)
        financials_df: Optional DataFrame für Fallback
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    if not APIFY_API_TOKEN:
        logger.warning("APIFY_API_TOKEN not set, falling back to legacy method")
        return _fallback_to_legacy(tickers, days_back, financials_df)
    
    results = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"Processing Reddit API: {i + 1}/{len(tickers)}")
            
            # Hole Posts für alle Subreddits
            posts_by_subreddit = _fetch_reddit_posts_apify(ticker, days_back)
            
            # Berechne gewichteten Score pro Subreddit
            subreddit_scores = {}
            total_weight = 0.0
            
            for subreddit, posts in posts_by_subreddit.items():
                if posts:
                    # Analysiere Sentiment für diese Subreddit-Posts
                    sentiment = _analyze_sentiment_groq(posts, ticker)
                    # Verwende Gewichtung aus SUBREDDIT_WEIGHTS (auch wenn Subreddit nicht in TOP_SUBREDDITS)
                    weight = SUBREDDIT_WEIGHTS.get(subreddit, 1.0)
                    
                    subreddit_scores[subreddit] = sentiment
                    total_weight += weight
            
            # Berechne gewichteten Durchschnitt
            if total_weight > 0:
                weighted_sum = sum(subreddit_scores[subreddit] * SUBREDDIT_WEIGHTS[subreddit] 
                                 for subreddit in subreddit_scores.keys())
                final_score = weighted_sum / total_weight
            else:
                # Keine Posts gefunden -> neutral
                final_score = 0.5
            
            results[ticker] = max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.warning(f"Reddit API failed for {ticker}: {e}, will use fallback")
            failed_tickers.append(ticker)
    
    # Für fehlgeschlagene Ticker: Fallback auf Legacy
    if failed_tickers:
        logger.info(f"Falling back to legacy method for {len(failed_tickers)} tickers")
        legacy_scores = _fallback_to_legacy(failed_tickers, days_back, financials_df)
        results.update(legacy_scores)
    
    # Für Ticker ohne Ergebnisse: Fallback
    missing_tickers = [t for t in tickers if t not in results]
    if missing_tickers:
        logger.info(f"Using legacy method for {len(missing_tickers)} tickers without API results")
        legacy_scores = _fallback_to_legacy(missing_tickers, days_back, financials_df)
        results.update(legacy_scores)
    
    return results


def _fallback_to_legacy(tickers: List[str], days_back: int, financials_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Fallback: Verwendet die Legacy-Methode aus reddit.py
    """
    try:
        from reddit import get_reddit_mentions
        logger.info("Using legacy Reddit method (reddit.py)")
        return get_reddit_mentions(tickers, days_back=days_back, financials_df=financials_df)
    except Exception as e:
        logger.error(f"Legacy fallback also failed: {e}")
        # Letzter Fallback: Neutrale Scores
        return {ticker: 0.5 for ticker in tickers}

