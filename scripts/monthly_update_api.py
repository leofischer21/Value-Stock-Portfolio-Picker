# monthly_update_api.py
"""
Monthly Update Orchestrator - API-Version
Führt monatlich alle Updates durch mit API-basierten Methoden:
1. Generiert neue Ticker-Liste (>50B MarketCap, US-Ticker)
2. Lädt Finanzdaten (Yahoo Finance)
3. Lädt Reddit-Sentiment (via Apify API)
4. Lädt X-Sentiment (via Apify API - kaitoeasyapi/tweet-scraper)
5. Lädt Superinvestor-Daten (via SEC EDGAR API)
6. Speichert alles in monatlichen Dateien

Bei API-Fehlern wird automatisch auf die Legacy-Methoden zurückgegriffen.
"""
from datetime import datetime
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
import argparse

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def update_universe(month: str = None, force: bool = False) -> List[str]:
    """
    Lädt oder generiert Ticker-Liste über 50B MarketCap (US-Ticker).
    Identisch zu monthly_update.py - keine Änderung nötig.
    
    Args:
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, generiert neue Liste. Wenn False, lädt vorhandene wenn verfügbar.
    
    Returns: List of ticker symbols
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Ticker-Liste bereits existiert
    tickers_path = ROOT_DIR / "data" / "tickers" / f"{month}.csv"
    if not force and tickers_path.exists():
        try:
            logger.info(f"Loading existing ticker list from {tickers_path}...")
            tickers_df = pd.read_csv(tickers_path)
            tickers = tickers_df['ticker'].dropna().str.strip().tolist()
            if tickers:
                logger.info(f"Loaded {len(tickers)} tickers from existing file")
                return tickers
        except Exception as e:
            logger.warning(f"Could not load existing ticker list: {e}. Generating new one...")
    
    # Generiere neue Ticker-Liste
    try:
        logger.info("Generating new ticker list...")
        import sys
        from pathlib import Path
        # Add scripts directory to path for import
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from generate_largecaps import main as generate_universe
        
        tickers = generate_universe()
        
        if not tickers:
            raise ValueError("No tickers generated!")
        
        logger.info(f"Universe update complete: {len(tickers)} tickers")
        return tickers
        
    except Exception as e:
        logger.error(f"Failed to update universe: {e}")
        raise


def _file_exists_for_month(month: str, file_type: str) -> bool:
    """
    Prüft ob eine Datei für den Monat bereits existiert.
    """
    if file_type == "financials":
        path = ROOT_DIR / "data" / "financials" / f"{month}.csv"
    elif file_type == "scores":
        path = ROOT_DIR / "data" / "scores" / f"{month}.json"
    elif file_type == "extended_scores":
        path = ROOT_DIR / "data" / "extended_scores" / f"{month}.json"
    elif file_type == "ai_scores":
        path = ROOT_DIR / "data" / "ai_scores" / f"{month}.json"
    else:
        return False
    
    return path.exists()


def update_financials(tickers: List[str], month: str = None, force: bool = False) -> pd.DataFrame:
    """
    Lädt Finanzdaten für alle Ticker.
    Identisch zu monthly_update.py - keine Änderung nötig.
    
    Args:
        tickers: List of ticker symbols
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    
    Returns:
        DataFrame mit Finanzdaten
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "financials"):
        logger.info(f"Financial data for {month} already exists. Loading from file...")
        financials_path = ROOT_DIR / "data" / "financials" / f"{month}.csv"
        try:
            financials_df = pd.read_csv(financials_path)
            logger.info(f"Loaded financial data from {financials_path}")
            return financials_df
        except Exception as e:
            logger.warning(f"Could not load existing financials file: {e}. Recalculating...")
    
    try:
        logger.info(f"Fetching financial data for {len(tickers)} tickers...")
        
        # Import data_providers
        from data_providers import fetch_fundamentals
        
        # Fetch fundamentals in parallel
        logger.info("Fetching fundamentals from Yahoo Finance...")
        info_map = fetch_fundamentals(tickers, max_workers=8)
        
        if not info_map:
            raise ValueError("No financial data fetched!")
        
        # Convert to DataFrame
        rows = []
        for ticker, info in info_map.items():
            if info:  # Only add if we have data
                row = {
                    'ticker': ticker,
                    'marketCap': info.get('marketCap'),
                    'sector': info.get('sector', 'Unknown'),
                    'trailingPE': info.get('trailingPE'),
                    'forwardPE': info.get('forwardPE'),
                    'pegRatio': info.get('pegRatio'),
                    'priceToFreeCashFlow': info.get('priceToFreeCashFlow'),
                    'beta': info.get('beta'),
                    'returnOnEquity': info.get('returnOnEquity'),
                    'debtToEquity': info.get('debtToEquity'),
                    'returnOnInvestedCapital': info.get('returnOnInvestedCapital'),
                    'grossMargins': info.get('grossMargins'),
                    'operatingMargins': info.get('operatingMargins'),
                    'profitMargins': info.get('profitMargins'),
                    'enterpriseToEbitda': info.get('enterpriseToEbitda'),
                    'priceMomentum12M': info.get('priceMomentum12M'),
                }
                # Only add if we have at least some key data
                if row['marketCap'] is not None or row['trailingPE'] is not None:
                    rows.append(row)
        
        if not rows:
            raise ValueError("No valid financial data rows created!")
        
        financial_df = pd.DataFrame(rows)
        
        # Save to data/financials/
        financials_dir = ROOT_DIR / "data/financials"
        financials_dir.mkdir(parents=True, exist_ok=True)
        
        financials_path = financials_dir / f"{month}.csv"
        financial_df.to_csv(financials_path, index=False)
        logger.info(f"Saved financial data to {financials_path}")
        
        # Also save as latest.csv
        latest_path = financials_dir / "latest.csv"
        financial_df.to_csv(latest_path, index=False)
        
        return financial_df
        
    except Exception as e:
        logger.error(f"Failed to update financials: {e}")
        raise


def update_sentiments_api(tickers: List[str], financials_df: pd.DataFrame, month: str = None, force: bool = False) -> Dict:
    """
    Lädt Sentiment-Daten via APIs (Reddit, X, Superinvestor).
    Verwendet API-Module mit automatischem Fallback auf Legacy-Methoden.
    
    Args:
        tickers: List of ticker symbols
        financials_df: DataFrame mit Finanzdaten
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    
    Returns:
        Dict mit 'superinvestor_score', 'reddit_score', 'x_score'
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "scores"):
        logger.info(f"Sentiment scores for {month} already exist. Loading from file...")
        scores_path = ROOT_DIR / "data" / "scores" / f"{month}.json"
        try:
            with open(scores_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            logger.info(f"Loaded sentiment scores from {scores_path}")
            return scores
        except Exception as e:
            logger.warning(f"Could not load existing scores file: {e}. Recalculating...")
    
    try:
        logger.info(f"Loading sentiment data via APIs for {len(tickers)} tickers...")
        
        # Import API modules
        from reddit_api import get_reddit_mentions_api
        from twitter_api import get_x_sentiment_score_api
        from dataroma_api import get_superinvestor_data_api
        
        # Load Reddit sentiment via API
        logger.info("Loading Reddit sentiment via Apify API...")
        try:
            reddit_scores = get_reddit_mentions_api(tickers, days_back=120, financials_df=financials_df)
            if not reddit_scores:
                raise ValueError("No Reddit scores returned from API")
            logger.info(f"Reddit API: {len(reddit_scores)} tickers scored")
        except Exception as e:
            logger.warning(f"Reddit API failed: {e}, using legacy method")
            from reddit import get_reddit_mentions
            reddit_scores = get_reddit_mentions(tickers, days_back=120, financials_df=financials_df)
        
        # Load X sentiment via API
        logger.info("Loading X (Twitter) sentiment via Apify API...")
        try:
            x_scores = get_x_sentiment_score_api(tickers, financials_df=financials_df)
            if not x_scores:
                raise ValueError("No X scores returned from API")
            logger.info(f"X API: {len(x_scores)} tickers scored")
        except Exception as e:
            logger.warning(f"X API failed: {e}, using legacy method")
            from twitter import get_x_sentiment_score
            x_scores = get_x_sentiment_score(tickers, financials_df=financials_df)
        
        # Load Superinvestor data via API
        logger.info("Loading Superinvestor data via SEC EDGAR API...")
        try:
            superinvestor_scores = get_superinvestor_data_api(universe=tickers)
            if not superinvestor_scores:
                raise ValueError("No Superinvestor scores returned from API")
            logger.info(f"Superinvestor API: {len(superinvestor_scores)} tickers scored")
        except Exception as e:
            logger.warning(f"Superinvestor API failed: {e}, using legacy method")
            from dataroma import get_superinvestor_data
            superinvestor_scores = get_superinvestor_data(universe=tickers)
        
        # Combine into single dict
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "superinvestor_score": superinvestor_scores,
            "reddit_score": reddit_scores,
            "x_score": x_scores
        }
        
        # Save to data/scores/
        scores_dir = ROOT_DIR / "data/scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        
        scores_path = scores_dir / f"{month}.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved sentiment scores to {scores_path}")
        
        # Also save as latest.json
        latest_path = scores_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        logger.info("All sentiment data loaded successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to update sentiments: {e}")
        raise


def update_extended_scores(tickers: List[str], month: str = None, force: bool = False) -> None:
    """
    Lädt PE History und Analyst Scores für alle Ticker.
    Identisch zu monthly_update.py - keine Änderung nötig.
    
    Args:
        tickers: List of ticker symbols
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "extended_scores"):
        logger.info(f"Extended scores for {month} already exist. Skipping...")
        return
    
    try:
        logger.info(f"Calculating extended scores for {len(tickers)} tickers...")
        
        # Import modules
        from portfolio_calculator import fetch_pe_history, fetch_analyst_scores
        
        # Fetch PE History
        logger.info("Fetching PE vs History scores...")
        pe_scores = fetch_pe_history(tickers)
        
        # Fetch Analyst Scores
        logger.info("Fetching Analyst scores...")
        analyst_scores = fetch_analyst_scores(tickers)
        
        # Combine
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "pe_vs_history_score": pe_scores,
            "analyst_score": analyst_scores,
        }
        
        # Save to data/extended_scores/
        extended_dir = ROOT_DIR / "data/extended_scores"
        extended_dir.mkdir(parents=True, exist_ok=True)
        
        extended_path = extended_dir / f"{month}.json"
        with open(extended_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved extended scores to {extended_path}")
        
        # Also save as latest.json
        latest_path = extended_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to update extended scores: {e}")
        raise


def update_ai_scores(tickers: List[str], financials_df: pd.DataFrame, scores_dict: Dict, month: str = None, force: bool = False) -> None:
    """
    Lädt AI Scores (Moat, Quality, Predicted Performance) für alle Ticker.
    Identisch zu monthly_update.py - keine Änderung nötig.
    
    Args:
        tickers: List of ticker symbols
        financials_df: DataFrame mit Finanzdaten
        scores_dict: Dict mit Sentiment-Scores
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "ai_scores"):
        logger.info(f"AI scores for {month} already exist. Skipping...")
        return
    
    try:
        logger.info(f"Calculating AI scores for {len(tickers)} tickers...")
        
        # Import AI scores module
        from ai_scores import get_ai_moat_score, get_ai_quality_score, get_ai_predicted_performance
        
        moat_scores = {}
        quality_scores = {}
        predicted_performance = {}
        
        # Convert financials_df to dict for faster lookup
        financials_dict = financials_df.set_index('ticker').to_dict('index')
        
        # Process tickers with progress logging
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing AI scores: {i + 1}/{len(tickers)} (failed: {len(failed_tickers)})")
            
            try:
                # Get financial data for this ticker
                financial_data = financials_dict.get(ticker, {})
                
                # Get existing scores
                existing_scores = {
                    'community_score': (
                        scores_dict.get('superinvestor_score', {}).get(ticker, 0.5) * 0.333 +
                        scores_dict.get('reddit_score', {}).get(ticker, 0.5) * 0.333 +
                        scores_dict.get('x_score', {}).get(ticker, 0.5) * 0.334
                    ),
                    'quality_score': 0.5
                }
                
                # Iterate over all 3 categories per ticker
                categories = [
                    ("moat", get_ai_moat_score, moat_scores),
                    ("quality", get_ai_quality_score, quality_scores),
                    ("performance", get_ai_predicted_performance, predicted_performance)
                ]
                
                for category_name, score_func, score_dict in categories:
                    try:
                        if category_name == "performance":
                            result = score_func(ticker, financial_data, existing_scores)
                            predicted_performance[ticker] = result
                        else:
                            result = score_func(ticker, financial_data, existing_scores)
                            score_dict[ticker] = result
                    except Exception as e:
                        logger.warning(f"Failed to get {category_name} score for {ticker}: {e}")
                        if category_name == "performance":
                            predicted_performance[ticker] = {
                                'cagr_1y': 8.0,
                                'cagr_2y': 8.0,
                                'cagr_5y': 8.0,
                                'cagr_10y': 8.0
                            }
                        else:
                            score_dict[ticker] = 0.5
                
            except Exception as e:
                logger.warning(f"Error calculating AI scores for {ticker}: {e}")
                failed_tickers.append(ticker)
                moat_scores[ticker] = 0.5
                quality_scores[ticker] = 0.5
                predicted_performance[ticker] = {
                    'cagr_1y': 8.0,
                    'cagr_2y': 8.0,
                    'cagr_5y': 8.0,
                    'cagr_10y': 8.0
                }
        
        # Save to data/ai_scores/
        ai_scores_dir = ROOT_DIR / "data/ai_scores"
        ai_scores_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "moat_scores": moat_scores,
            "quality_scores": quality_scores,
            "predicted_performance": predicted_performance
        }
        
        ai_scores_path = ai_scores_dir / f"{month}.json"
        with open(ai_scores_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved AI scores to {ai_scores_path}")
        
        # Also save as latest.json
        latest_path = ai_scores_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        if failed_tickers:
            logger.warning(f"Failed to calculate AI scores for {len(failed_tickers)} tickers")
        
    except Exception as e:
        logger.error(f"Failed to update AI scores: {e}")
        raise


def main():
    """
    Hauptfunktion: Führt alle Updates durch.
    """
    parser = argparse.ArgumentParser(description="Monthly Update - API Version")
    parser.add_argument("--force", action="store_true", help="Force recalculation of all data")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Monthly Update - API Version")
    logger.info("=" * 60)
    
    month = datetime.now().strftime("%Y-%m")
    logger.info(f"Month: {month}")
    logger.info(f"Force mode: {args.force}")
    
    try:
        # 1. Update Universe
        logger.info("\n[1/5] Updating Universe...")
        tickers = update_universe(month=month, force=args.force)
        logger.info(f"Universe: {len(tickers)} tickers")
        
        # 2. Update Financials
        logger.info("\n[2/5] Updating Financials...")
        financials_df = update_financials(tickers, month=month, force=args.force)
        logger.info(f"Financials: {len(financials_df)} tickers")
        
        # 3. Update Sentiments (API-Version)
        logger.info("\n[3/5] Updating Sentiments (API)...")
        scores_dict = update_sentiments_api(tickers, financials_df, month=month, force=args.force)
        logger.info(f"Sentiments: {len(scores_dict.get('reddit_score', {}))} tickers")
        
        # 4. Update Extended Scores
        logger.info("\n[4/5] Updating Extended Scores...")
        update_extended_scores(tickers, month=month, force=args.force)
        logger.info("Extended scores updated")
        
        # 5. Update AI Scores
        logger.info("\n[5/5] Updating AI Scores...")
        update_ai_scores(tickers, financials_df, scores_dict, month=month, force=args.force)
        logger.info("AI scores updated")
        
        logger.info("\n" + "=" * 60)
        logger.info("Monthly Update - API Version completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Monthly Update failed: {e}")
        raise


if __name__ == "__main__":
    main()

