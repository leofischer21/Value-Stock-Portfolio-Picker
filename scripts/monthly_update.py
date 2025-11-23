# monthly_update.py
"""
Monthly Update Orchestrator
Führt monatlich alle Updates durch:
1. Generiert neue Ticker-Liste (>50B MarketCap, US-Ticker)
2. Lädt Finanzdaten (Yahoo Finance)
3. Lädt Reddit-Sentiment
4. Lädt X-Sentiment
5. Lädt Superinvestor-Daten
6. Speichert alles in monatlichen Dateien
"""
from datetime import datetime
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict

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
    Prüft, ob für den gegebenen Monat bereits eine Datei existiert.
    
    Args:
        month: Monat im Format "YYYY-MM"
        file_type: "financials", "extended_scores", "scores", "ai_scores"
    
    Returns:
        True wenn Datei existiert, False sonst
    """
    if file_type == "financials":
        path = ROOT_DIR / "data" / "financials" / f"{month}.csv"
    elif file_type == "extended_scores":
        path = ROOT_DIR / "data" / "extended_scores" / f"{month}.json"
    elif file_type == "scores":
        path = ROOT_DIR / "data" / "scores" / f"{month}.json"
    elif file_type == "ai_scores":
        path = ROOT_DIR / "data" / "ai_scores" / f"{month}.json"
    else:
        return False
    
    return path.exists() and path.stat().st_size > 0


def update_financials(tickers: List[str], month: str = None, force: bool = False) -> None:
    """
    Lädt Finanzdaten für alle Ticker von Yahoo Finance.
    Speichert in data/financials/YYYY-MM.csv und latest.csv
    
    Args:
        tickers: List of ticker symbols
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "financials"):
        logger.info(f"Financials for {month} already exist. Skipping update. (Use force=True to override)")
        return
    
    try:
        logger.info(f"Loading financial data for {len(tickers)} tickers...")
        
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
        
        # Save monthly file
        monthly_path = financials_dir / f"{month}.csv"
        financial_df.to_csv(monthly_path, index=False)
        logger.info(f"Saved financials to {monthly_path} ({len(financial_df)} tickers)")
        
        # Save latest file
        latest_path = financials_dir / "latest.csv"
        financial_df.to_csv(latest_path, index=False)
        logger.info(f"Saved financials to {latest_path}")
        
    except Exception as e:
        logger.error(f"Failed to update financials: {e}")
        raise


def update_sentiments(tickers: List[str], financials_df: pd.DataFrame = None, month: str = None, force: bool = False) -> Dict:
    """
    Lädt alle Sentiment-Daten für die gegebenen Ticker.
    
    Args:
        tickers: List of ticker symbols
        financials_df: Optional DataFrame with ticker, marketCap, sector columns for dynamic fallback generation
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    
    Returns: Dict mit superinvestor_score, reddit_score, x_score
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
        logger.info(f"Loading sentiment data for {len(tickers)} tickers...")
        
        # Import modules (all in scripts/)
        from reddit import get_reddit_mentions
        from twitter import get_x_sentiment_score
        from dataroma import get_superinvestor_data
        
        # Load Reddit sentiment
        logger.info("Loading Reddit sentiment...")
        reddit_scores = get_reddit_mentions(tickers, days_back=120, financials_df=financials_df)
        if not reddit_scores:
            raise ValueError("Failed to load Reddit sentiment data")
        logger.info(f"Reddit: {len(reddit_scores)} tickers scored")
        
        # Load X sentiment
        logger.info("Loading X (Twitter) sentiment...")
        x_scores = get_x_sentiment_score(tickers, financials_df=financials_df)
        if not x_scores:
            raise ValueError("Failed to load X sentiment data")
        logger.info(f"X: {len(x_scores)} tickers scored")
        
        # Load Superinvestor data
        logger.info("Loading Superinvestor data...")
        superinvestor_scores = get_superinvestor_data(universe=tickers)
        if not superinvestor_scores:
            raise ValueError("Failed to load Superinvestor data")
        logger.info(f"Superinvestor: {len(superinvestor_scores)} tickers scored")
        
        # Validate scores - check for too many neutral/default values
        def validate_scores(score_dict: Dict, score_name: str, threshold: float = 0.5, max_neutral_ratio: float = 0.7):
            """Check if too many scores are at the neutral/default value"""
            if not score_dict:
                logger.warning(f"{score_name}: Empty score dictionary")
                return False
            
            neutral_count = sum(1 for v in score_dict.values() if abs(v - threshold) < 0.01)
            total_count = len(score_dict)
            neutral_ratio = neutral_count / total_count if total_count > 0 else 1.0
            
            if neutral_ratio > max_neutral_ratio:
                logger.warning(
                    f"{score_name}: {neutral_count}/{total_count} ({neutral_ratio*100:.1f}%) scores are neutral ({threshold}). "
                    f"This might indicate scraping issues or missing data."
                )
                return False
            
            logger.info(f"{score_name}: {neutral_count}/{total_count} ({neutral_ratio*100:.1f}%) neutral scores (acceptable)")
            return True
        
        # Validate each score type
        validate_scores(superinvestor_scores, "Superinvestor", threshold=0.5, max_neutral_ratio=0.6)
        validate_scores(reddit_scores, "Reddit", threshold=0.3, max_neutral_ratio=0.7)
        validate_scores(x_scores, "X/Twitter", threshold=0.5, max_neutral_ratio=0.5)
        
        # Combine into single dict
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "superinvestor_score": superinvestor_scores,
            "reddit_score": reddit_scores,
            "x_score": x_scores
        }
        
        logger.info("All sentiment data loaded successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to update sentiments: {e}")
        raise


def update_extended_scores(tickers: List[str], month: str = None, force: bool = False) -> None:
    """
    Lädt PE History und Analyst Scores für alle Ticker.
    Speichert in data/extended_scores/YYYY-MM.json und latest.json
    
    Args:
        tickers: List of ticker symbols
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "extended_scores"):
        logger.info(f"Extended scores for {month} already exist. Skipping update. (Use force=True to override)")
        return
    
    try:
        logger.info(f"Loading extended scores (PE History & Analyst) for {len(tickers)} tickers...")
        
        # Import data_providers
        from data_providers import get_pe_history_features, get_analyst_summary
        
        pe_history_data = {}
        analyst_data = {}
        
        # Process tickers with progress logging
        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing extended scores: {i + 1}/{len(tickers)}")
            
            try:
                # PE History
                pe_feats = get_pe_history_features(ticker)
                pe_history_data[ticker] = {
                    'pe_current': pe_feats.get('pe_current'),
                    'pe_low_2y': pe_feats.get('pe_low_2y'),
                    'pe_low_5y': pe_feats.get('pe_low_5y'),
                    'pe_score': pe_feats.get('pe_score', 0.5)
                }
                
                # Analyst Summary
                analyst_summary = get_analyst_summary(ticker)
                analyst_data[ticker] = {
                    'analyst_score': analyst_summary.get('analyst_score', 0.5),
                    'recommendationMean': analyst_summary.get('recommendationMean'),
                    'targetMeanPrice': analyst_summary.get('targetMeanPrice'),
                    'target_delta': analyst_summary.get('target_delta')
                }
            except Exception as e:
                logger.debug(f"Error loading extended scores for {ticker}: {e}")
                # Use defaults on error
                pe_history_data[ticker] = {
                    'pe_current': None,
                    'pe_low_2y': None,
                    'pe_low_5y': None,
                    'pe_score': 0.5
                }
                analyst_data[ticker] = {
                    'analyst_score': 0.5,
                    'recommendationMean': None,
                    'targetMeanPrice': None,
                    'target_delta': None
                }
        
        # Combine into single dict
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "pe_history": pe_history_data,
            "analyst": analyst_data
        }
        
        # Save to data/extended_scores/
        extended_scores_dir = ROOT_DIR / "data/extended_scores"
        extended_scores_dir.mkdir(parents=True, exist_ok=True)
        
        # Save monthly file
        monthly_path = extended_scores_dir / f"{month}.json"
        with open(monthly_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extended scores to {monthly_path}")
        
        # Save latest file
        latest_path = extended_scores_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extended scores to {latest_path}")
        
    except Exception as e:
        logger.error(f"Failed to update extended scores: {e}")
        raise


def update_ai_scores(tickers: List[str], financials_df: pd.DataFrame, scores_dict: Dict, month: str = None, force: bool = False) -> None:
    """
    Berechnet KI-Scores (Moat, Quality, Predicted Performance) für alle Ticker.
    Speichert in data/ai_scores/YYYY-MM.json und latest.json
    
    Args:
        tickers: List of ticker symbols
        financials_df: DataFrame with financial data
        scores_dict: Dictionary with sentiment scores
        month: Monat im Format "YYYY-MM" (default: aktueller Monat)
        force: Wenn True, überschreibt vorhandene Dateien. Wenn False, überspringt wenn Datei existiert.
    """
    if month is None:
        month = datetime.now().strftime("%Y-%m")
    
    # Prüfe ob Datei bereits existiert
    if not force and _file_exists_for_month(month, "ai_scores"):
        logger.info(f"AI scores for {month} already exist. Skipping update. (Use force=True to override)")
        return
    
    try:
        logger.info(f"Calculating AI scores for {len(tickers)} tickers...")
        
        # Import AI scores module
        from ai_scores import get_ai_moat_score, get_ai_quality_score, get_ai_predicted_performance
        
        moat_scores = {}
        quality_scores = {}
        predicted_performance = {}
        
        # Create mapping from ticker to financial data
        financials_dict = financials_df.set_index('ticker').to_dict('index')
        
        # Process tickers with progress logging
        # Developer Tier: Bis zu 500 RPM - keine Delays mehr nötig!
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:  # Häufigeres Logging für besseres Feedback
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
                    'quality_score': 0.5  # Will be calculated separately
                }
                
                # Iteriere über alle 3 Kategorien pro Ticker
                # Developer Tier: Keine Delays mehr nötig (bis zu 500 RPM)
                categories = [
                    ("moat", get_ai_moat_score, moat_scores),
                    ("quality", get_ai_quality_score, quality_scores),
                    ("performance", get_ai_predicted_performance, predicted_performance)
                ]
                
                for category_name, score_func, score_dict in categories:
                    # Calculate score - kann einzeln fehlschlagen
                    try:
                        if category_name == "performance":
                            result = score_func(ticker, financial_data, existing_scores)
                            # Performance ist ein Dict, nicht ein einzelner Wert
                            predicted_performance[ticker] = result
                        else:
                            result = score_func(ticker, financial_data, existing_scores)
                            score_dict[ticker] = result
                    except Exception as e:
                        logger.warning(f"Failed to get {category_name} score for {ticker}: {e}")
                        # Use defaults on error
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
                # Use defaults on error
                moat_scores[ticker] = 0.5
                quality_scores[ticker] = 0.5
                predicted_performance[ticker] = {
                    'cagr_1y': 8.0,
                    'cagr_2y': 8.0,
                    'cagr_5y': 8.0,
                    'cagr_10y': 8.0
                }
        
        if failed_tickers:
            logger.warning(f"Failed to calculate AI scores for {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
        
        # Combine into single dict
        result = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "moat_scores": moat_scores,
            "quality_scores": quality_scores,
            "predicted_performance": predicted_performance
        }
        
        # Save to data/ai_scores/
        ai_scores_dir = ROOT_DIR / "data/ai_scores"
        ai_scores_dir.mkdir(parents=True, exist_ok=True)
        
        # Save monthly file
        monthly_path = ai_scores_dir / f"{month}.json"
        with open(monthly_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved AI scores to {monthly_path}")
        
        # Save latest file
        latest_path = ai_scores_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved AI scores to {latest_path}")
        
    except Exception as e:
        logger.error(f"Failed to update AI scores: {e}")
        # Don't raise - AI scores are optional, continue with fallbacks
        logger.warning("Continuing without AI scores - will use fallback values")


def save_scores(scores: Dict, month: str = None) -> None:
    """
    Speichert Sentiment-Daten in data/scores/YYYY-MM.json und latest.json
    """
    try:
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
        scores_dir = ROOT_DIR / "data/scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        
        # Save monthly file
        monthly_path = scores_dir / f"{month}.json"
        with open(monthly_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scores to {monthly_path}")
        
        # Save latest file
        latest_path = scores_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved scores to {latest_path}")
        
    except Exception as e:
        logger.error(f"Failed to save scores: {e}")
        raise


def main(force: bool = False):
    """
    Hauptfunktion: Führt alle monatlichen Updates durch.
    Überspringt Schritte, für die bereits Daten für den aktuellen Monat existieren.
    
    Args:
        force: Wenn True, werden alle Schritte neu ausgeführt, auch wenn Daten bereits existieren.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting monthly update process")
        if force:
            logger.info("FORCE MODE: Will overwrite existing data")
        else:
            logger.info("SKIP MODE: Will skip steps with existing data")
        logger.info("=" * 60)
        
        month = datetime.now().strftime("%Y-%m")
        
        # Step 1: Update universe (lädt vorhandene wenn verfügbar)
        tickers = update_universe(month, force=force)
        
        # Step 2: Update financials (überspringt wenn bereits vorhanden)
        update_financials(tickers, month, force=force)
        
        # Step 3: Load financials_df for dynamic fallback generation
        financials_path = ROOT_DIR / "data/financials/latest.csv"
        financials_df = None
        if financials_path.exists():
            try:
                financials_df = pd.read_csv(financials_path)
                logger.info(f"Loaded financials_df with {len(financials_df)} tickers for fallback generation")
            except Exception as e:
                logger.warning(f"Could not load financials_df: {e}. Will use basic fallbacks.")
        
        # Step 4: Update extended scores (PE History & Analyst) - überspringt wenn bereits vorhanden
        update_extended_scores(tickers, month, force=force)
        
        # Step 5: Update sentiments (with financials_df for dynamic fallbacks) - lädt aus Datei wenn vorhanden
        scores = update_sentiments(tickers, financials_df=financials_df, month=month, force=force)
        
        # Step 6: Save sentiment scores (nur wenn neu berechnet)
        if force or not _file_exists_for_month(month, "scores"):
            save_scores(scores, month)
        
        # Step 7: Update AI scores (requires financials and scores) - überspringt wenn bereits vorhanden
        # Reload financials_df if not already loaded
        if financials_df is None:
            financials_path = ROOT_DIR / "data/financials/latest.csv"
            if financials_path.exists():
                try:
                    financials_df = pd.read_csv(financials_path)
                except Exception as e:
                    logger.warning(f"Could not load financials_df for AI scores: {e}")
                    financials_df = None
        
        if financials_df is not None:
            update_ai_scores(tickers, financials_df, scores, month, force=force)
        else:
            logger.warning("Financials not available for AI score calculation - skipping")
        
        logger.info("=" * 60)
        logger.info("Monthly update completed successfully!")
        logger.info(f"Universe: {len(tickers)} tickers")
        logger.info(f"Financials: saved to data/financials/{month}.csv")
        logger.info(f"Scores: {len(scores['superinvestor_score'])} superinvestor, "
                   f"{len(scores['reddit_score'])} reddit, "
                   f"{len(scores['x_score'])} x")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Monthly update FAILED: {e}")
        logger.error("=" * 60)
        raise


if __name__ == '__main__':
    import sys
    # Prüfe ob --force Flag gesetzt ist
    force = '--force' in sys.argv or '-f' in sys.argv
    main(force=force)
