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


def update_universe() -> List[str]:
    """
    Generiert neue Ticker-Liste über 50B MarketCap (US-Ticker).
    Returns: List of ticker symbols
    """
    try:
        logger.info("Starting universe update...")
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


def update_financials(tickers: List[str], month: str = None) -> None:
    """
    Lädt Finanzdaten für alle Ticker von Yahoo Finance.
    Speichert in data/financials/YYYY-MM.csv und latest.csv
    """
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
                }
                # Only add if we have at least some key data
                if row['marketCap'] is not None or row['trailingPE'] is not None:
                    rows.append(row)
        
        if not rows:
            raise ValueError("No valid financial data rows created!")
        
        financial_df = pd.DataFrame(rows)
        
        # Determine month
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        
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


def update_sentiments(tickers: List[str]) -> Dict:
    """
    Lädt alle Sentiment-Daten für die gegebenen Ticker.
    Returns: Dict mit superinvestor_score, reddit_score, x_score
    """
    try:
        logger.info(f"Loading sentiment data for {len(tickers)} tickers...")
        
        # Import modules (all in scripts/)
        from reddit import get_reddit_mentions
        from twitter import get_x_sentiment_score
        from dataroma import get_superinvestor_data
        
        # Load Reddit sentiment
        logger.info("Loading Reddit sentiment...")
        reddit_scores = get_reddit_mentions(tickers, days_back=120)
        if not reddit_scores:
            raise ValueError("Failed to load Reddit sentiment data")
        logger.info(f"Reddit: {len(reddit_scores)} tickers scored")
        
        # Load X sentiment
        logger.info("Loading X (Twitter) sentiment...")
        x_scores = get_x_sentiment_score(tickers)
        if not x_scores:
            raise ValueError("Failed to load X sentiment data")
        logger.info(f"X: {len(x_scores)} tickers scored")
        
        # Load Superinvestor data
        logger.info("Loading Superinvestor data...")
        superinvestor_scores = get_superinvestor_data(universe=tickers)
        if not superinvestor_scores:
            raise ValueError("Failed to load Superinvestor data")
        logger.info(f"Superinvestor: {len(superinvestor_scores)} tickers scored")
        
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


def main():
    """
    Hauptfunktion: Führt alle monatlichen Updates durch
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting monthly update process")
        logger.info("=" * 60)
        
        # Step 1: Update universe
        tickers = update_universe()
        
        # Step 2: Update financials
        month = datetime.now().strftime("%Y-%m")
        update_financials(tickers, month)
        
        # Step 3: Update sentiments
        scores = update_sentiments(tickers)
        
        # Step 4: Save scores
        save_scores(scores, month)
        
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
    main()
