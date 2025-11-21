# portfolio_calculator.py
"""
Portfolio Calculator
Berechnet Portfolios dynamisch aus monatlichen Daten (tickers, financials, scores).
"""
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Root-Verzeichnis bestimmen
ROOT_DIR = Path(__file__).parent.parent

logger = logging.getLogger(__name__)

# Anlagehorizont-Gewichtungen
HORIZON_WEIGHTS = {
    "1 Jahr": {
        'value': 0.50, 'quality': 0.15, 'community': 0.20,
        'pe_vs_history': 0.10, 'insider': 0.03, 'analyst': 0.02, 'ki_moat': 0.00
    },
    "2 Jahre": {
        'value': 0.40, 'quality': 0.20, 'community': 0.20,
        'pe_vs_history': 0.08, 'insider': 0.05, 'analyst': 0.05, 'ki_moat': 0.07
    },
    "5+ Jahre": {
        'value': 0.30, 'quality': 0.30, 'community': 0.20,
        'pe_vs_history': 0.05, 'insider': 0.03, 'analyst': 0.05, 'ki_moat': 0.12
    }
}


def load_monthly_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Lädt alle monatlichen Daten.
    Returns: (tickers_df, financials_df, scores_dict)
    """
    try:
        # Load tickers (try latest.csv first, then fallback to tickers_over_50B_full.csv)
        tickers_path = ROOT_DIR / "data/tickers/latest.csv"
        if not tickers_path.exists():
            # Fallback to static file
            tickers_path = ROOT_DIR / "data/tickers/tickers_over_50B_full.csv"
            if not tickers_path.exists():
                raise FileNotFoundError(f"Tickers file not found: {tickers_path}")
        tickers_df = pd.read_csv(tickers_path)
        logger.info(f"Loaded {len(tickers_df)} tickers from {tickers_path}")
        
        # Load financials
        financials_path = ROOT_DIR / "data/financials/latest.csv"
        if not financials_path.exists():
            raise FileNotFoundError(f"Financials file not found: {financials_path}")
        financials_df = pd.read_csv(financials_path)
        logger.info(f"Loaded financials for {len(financials_df)} tickers from {financials_path}")
        
        # Load scores
        scores_path = ROOT_DIR / "data/scores/latest.json"
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores file not found: {scores_path}")
        with open(scores_path, 'r', encoding='utf-8') as f:
            scores_dict = json.load(f)
        logger.info(f"Loaded scores from {scores_path}")
        
        return tickers_df, financials_df, scores_dict
        
    except Exception as e:
        logger.error(f"Failed to load monthly data: {e}")
        raise


def combine_data(tickers_df: pd.DataFrame, financials_df: pd.DataFrame, scores_dict: Dict) -> pd.DataFrame:
    """
    Kombiniert alle Daten zu einem DataFrame.
    """
    try:
        # Start with financials as base
        df = financials_df.copy()
        
        # Add marketCap from tickers_df if not in financials
        if 'marketCap' not in df.columns and 'marketCap' in tickers_df.columns:
            tickers_map = tickers_df.set_index('ticker')['marketCap'].to_dict()
            df['marketCap'] = df['ticker'].map(tickers_map)
        
        # Ensure sector column exists
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        
        # Add sentiment scores
        superinvestor_scores = scores_dict.get('superinvestor_score', {})
        reddit_scores = scores_dict.get('reddit_score', {})
        x_scores = scores_dict.get('x_score', {})
        
        df['superinvestor_score'] = df['ticker'].map(superinvestor_scores).fillna(0.5)
        df['reddit_score'] = df['ticker'].map(reddit_scores).fillna(0.5)
        df['x_score'] = df['ticker'].map(x_scores).fillna(0.5)
        
        # Load AI moat if available
        ai_moat_path = ROOT_DIR / "data/community_data/ai_moat.json"
        if ai_moat_path.exists():
            with open(ai_moat_path, 'r', encoding='utf-8') as f:
                ai_moat_data = json.load(f)
            ki_moat_scores = ai_moat_data.get('ki_moat_score', {})
            df['ki_moat_score'] = df['ticker'].map(ki_moat_scores).fillna(0.5)
        else:
            df['ki_moat_score'] = 0.5
        
        # Load PE vs History, Insider, and Analyst scores
        from data_providers import get_pe_history_features, get_analyst_summary
        
        pe_scores = {}
        analyst_scores = {}
        
        # Load scores for all tickers (with caching)
        for ticker in df['ticker'].unique():
            try:
                # PE vs History
                pe_feats = get_pe_history_features(ticker)
                pe_scores[ticker] = pe_feats.get('pe_score', 0.5)
                
                # Analyst Score
                analyst_summary = get_analyst_summary(ticker)
                analyst_scores[ticker] = analyst_summary.get('analyst_score', 0.5)
            except Exception:
                pe_scores[ticker] = 0.5
                analyst_scores[ticker] = 0.5
        
        df['pe_vs_history_score'] = df['ticker'].map(pe_scores).fillna(0.5)
        df['insider_score'] = 0.5  # Not important per user request
        df['analyst_score'] = df['ticker'].map(analyst_scores).fillna(0.5)
        
        logger.info(f"Combined data: {len(df)} tickers with all data")
        return df
        
    except Exception as e:
        logger.error(f"Failed to combine data: {e}")
        raise


def compute_scores_with_horizon(df: pd.DataFrame, horizon: str = "2 Jahre", min_market_cap: float = 30_000_000_000) -> pd.DataFrame:
    """
    Berechnet Scores mit anlagehorizont-spezifischen Gewichtungen.
    """
    try:
        # Filter auf Mindest-MarketCap und vorhandene P/E-Daten
        df = df[df['marketCap'] >= min_market_cap].dropna(subset=['trailingPE', 'forwardPE']).copy()
        
        if len(df) == 0:
            raise ValueError("No tickers after filtering!")
        
        # 1. Value Score (erweitert mit PEG und P/FCF)
        # Ensure columns exist and fill missing values
        for col in ['trailingPE', 'forwardPE', 'pegRatio', 'priceToFreeCashFlow']:
            if col not in df.columns:
                df[col] = None
        
        # Rank each metric (lower is better for value)
        trailing_pe_rank = df['trailingPE'].rank(ascending=True, pct=True, na_option='keep')
        forward_pe_rank = df['forwardPE'].rank(ascending=True, pct=True, na_option='keep')
        peg_rank = df['pegRatio'].rank(ascending=True, pct=True, na_option='keep')
        pfcf_rank = df['priceToFreeCashFlow'].rank(ascending=True, pct=True, na_option='keep')
        
        # Calculate weighted value score
        # Use available metrics only (if PEG/P/FCF missing, weight shifts to P/E)
        value_components = []
        weights = []
        
        # Trailing PE (always available)
        if trailing_pe_rank.notna().any():
            value_components.append((1 - trailing_pe_rank.fillna(0.5)) * 0.35)
            weights.append(0.35)
        
        # Forward PE
        if forward_pe_rank.notna().any():
            value_components.append((1 - forward_pe_rank.fillna(0.5)) * 0.25)
            weights.append(0.25)
        
        # PEG Ratio
        if peg_rank.notna().any():
            value_components.append((1 - peg_rank.fillna(0.5)) * 0.20)
            weights.append(0.20)
        
        # P/FCF
        if pfcf_rank.notna().any():
            value_components.append((1 - pfcf_rank.fillna(0.5)) * 0.20)
            weights.append(0.20)
        
        # Normalize weights if some metrics are missing
        if value_components:
            total_weight = sum(weights)
            if total_weight > 0:
                # Normalize components
                normalized_components = [comp * (1.0 / total_weight) for comp in value_components]
                df['value_score'] = sum(normalized_components)
            else:
                df['value_score'] = 0.5
        else:
            # Fallback: use only P/E if nothing else available
            df['value_score'] = (
                (1 - trailing_pe_rank.fillna(0.5)) * 0.6 +
                (1 - forward_pe_rank.fillna(0.5)) * 0.4
            )
        
        # 2. Quality Score
        median_debt = df['debtToEquity'].median()
        df['debtToEquity_fixed'] = df['debtToEquity'].fillna(median_debt)
        df['quality_score'] = (
            df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 +
            (1 - df['debtToEquity_fixed'].rank(pct=True)) * 0.3
        )
        df = df.drop(columns=['debtToEquity_fixed'])
        
        # 3. Community Score
        for c in ['superinvestor_score', 'reddit_score', 'x_score']:
            if c not in df.columns:
                df[c] = 0.5
            else:
                df[c] = df[c].fillna(0.5)
        
        df['community_score'] = (
            df['superinvestor_score'] * 0.333 +
            df['reddit_score'] * 0.333 +
            df['x_score'] * 0.334
        )
        
        # 4. Final Score mit horizon-spezifischen Gewichtungen
        weights = HORIZON_WEIGHTS.get(horizon, HORIZON_WEIGHTS["2 Jahre"])
        
        # Ensure all required columns exist
        for col in ['pe_vs_history_score', 'insider_score', 'analyst_score', 'ki_moat_score']:
            if col not in df.columns:
                df[col] = 0.5
            else:
                df[col] = df[col].fillna(0.5)
        
        df['final_score'] = (
            df['value_score'] * weights['value'] +
            df['quality_score'] * weights['quality'] +
            df['community_score'] * weights['community'] +
            df['pe_vs_history_score'] * weights['pe_vs_history'] +
            df['insider_score'] * weights['insider'] +
            df['analyst_score'] * weights['analyst'] +
            df['ki_moat_score'] * weights['ki_moat']
        )
        
        return df.sort_values('final_score', ascending=False)
        
    except Exception as e:
        logger.error(f"Failed to compute scores: {e}")
        raise


def calculate_portfolio_from_monthly_data(
    portfolio_size: int = 20,
    horizon: str = "5+ Jahre",
    min_market_cap: float = 30_000_000_000
) -> pd.DataFrame:
    """
    Hauptfunktion: Berechnet Portfolio aus monatlichen Daten.
    
    Args:
        portfolio_size: Anzahl Aktien im Portfolio (5, 10, oder 20)
        horizon: Anlagehorizont ("1 Jahr", "2 Jahre", "5+ Jahre")
        min_market_cap: Minimale Marktkapitalisierung
    
    Returns:
        DataFrame mit Portfolio (inkl. weight Spalte)
    """
    try:
        logger.info(f"Calculating portfolio: size={portfolio_size}, horizon={horizon}")
        
        # Step 1: Load monthly data
        tickers_df, financials_df, scores_dict = load_monthly_data()
        
        # Step 2: Combine data
        df = combine_data(tickers_df, financials_df, scores_dict)
        
        # Step 3: Compute scores with horizon-specific weights
        scored_df = compute_scores_with_horizon(df, horizon=horizon, min_market_cap=min_market_cap)
        
        # Step 4: Construct portfolio
        from portfolio import construct_portfolio
        portfolio = construct_portfolio(scored_df, n=portfolio_size)
        
        # Add weight_% column for display
        portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'
        
        logger.info(f"Portfolio calculated: {len(portfolio)} stocks")
        return portfolio
        
    except Exception as e:
        logger.error(f"Failed to calculate portfolio: {e}")
        raise

