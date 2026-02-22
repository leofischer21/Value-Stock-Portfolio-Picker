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
        'value': 0.145, 'quality': 0.175, 'community': 0.45,
        'pe_vs_history': 0.05, 'insider': 0.03, 'analyst': 0.05, 'ki_moat': 0.10
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


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Datenvalidierung, Outlier-Erkennung und Fallback-Strategien.
    Returns DataFrame mit bereinigten Daten und data_quality_score.
    """
    try:
        df = df.copy()
        
        # Convert numeric columns to float to avoid dtype warnings
        numeric_cols = ['trailingPE', 'forwardPE', 'returnOnEquity', 'beta', 
                       'grossMargins', 'operatingMargins', 'profitMargins',
                       'debtToEquity', 'pegRatio', 'priceToFreeCashFlow']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        # Define expected metrics for quality score calculation
        expected_metrics = ['trailingPE', 'forwardPE', 'returnOnEquity', 'beta', 
                          'grossMargins', 'operatingMargins', 'profitMargins',
                          'debtToEquity', 'pegRatio', 'priceToFreeCashFlow']
        
        # Track data quality per ticker
        quality_scores = []
        
        # Outlier detection and correction
        for idx, row in df.iterrows():
            ticker = row.get('ticker', 'Unknown')
            sector = row.get('sector', 'Unknown')
            valid_metrics = 0
            total_metrics = len(expected_metrics)
            
            # Get sector median for fallbacks
            sector_df = df[df['sector'] == sector] if sector != 'Unknown' else df
            sector_median_pe = sector_df['trailingPE'].median() if 'trailingPE' in sector_df.columns else None
            sector_median_roe = sector_df['returnOnEquity'].median() if 'returnOnEquity' in sector_df.columns else None
            sector_median_margin = sector_df[['grossMargins', 'operatingMargins', 'profitMargins']].mean(axis=1).median() if all(c in sector_df.columns for c in ['grossMargins', 'operatingMargins', 'profitMargins']) else None
            
            # Check and fix P/E outliers
            if 'trailingPE' in df.columns:
                pe_val = row.get('trailingPE')
                if pe_val is not None:
                    if pe_val > 100 or pe_val < 0:
                        # Outlier: use sector median or overall median
                        replacement = sector_median_pe if sector_median_pe is not None else df['trailingPE'].median()
                        if replacement is not None:
                            df.at[idx, 'trailingPE'] = float(replacement)
                            logger.debug(f"Fixed P/E outlier for {ticker}: {pe_val} -> {replacement}")
                    else:
                        valid_metrics += 1
                elif pe_val is None:
                    # Missing: try to fill with sector median
                    if sector_median_pe is not None:
                        df.at[idx, 'trailingPE'] = sector_median_pe
            
            # Check and fix ROE outliers
            if 'returnOnEquity' in df.columns:
                roe_val = row.get('returnOnEquity')
                if roe_val is not None:
                    if roe_val > 1.0:  # > 100%
                        replacement = sector_median_roe if sector_median_roe is not None else df['returnOnEquity'].median()
                        if replacement is not None:
                            df.at[idx, 'returnOnEquity'] = float(replacement)
                            logger.debug(f"Fixed ROE outlier for {ticker}: {roe_val} -> {replacement}")
                    else:
                        valid_metrics += 1
                elif roe_val is None:
                    if sector_median_roe is not None:
                        df.at[idx, 'returnOnEquity'] = float(sector_median_roe)
            
            # Check and fix Beta outliers
            if 'beta' in df.columns:
                beta_val = row.get('beta')
                if beta_val is not None:
                    if beta_val < 0 or beta_val > 5:
                        df.at[idx, 'beta'] = float(1.0)  # Market average
                        logger.debug(f"Fixed Beta outlier for {ticker}: {beta_val} -> 1.0")
                    else:
                        valid_metrics += 1
                elif beta_val is None:
                    df.at[idx, 'beta'] = float(1.0)  # Market average
                    valid_metrics += 1  # Count as valid after filling
            
            # Check and fix Margin outliers
            for margin_col in ['grossMargins', 'operatingMargins', 'profitMargins']:
                if margin_col in df.columns:
                    margin_val = row.get(margin_col)
                    if margin_val is not None:
                        if margin_val > 1.0 or margin_val < -1.0:
                            if sector_median_margin is not None:
                                df.at[idx, margin_col] = float(sector_median_margin)
                                logger.debug(f"Fixed {margin_col} outlier for {ticker}: {margin_val} -> {sector_median_margin}")
                        else:
                            valid_metrics += 1
                    elif margin_val is None:
                        if sector_median_margin is not None:
                            df.at[idx, margin_col] = float(sector_median_margin)
            
            # Check other metrics for validity (not outliers, just presence)
            for metric in ['forwardPE', 'debtToEquity', 'pegRatio', 'priceToFreeCashFlow']:
                if metric in df.columns:
                    val = row.get(metric)
                    if val is not None and not pd.isna(val):
                        valid_metrics += 1
            
            # Calculate data quality score
            data_quality_score = valid_metrics / total_metrics if total_metrics > 0 else 0.0
            quality_scores.append(data_quality_score)
        
        # Add data quality score column
        df['data_quality_score'] = quality_scores
        
        logger.info(f"Data validation complete. Average quality score: {df['data_quality_score'].mean():.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to validate and clean data: {e}")
        # Return original dataframe with default quality score if validation fails
        if 'data_quality_score' not in df.columns:
            df['data_quality_score'] = 0.5
        return df


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
        
        # Add sentiment scores (nur Reddit + X, ohne Superinvestor)
        reddit_scores = scores_dict.get('reddit_score', {})
        x_scores = scores_dict.get('x_score', {})
        
        df['reddit_score'] = df['ticker'].map(reddit_scores).fillna(0.5)
        df['x_score'] = df['ticker'].map(x_scores).fillna(0.5)
        
        # Load AI scores (Moat, Quality, Predicted Performance) from saved monthly data
        ai_scores_path = ROOT_DIR / "data/ai_scores/latest.json"
        if ai_scores_path.exists():
            try:
                with open(ai_scores_path, 'r', encoding='utf-8') as f:
                    ai_data = json.load(f)
                
                # AI Moat Scores (prefer AI over manual)
                ai_moat_scores = ai_data.get('moat_scores', {})
                df['ai_moat_score'] = df['ticker'].map(ai_moat_scores)
                
                # AI Quality Scores
                ai_quality_scores = ai_data.get('quality_scores', {})
                df['ai_quality_score'] = df['ticker'].map(ai_quality_scores)
                
                # Predicted Performance (CAGR)
                predicted_perf = ai_data.get('predicted_performance', {})
                def get_cagr(ticker, key):
                    perf = predicted_perf.get(ticker, {})
                    if isinstance(perf, dict):
                        return perf.get(key, 8.0)
                    return 8.0
                
                df['predicted_cagr_1y'] = df['ticker'].map(lambda t: get_cagr(t, 'cagr_1y')).fillna(8.0)
                df['predicted_cagr_2y'] = df['ticker'].map(lambda t: get_cagr(t, 'cagr_2y')).fillna(8.0)
                df['predicted_cagr_5y'] = df['ticker'].map(lambda t: get_cagr(t, 'cagr_5y')).fillna(8.0)
                df['predicted_cagr_10y'] = df['ticker'].map(lambda t: get_cagr(t, 'cagr_10y')).fillna(8.0)
                
            except Exception as e:
                logger.warning(f"Failed to load AI scores from file: {e}, using fallbacks")
                df['ai_moat_score'] = None
                df['ai_quality_score'] = None
                df['predicted_cagr_1y'] = 8.0
                df['predicted_cagr_2y'] = 8.0
                df['predicted_cagr_5y'] = 8.0
                df['predicted_cagr_10y'] = 8.0
        else:
            logger.warning(f"AI scores file not found at {ai_scores_path}, using fallbacks")
            df['ai_moat_score'] = None
            df['ai_quality_score'] = None
            df['predicted_cagr_1y'] = 8.0
            df['predicted_cagr_2y'] = 8.0
            df['predicted_cagr_5y'] = 8.0
            df['predicted_cagr_10y'] = 8.0
        
        # Ensure ai_moat_score and ai_quality_score are filled
        # First check if we need to load manual AI moat as fallback
        if 'ai_moat_score' not in df.columns or df['ai_moat_score'].isna().all() if 'ai_moat_score' in df.columns else True:
            ai_moat_path = ROOT_DIR / "data/community_data/ai_moat.json"
            if ai_moat_path.exists():
                with open(ai_moat_path, 'r', encoding='utf-8') as f:
                    ai_moat_data = json.load(f)
                ki_moat_scores = ai_moat_data.get('ki_moat_score', {})
                if 'ai_moat_score' not in df.columns:
                    df['ai_moat_score'] = df['ticker'].map(ki_moat_scores)
                else:
                    df['ai_moat_score'] = df['ai_moat_score'].fillna(df['ticker'].map(ki_moat_scores))
        
        # Ensure all AI scores are filled with defaults
        if 'ai_moat_score' not in df.columns:
            df['ai_moat_score'] = 0.5
        df['ai_moat_score'] = df['ai_moat_score'].fillna(0.5).astype('float64')
        
        if 'ai_quality_score' not in df.columns:
            df['ai_quality_score'] = 0.5
        df['ai_quality_score'] = df['ai_quality_score'].fillna(0.5).astype('float64')
        
        # Keep ki_moat_score for backward compatibility (use ai_moat_score if available)
        df['ki_moat_score'] = df['ai_moat_score']
        
        # Load PE vs History and Analyst scores from saved monthly data
        extended_scores_path = ROOT_DIR / "data/extended_scores/latest.json"
        if extended_scores_path.exists():
            try:
                with open(extended_scores_path, 'r', encoding='utf-8') as f:
                    extended_data = json.load(f)
                pe_history_data = extended_data.get('pe_history', {})
                analyst_data = extended_data.get('analyst', {})
                
                pe_scores = {t: d.get('pe_score', 0.5) for t, d in pe_history_data.items()}
                analyst_scores = {t: d.get('analyst_score', 0.5) for t, d in analyst_data.items()}
            except Exception as e:
                logger.warning(f"Failed to load extended scores from file: {e}, using defaults")
                pe_scores = {}
                analyst_scores = {}
        else:
            logger.warning(f"Extended scores file not found at {extended_scores_path}, using defaults")
            pe_scores = {}
            analyst_scores = {}
        
        # Fallback to 0.5 for any missing tickers
        all_tickers = df['ticker'].unique()
        for ticker in all_tickers:
            if ticker not in pe_scores:
                pe_scores[ticker] = 0.5
            if ticker not in analyst_scores:
                analyst_scores[ticker] = 0.5
        
        df['pe_vs_history_score'] = df['ticker'].map(pe_scores).fillna(0.5)
        df['insider_score'] = 0.5  # Not important per user request
        df['analyst_score'] = df['ticker'].map(analyst_scores).fillna(0.5)
        
        # Validate and clean data (outlier detection, fallback strategies)
        df = validate_and_clean_data(df)
        
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
        
        # Filter negative P/E und Forward P/E Werte aus
        df = df[(df['trailingPE'] > 0) & (df['forwardPE'] > 0)].copy()
        
        if len(df) == 0:
            raise ValueError("No tickers after filtering!")
        
        # 1. Value Score (erweitert mit PEG, P/FCF und EV/EBITDA)
        # Ensure columns exist and fill missing values
        for col in ['trailingPE', 'forwardPE', 'pegRatio', 'priceToFreeCashFlow', 'enterpriseToEbitda']:
            if col not in df.columns:
                df[col] = None
        
        # Rank each metric (lower is better for value)
        trailing_pe_rank = df['trailingPE'].rank(ascending=True, pct=True, na_option='keep')
        forward_pe_rank = df['forwardPE'].rank(ascending=True, pct=True, na_option='keep')
        peg_rank = df['pegRatio'].rank(ascending=True, pct=True, na_option='keep')
        pfcf_rank = df['priceToFreeCashFlow'].rank(ascending=True, pct=True, na_option='keep')
        ev_ebitda_rank = df['enterpriseToEbitda'].rank(ascending=True, pct=True, na_option='keep')
        
        # Calculate weighted value score
        # Use available metrics only (if metrics missing, weight shifts to available ones)
        value_components = []
        weights = []
        
        # Trailing PE (30%)
        if trailing_pe_rank.notna().any():
            value_components.append((1 - trailing_pe_rank.fillna(0.5)) * 0.30)
            weights.append(0.30)
        
        # Forward PE (20%)
        if forward_pe_rank.notna().any():
            value_components.append((1 - forward_pe_rank.fillna(0.5)) * 0.20)
            weights.append(0.20)
        
        # PEG Ratio (15%)
        if peg_rank.notna().any():
            value_components.append((1 - peg_rank.fillna(0.5)) * 0.15)
            weights.append(0.15)
        
        # P/FCF (15%)
        if pfcf_rank.notna().any():
            value_components.append((1 - pfcf_rank.fillna(0.5)) * 0.15)
            weights.append(0.15)
        
        # EV/EBITDA (20%)
        if ev_ebitda_rank.notna().any():
            value_components.append((1 - ev_ebitda_rank.fillna(0.5)) * 0.20)
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
        
        # 2. Quality Score (erweitert mit ROIC und Profit Margins, kombiniert mit AI Quality Score)
        # Ensure columns exist
        for col in ['returnOnEquity', 'returnOnInvestedCapital', 'grossMargins', 'operatingMargins', 'profitMargins', 'debtToEquity', 'ai_quality_score']:
            if col not in df.columns:
                df[col] = None
        
        # ROIC Rank (höher = besser)
        roic_rank = df['returnOnInvestedCapital'].rank(ascending=False, pct=True, na_option='keep')
        
        # Average Margin Rank (höher = besser)
        # Calculate avgMargin from available margins
        margin_cols = ['grossMargins', 'operatingMargins', 'profitMargins']
        available_margins = df[margin_cols].notna().any(axis=1)
        df['avgMargin'] = df[margin_cols].mean(axis=1)
        margin_rank = df['avgMargin'].rank(ascending=False, pct=True, na_option='keep')
        
        # ROE Rank (höher = besser)
        roe_rank = df['returnOnEquity'].rank(ascending=False, pct=True, na_option='keep')
        
        # Debt Rank (niedriger = besser, daher 1 - rank)
        median_debt = df['debtToEquity'].median()
        df['debtToEquity_fixed'] = df['debtToEquity'].fillna(median_debt)
        debt_rank = df['debtToEquity_fixed'].rank(pct=True, na_option='keep')
        
        # Calculate base quality score
        base_quality_score = (
            roe_rank.fillna(0.5) * 0.30 +
            roic_rank.fillna(0.5) * 0.25 +
            margin_rank.fillna(0.5) * 0.20 +
            (1 - debt_rank.fillna(0.5)) * 0.25
        )
        
        # Combine with AI quality score (70% base, 30% AI)
        ai_quality = df['ai_quality_score'].fillna(0.5)
        df['quality_score'] = base_quality_score * 0.7 + ai_quality * 0.3
        
        df = df.drop(columns=['debtToEquity_fixed', 'avgMargin'])
        
        # 3. Community Score (nur Reddit + X, ohne Superinvestor)
        for c in ['reddit_score', 'x_score']:
            if c not in df.columns:
                df[c] = 0.5
            else:
                df[c] = df[c].fillna(0.5)
        
        df['community_score'] = (
            df['reddit_score'] * 0.5 +
            df['x_score'] * 0.5
        )
        
        # 4. Final Score mit horizon-spezifischen Gewichtungen
        weights = HORIZON_WEIGHTS.get(horizon, HORIZON_WEIGHTS["2 Jahre"])
        
        # Ensure all required columns exist
        for col in ['pe_vs_history_score', 'insider_score', 'analyst_score', 'ai_moat_score']:
            if col not in df.columns:
                df[col] = 0.5
            else:
                df[col] = df[col].fillna(0.5)
        
        # Use ai_moat_score (which may fall back to ki_moat_score)
        moat_score = df.get('ai_moat_score', df.get('ki_moat_score', 0.5))
        if 'ai_moat_score' in df.columns:
            moat_score = df['ai_moat_score']
        elif 'ki_moat_score' in df.columns:
            moat_score = df['ki_moat_score']
        else:
            moat_score = pd.Series([0.5] * len(df))
        
        df['final_score'] = (
            df['value_score'] * weights['value'] +
            df['quality_score'] * weights['quality'] +
            df['community_score'] * weights['community'] +
            df['pe_vs_history_score'] * weights['pe_vs_history'] +
            df['insider_score'] * weights['insider'] +
            df['analyst_score'] * weights['analyst'] +
            moat_score * weights['ki_moat']
        )
        
        # Sektor-Penalty für typische Value-Sektoren (die oft niedrige P/E haben)
        # Diese Sektoren sind tendenziell überrepräsentiert, daher leichte Penalty
        value_sector_penalties = {
            'Financial Services': -0.08,  # Banken, Versicherungen haben oft niedriges P/E
            'Energy': -0.08,              # Öl & Gas haben oft niedriges P/E
            'Utilities': -0.05,           # Versorger haben oft niedriges P/E
            'Real Estate': -0.05,         # REITs haben oft niedriges P/E
        }
        
        # Apply sector penalty to final_score
        if 'sector' in df.columns:
            for sector, penalty in value_sector_penalties.items():
                mask = df['sector'] == sector
                df.loc[mask, 'final_score'] = df.loc[mask, 'final_score'] + penalty
        
        # Ensure final_score stays in reasonable range (0.0 to 1.0)
        df['final_score'] = df['final_score'].clip(lower=0.0, upper=1.0)
        
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
        
        # Step 3: Compute scores with horizon-specific weights (initial, ohne committee_score)
        scored_df = compute_scores_with_horizon(df, horizon=horizon, min_market_cap=min_market_cap)
        
        # Step 3.5: LLM Committee Evaluation (nur für Top 20)
        # Wähle Top 20 basierend auf initialem final_score
        top_tickers = scored_df.head(portfolio_size)['ticker'].tolist()
        logger.info(f"Running LLM committee evaluation for {len(top_tickers)} tickers")
        
        committee_scores = {}
        try:
            from llm_committee import get_llm_committee_score
            committee_scores = get_llm_committee_score(top_tickers, scored_df)
            
            # Füge committee_score zu DataFrame hinzu
            scored_df['committee_score'] = scored_df['ticker'].map(committee_scores).fillna(0.5)
            logger.info(f"LLM committee scores calculated for {len(committee_scores)} tickers")
        except Exception as e:
            logger.warning(f"LLM committee evaluation failed: {e}, using neutral scores")
            scored_df['committee_score'] = 0.5
        
        # Step 3.6: Recalculate final_score mit committee_score (25% Gewichtung)
        if 'committee_score' in scored_df.columns:
            weights = HORIZON_WEIGHTS.get(horizon, HORIZON_WEIGHTS["2 Jahre"])
            committee_weight = 0.25  # 25% Gewichtung für committee_score
            
            # Berechne Summe der bestehenden Gewichtungen
            base_weight_sum = sum([
                weights.get('value', 0),
                weights.get('quality', 0),
                weights.get('community', 0),
                weights.get('pe_vs_history', 0),
                weights.get('insider', 0),
                weights.get('analyst', 0),
                weights.get('ki_moat', 0)
            ])
            
            # Skaliere bestehende Gewichtungen (75% bleiben für andere Scores)
            remaining_weight = 1.0 - committee_weight
            if base_weight_sum > 0:
                scaled_weights = {k: v * (remaining_weight / base_weight_sum) for k, v in weights.items()}
            else:
                scaled_weights = weights
            
            # Use ai_moat_score (which may fall back to ki_moat_score)
            moat_score = scored_df.get('ai_moat_score', scored_df.get('ki_moat_score', 0.5))
            if 'ai_moat_score' in scored_df.columns:
                moat_score = scored_df['ai_moat_score']
            elif 'ki_moat_score' in scored_df.columns:
                moat_score = scored_df['ki_moat_score']
            else:
                moat_score = pd.Series([0.5] * len(scored_df))
            
            # Neu berechnen mit committee_score
            scored_df['final_score'] = (
                scored_df['value_score'] * scaled_weights.get('value', 0) +
                scored_df['quality_score'] * scaled_weights.get('quality', 0) +
                scored_df['community_score'] * scaled_weights.get('community', 0) +
                scored_df['pe_vs_history_score'] * scaled_weights.get('pe_vs_history', 0) +
                scored_df['insider_score'] * scaled_weights.get('insider', 0) +
                scored_df['analyst_score'] * scaled_weights.get('analyst', 0) +
                moat_score * scaled_weights.get('ki_moat', 0) +
                scored_df['committee_score'] * committee_weight
            )
            
            # Sektor-Penalty (wie vorher)
            value_sector_penalties = {
                'Financial Services': -0.08,
                'Energy': -0.08,
                'Utilities': -0.05,
                'Real Estate': -0.05,
            }
            
            if 'sector' in scored_df.columns:
                for sector, penalty in value_sector_penalties.items():
                    mask = scored_df['sector'] == sector
                    scored_df.loc[mask, 'final_score'] = scored_df.loc[mask, 'final_score'] + penalty
            
            # Ensure final_score stays in reasonable range
            scored_df['final_score'] = scored_df['final_score'].clip(lower=0.0, upper=1.0)
            
            # Neu sortieren basierend auf aktualisiertem final_score
            scored_df = scored_df.sort_values('final_score', ascending=False)
            logger.info("Final scores recalculated with LLM committee scores")
        
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

