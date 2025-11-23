# src/run_picker.py
import argparse
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime
import requests
import os
from scripts.portfolio import construct_portfolio, save_portfolio

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

# NEUE KONSTANTE: Setze deinen Alpha Vantage Key hier als Umgebungsvariable
# Ersetze "DEMO" durch deinen Key, falls os.environ.get ihn nicht findet
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "DEMO")

MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20

# Config override (optional YAML)
CACHE_TTL = 24 * 3600
try:
    import yaml
    cfg_path = ROOT_DIR / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        PORTFOLIO_SIZE = int(cfg.get('portfolio_size', PORTFOLIO_SIZE))
        MIN_MARKET_CAP = int(cfg.get('min_market_cap', MIN_MARKET_CAP))
        ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', cfg.get('alpha_vantage_key', ALPHA_VANTAGE_KEY))
        CACHE_TTL = int(cfg.get('cache_ttl_seconds', CACHE_TTL))
except Exception:
    # PyYAML not installed or config missing — continue with defaults
    pass

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

def load_universe():
    return ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V",
            "MA","UNH","PG","HD","KO","PEP","ADBE","CRM","ORCL","CSCO"]

# NEUE FUNKTION: Caching für API-Antworten
from scripts.cache import get as cache_get, set as cache_set
        
# NEUE FUNKTION: Ruft historisches P/E von Alpha Vantage ab
def fetch_historical_pe(tk):
    """Ruft P/E vs. 52-Wochen-Durchschnitt mithilfe der Alpha Vantage API ab."""
    cached_data = cache_get(f"{tk}_alpha_vantage_pe")
    if cached_data:
        return cached_data.get('pe_ratio_score', 0.5)

    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={tk}&apikey={ALPHA_VANTAGE_KEY}"
        from scripts.httpx import get_json
        data = get_json(url, timeout=10) or {}

        # Sicheres Lesen verschiedener möglicher Keys
        current_pe = safe_float(data.get('TrailingPE') or data.get('PERatio') or data.get('PE'), default=0.0)
        eps = safe_float(data.get('EPS'), default=1e-6)

        pe_52wk_high = safe_float(data.get('52WeekHigh'), default=0.0)
        pe_52wk_low = safe_float(data.get('52WeekLow'), default=0.0)

        if eps == 0:
            eps = 1e-6

        if pe_52wk_high > 0 and pe_52wk_low > 0:
            pe_52wk_high_ratio = pe_52wk_high / eps
            pe_52wk_low_ratio = pe_52wk_low / eps
            avg_52wk_pe = (pe_52wk_high_ratio + pe_52wk_low_ratio) / 2
        else:
            avg_52wk_pe = current_pe if current_pe > 0 else 1.0

        pe_vs_history_score = 0.5
        if current_pe > 0 and avg_52wk_pe > 0:
            pe_ratio = current_pe / avg_52wk_pe
            pe_vs_history_score = max(0.0, min(1.0, 1.5 - pe_ratio))

        cache_set(f"{tk}_alpha_vantage_pe", {'pe_ratio_score': pe_vs_history_score}, ttl_seconds=CACHE_TTL)
        time.sleep(0.5)
        return pe_vs_history_score

    except Exception as e:
        logger.warning("Alpha Vantage Fehler bei %s: %s. Fallback auf 0.5.", tk, e)
        return 0.5


def gather(universe):
    logger.info("Lade Fundamentaldaten + erweiterte Signale...")

    # Load community signals and AI moat robustly (supports several JSON formats)
    from scripts.community import load_community_signals, load_ai_moat
    from scripts.dataroma import get_superinvestor_data as fetch_dataroma_fallback
    from scripts.reddit import get_reddit_mentions as fetch_reddit_fallback
    from scripts.twitter import get_x_sentiment_score as fetch_x_fallback

    superinvestor, reddit, x_sent = load_community_signals()
    ki_moat = load_ai_moat()
    # If any mapping is empty, fallback to scrapers (use universe-aware fetchers)
    if not superinvestor:
        try:
            superinvestor = fetch_dataroma_fallback()
        except Exception:
            superinvestor = {}
    if not reddit:
        try:
            reddit = fetch_reddit_fallback(universe)
        except Exception:
            reddit = {}
    if not x_sent:
        try:
            x_sent = fetch_x_fallback(universe)
        except Exception:
            x_sent = {}
    if not ki_moat:
        ki_moat = {}

    # Try to ingest on-disk scraped caches (these are valuable real scrape results)
    cache_dir = ROOT_DIR / 'data'
    cache_fill_counts = {'reddit': 0, 'x': 0, 'superinvestor': 0}
    try:
        if cache_dir.exists():
            rc_path = cache_dir / 'reddit_cache.json'
            xc_path = cache_dir / 'x_sentiment_cache.json'
            dr_path = cache_dir / 'dataroma_cache.json'

            if rc_path.exists():
                try:
                    with open(rc_path, 'r', encoding='utf-8') as f:
                        rc = json.load(f) or {}
                    for k, v in rc.items():
                        if k not in reddit:
                            reddit[k] = v
                            cache_fill_counts['reddit'] += 1
                except Exception:
                    logger.debug('Fehler beim Laden von reddit_cache.json', exc_info=True)

            if xc_path.exists():
                try:
                    with open(xc_path, 'r', encoding='utf-8') as f:
                        xc = json.load(f) or {}
                    for k, v in xc.items():
                        if k not in x_sent:
                            x_sent[k] = v
                            cache_fill_counts['x'] += 1
                except Exception:
                    logger.debug('Fehler beim Laden von x_sentiment_cache.json', exc_info=True)

            if dr_path.exists():
                try:
                    with open(dr_path, 'r', encoding='utf-8') as f:
                        dr = json.load(f) or {}
                    for k, v in dr.items():
                        if k not in superinvestor:
                            superinvestor[k] = v
                            cache_fill_counts['superinvestor'] += 1
                except Exception:
                    logger.debug('Fehler beim Laden von dataroma_cache.json', exc_info=True)
    except Exception:
        logger.debug('Fehler beim Zugriff auf data/ Caches', exc_info=True)

    logger.info('Cache fills from disk: reddit=%d, x=%d, superinvestor=%d',
                cache_fill_counts['reddit'], cache_fill_counts['x'], cache_fill_counts['superinvestor'])

    # Fetch fundamentals in batch to reduce calls and parallelize
    from scripts.data_providers import fetch_fundamentals
    info_map = fetch_fundamentals(universe)

    # helper: tolerant lookup that handles BRK-B vs BRK.B and case differences
    def lookup_signal(mapping: dict, tk: str, default: float = 0.5):
        if not isinstance(mapping, dict):
            return default
        candidates = [tk, tk.replace('-', '.'), tk.replace('.', '-'), tk.upper()]
        for c in candidates:
            if c in mapping:
                return mapping[c]
        # try uppercase keys fallback
        up = {k.upper(): v for k, v in mapping.items()}
        for c in candidates:
            if c.upper() in up:
                return up[c.upper()]
        return default

    rows = []
    for tk in universe:
        try:
            info = info_map.get(tk, {})

            # 1. P/E vs history (via data_providers)
            from scripts.data_providers import get_pe_history_features, get_insider_summary, get_analyst_summary

            pe_feats = get_pe_history_features(tk)
            pe_vs_history_score = pe_feats.get('pe_score', 0.5)

            # 2. Insider summary (centralized provider)
            insider_summary = get_insider_summary(tk)
            insider_score = insider_summary.get('insider_score', 0.5)

            # 3. Analyst summary (centralized provider)
            analyst_summary = get_analyst_summary(tk)
            analyst_score = analyst_summary.get('analyst_score', 0.5)

            rows.append({
                'ticker': tk,
                'marketCap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'beta': info.get('beta'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
                'superinvestor_score': lookup_signal(superinvestor, tk, 0.5),
                'reddit_score': lookup_signal(reddit, tk, 0.5),
                'x_score': lookup_signal(x_sent, tk, 0.5),
                'ki_moat_score': lookup_signal(ki_moat, tk, 0.5),
                'pe_vs_history_score': pe_vs_history_score,
                'insider_score': insider_score,
                'analyst_score': analyst_score,
            })
        except Exception as e:
            logger.warning("Fehler beim Laden der Ticker-Daten für %s: %s", tk, e)
            
    return pd.DataFrame(rows)


def compute_scores(df, min_market_cap=MIN_MARKET_CAP):
    # Filtern auf Mindest-MarketCap und vorhandene P/E-Daten
    df = df[df['marketCap'] >= min_market_cap].dropna(subset=['trailingPE','forwardPE']).copy()

    # 1. Value Score
    # Lower P/E should give a higher value score -> invert the percentile rank
    df['value_score'] = (
        (1 - df['trailingPE'].rank(ascending=True, pct=True)) * 0.6 +
        (1 - df['forwardPE'].rank(ascending=True, pct=True)) * 0.4
    )

    # 2. Quality Score (KORRIGIERT: Umgang mit NaN in debtToEquity für Finanzwerte)
    median_debt = df['debtToEquity'].median()
    # Fülle fehlende Debt/Equity-Werte mit dem Median, um Ticker (wie JPM) nicht auszuschließen
    df['debtToEquity_fixed'] = df['debtToEquity'].fillna(median_debt) 

    df['quality_score'] = (
        df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 +
        (1 - df['debtToEquity_fixed'].rank(pct=True)) * 0.3
    )
    df = df.drop(columns=['debtToEquity_fixed'])


    # 3. Community Score (gleichgewichtet) — ensure NaNs become neutral 0.5
    for c in ['superinvestor_score', 'reddit_score', 'x_score']:
        if c not in df.columns:
            df[c] = 0.5
        else:
            df[c] = df[c].fillna(0.5)

    df['community_score'] = (
        df['superinvestor_score'] * 0.333 +
        df['reddit_score']        * 0.333 +
        df['x_score']             * 0.334
    )

    # 4. Final Score with dynamic community de-weighting if signals are sparse
    base_weights = {
        'value': 0.40,
        'quality': 0.20,
        'community': 0.20,
        'pe_vs_history': 0.08,
        'insider': 0.05,
        'analyst': 0.05,
        'ki_moat': 0.07,
    }

    # fraction of tickers missing (all neutral) community signals
    missing_community = ((df['superinvestor_score'] == 0.5) & (df['reddit_score'] == 0.5) & (df['x_score'] == 0.5)).mean()
    if missing_community > 0.30:
        # reduce community weight proportionally to missing fraction and reallocate to value
        new_community_w = base_weights['community'] * (1.0 - missing_community)
        delta = base_weights['community'] - new_community_w
        base_weights['community'] = new_community_w
        base_weights['value'] = base_weights['value'] + delta
        logger.info("Community signals sparse (%.0f%% missing) — reducing community weight to %.3f and increasing value weight to %.3f", missing_community*100, base_weights['community'], base_weights['value'])

    df['final_score'] = (
        df['value_score']           * base_weights['value'] +
        df['quality_score']         * base_weights['quality'] +
        df['community_score']       * base_weights['community'] +
        df['pe_vs_history_score']   * base_weights['pe_vs_history'] +
        df['insider_score']         * base_weights['insider'] +
        df['analyst_score']         * base_weights['analyst'] +
        df['ki_moat_score']         * base_weights['ki_moat']
    )

    return df.sort_values('final_score', ascending=False)

def main():
    parser = argparse.ArgumentParser(description="Erweitertes Value Portfolio Generator")
    parser.add_argument('--portfolio-size', type=int, default=PORTFOLIO_SIZE, help='Anzahl Aktien im Portfolio')
    parser.add_argument('--min-market-cap', type=float, default=MIN_MARKET_CAP, help='Minimale Marktkapitalisierung')
    args = parser.parse_args()

    logger.info("Starte erweitertes Value Portfolio (mit Insider, Analysten, P/E-History & KI-Moat)")
    universe = load_universe()
    logger.info("Universe: %d Aktien", len(universe))

    df = gather(universe)
    portfolio = compute_scores(df, min_market_cap=args.min_market_cap).head(args.portfolio_size)
    portfolio = construct_portfolio(portfolio, n=args.portfolio_size)
    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.csv")

    out = portfolio[['ticker','marketCap','sector','trailingPE','forwardPE','beta',
                     'returnOnEquity','debtToEquity','superinvestor_score',
                     'reddit_score','x_score','ki_moat_score',
                     'pe_vs_history_score','insider_score','analyst_score',
                     'final_score','weight_%']].copy()

    out['marketCap'] = (out['marketCap']/1e9).round(1).astype(str) + ' Mrd'
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['returnOnEquity'] = (out['returnOnEquity']*100).round(1).astype(str) + '%'
    out['debtToEquity'] = out['debtToEquity'].round(1)
    for col in ['superinvestor_score','reddit_score','x_score','ki_moat_score',
                'pe_vs_history_score','insider_score','analyst_score','final_score']:
        out[col] = out[col].round(3)

    logger.info("\n%s", out.to_string(index=False))
    logger.info("Portfolio-Beta: %.2f", portfolio['beta'].mean())
    logger.info("Durchschn. Forward P/E: %.1f", portfolio['forwardPE'].mean())

    save_portfolio(portfolio, filename)
    logger.info("Gespeichert → %s", filename)

if __name__ == '__main__':
    main()