import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _safe_info(ticker_obj) -> dict:
    try:
        info = ticker_obj.info or {}
    except Exception:
        try:
            # Some yfinance versions provide fast_info
            info = getattr(ticker_obj, 'fast_info', {}) or {}
        except Exception:
            info = {}
    return info or {}


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _normalize_info(info: dict) -> dict:
    # Standardize common fields and provide P/E fallbacks where possible
    out = {}
    out['marketCap'] = _safe_float(info.get('marketCap'))
    out['trailingPE'] = _safe_float(info.get('trailingPE'))
    out['forwardPE'] = _safe_float(info.get('forwardPE'))
    out['beta'] = _safe_float(info.get('beta'))
    # returnOnEquity sometimes in percent (e.g. 15) or fraction (0.15)
    roe = info.get('returnOnEquity')
    if roe is None:
        # try alternate keys
        roe = info.get('returnOnEquity')
    if roe is not None:
        try:
            roe_f = float(roe)
            # If looks like percent >1, convert
            if roe_f > 1:
                roe_f = roe_f / 100.0
            out['returnOnEquity'] = roe_f
        except Exception:
            out['returnOnEquity'] = None
    else:
        out['returnOnEquity'] = None

    out['debtToEquity'] = _safe_float(info.get('debtToEquity'))
    out['recommendationMean'] = _safe_float(info.get('recommendationMean'))
    out['currentPrice'] = _safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
    # try trailingEps keys
    trailing_eps = _safe_float(info.get('trailingEps') or info.get('epsTrailingTwelveMonths') or info.get('EPS'))
    forward_eps = _safe_float(info.get('forwardEps') or info.get('forwardEps') or info.get('epsForward'))
    out['trailingEps'] = trailing_eps
    out['forwardEps'] = forward_eps

    # Fallback computation: if trailingPE missing but we have price & trailing_eps
    if out['trailingPE'] is None and out['currentPrice'] is not None and trailing_eps:
        try:
            if trailing_eps != 0:
                out['trailingPE'] = out['currentPrice'] / trailing_eps
        except Exception:
            out['trailingPE'] = None

    # forwardPE fallback
    if out['forwardPE'] is None and out['currentPrice'] is not None and forward_eps:
        try:
            if forward_eps != 0:
                out['forwardPE'] = out['currentPrice'] / forward_eps
        except Exception:
            out['forwardPE'] = None

    return out


def _fetch_single(tk: str) -> tuple[str, dict]:
    try:
        t = yf.Ticker(tk)
        info = _safe_info(t)
        norm = _normalize_info(info)
        return tk, norm
    except Exception as e:
        logger.debug("Fehler beim Laden von %s: %s", tk, e)
        return tk, {}


def fetch_fundamentals(universe: List[str], max_workers: int = 8) -> Dict[str, Dict[str, Any]]:
    """Fetch key fundamental fields for a list of tickers in parallel (best-effort).

    Returns a mapping ticker -> normalized info dict (may be empty dict on failure).
    """
    if not universe:
        return {}

    workers = min(max_workers, max(1, len(universe)))
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_single, tk): tk for tk in universe}
        for fut in as_completed(futures):
            tk = futures[fut]
            try:
                _, info = fut.result()
                results[tk] = info or {}
            except Exception as e:
                logger.debug("Exception fetching %s: %s", tk, e)
                results[tk] = {}
    return results


from datetime import datetime, timedelta

from src.cache import get as cache_get, set as cache_set
from src.http import get_text


def get_pe_history_features(ticker: str, cache_ttl: int = 7 * 24 * 3600) -> Dict[str, Any]:
    """Estimate P/E related features by comparing current P/E to price lows over 2y and 5y.

    Uses yfinance historical prices and the normalized trailing EPS from ticker info.
    Results are cached to avoid repeated downloads.
    Returns dict with keys: pe_current, pe_low_2y, pe_low_5y, pe_score
    """
    key = f"pe_history:{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    out: Dict[str, Any] = {
        'pe_current': None,
        'pe_low_2y': None,
        'pe_low_5y': None,
        'pe_score': 0.5,
    }
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y")[['Close']].dropna()
        if hist.empty:
            cache_set(key, out, ttl_seconds=cache_ttl)
            return out

        now = datetime.utcnow()
        two_years_ago = now - timedelta(days=365 * 2)
        close_2y = hist[hist.index >= two_years_ago]
        close_5y = hist

        min_2y = float(close_2y['Close'].min()) if not close_2y.empty else float('nan')
        min_5y = float(close_5y['Close'].min()) if not close_5y.empty else float('nan')

        info = _normalize_info(_safe_info(t))
        trailing_eps = info.get('trailingEps')
        current_price = info.get('currentPrice') or (hist['Close'].iloc[-1] if not hist.empty else None)

        if trailing_eps and trailing_eps != 0 and current_price:
            pe_current = current_price / trailing_eps
            pe_low_2y = min_2y / trailing_eps if not (min_2y != min_2y) else None
            pe_low_5y = min_5y / trailing_eps if not (min_5y != min_5y) else None

            # Score relative to lows: closer to low -> higher score
            def _score_vs_low(curr, low):
                try:
                    if curr is None or low is None or low <= 0:
                        return 0.5
                    diff = max(0.0, curr - low)
                    denom = max(curr, low, 1e-6)
                    frac = diff / denom
                    scr = max(0.0, min(1.0, 1.0 - frac))
                    return scr
                except Exception:
                    return 0.5

            s2 = _score_vs_low(pe_current, pe_low_2y)
            s5 = _score_vs_low(pe_current, pe_low_5y)
            # weight shorter horizon slightly more
            pe_score = 0.6 * s2 + 0.4 * s5

            out.update({'pe_current': pe_current, 'pe_low_2y': pe_low_2y, 'pe_low_5y': pe_low_5y, 'pe_score': pe_score})
        else:
            out['pe_score'] = 0.5

    except Exception:
        out['pe_score'] = 0.5

    cache_set(key, out, ttl_seconds=cache_ttl)
    return out


def get_insider_summary(ticker: str, cache_ttl: int = 24 * 3600) -> Dict[str, Any]:
    """Scrape OpenInsider for quick heuristics about recent insider buys.

    Returns dict: insider_score (0-1), recent_buys (int), senior_buy (bool)
    """
    key = f"insider:{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    out = {'insider_score': 0.0, 'recent_buys': 0, 'senior_buy': False}
    try:
        url = f"http://openinsider.com/search?q={ticker}"
        txt = get_text(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        if not txt:
            cache_set(key, out, ttl_seconds=cache_ttl)
            return out

        u = txt.upper()
        # Simple heuristics
        recent_buys = u.count('OPEN MARKET PURCHASE') + u.count('P - ')
        senior_titles = ['CEO', 'CFO', 'PRESIDENT', 'COO', 'DIRECTOR']
        senior_buy = any(title in u and 'PURCHASE' in u for title in senior_titles)

        score = 0.0
        if recent_buys > 0:
            score = 0.6 if recent_buys == 1 else min(1.0, 0.6 + 0.1 * (recent_buys - 1))
        if senior_buy:
            score = max(score, 0.8)

        out.update({'insider_score': float(score), 'recent_buys': int(recent_buys), 'senior_buy': bool(senior_buy)})
    except Exception:
        out = {'insider_score': 0.0, 'recent_buys': 0, 'senior_buy': False}

    cache_set(key, out, ttl_seconds=cache_ttl)
    return out


def get_analyst_summary(ticker: str, cache_ttl: int = 12 * 3600) -> Dict[str, Any]:
    """Return analyst-based signals: recommendation mean and target delta.

    Returns dict: analyst_score (0-1), recommendationMean, targetMeanPrice, target_delta
    """
    key = f"analyst:{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    out = {'analyst_score': 0.5, 'recommendationMean': None, 'targetMeanPrice': None, 'target_delta': None}
    try:
        t = yf.Ticker(ticker)
        info = _normalize_info(_safe_info(t))
        rec = info.get('recommendationMean')
        target = _safe_float(_safe_info(t).get('targetMeanPrice'))
        current = info.get('currentPrice')

        rec_comp = 0.5
        if rec is not None:
            rec_comp = max(0.0, min(1.0, (5.0 - rec) / 4.0))

        target_delta = None
        target_comp = 0.5
        if target and current:
            try:
                td = (target - current) / max(current, 1e-6)
                target_delta = td
                # Positive delta increases score up to 100% -> map to 0.0-1.0
                target_comp = max(0.0, min(1.0, (td + 1.0) / 2.0))
            except Exception:
                target_comp = 0.5

        analyst_score = 0.6 * rec_comp + 0.4 * target_comp
        out.update({'analyst_score': float(analyst_score), 'recommendationMean': rec, 'targetMeanPrice': target, 'target_delta': target_delta})
    except Exception:
        out = {'analyst_score': 0.5, 'recommendationMean': None, 'targetMeanPrice': None, 'target_delta': None}

    cache_set(key, out, ttl_seconds=cache_ttl)
    return out
