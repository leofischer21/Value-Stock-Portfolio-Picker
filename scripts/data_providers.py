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
    # Sector and industry (string fields)
    out['sector'] = info.get('sector') or info.get('sectorName') or None
    out['industry'] = info.get('industry') or info.get('industryName') or None
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
    
    # Earnings Growth for PEG Ratio
    earnings_growth = _safe_float(info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth'))
    if earnings_growth is None:
        # Try earningsGrowthQuarterlyYoy
        earnings_growth = _safe_float(info.get('earningsGrowthQuarterlyYoy'))
    out['earningsGrowth'] = earnings_growth
    
    # Calculate PEG Ratio (Price/Earnings to Growth)
    if out['trailingPE'] is not None and earnings_growth is not None and earnings_growth > 0:
        out['pegRatio'] = out['trailingPE'] / (earnings_growth * 100)  # earnings_growth is often in decimal (0.15 = 15%)
    elif out['trailingPE'] is not None and earnings_growth is not None and earnings_growth > 1:
        # If earnings_growth is already in percent (e.g., 15 instead of 0.15)
        out['pegRatio'] = out['trailingPE'] / earnings_growth
    else:
        out['pegRatio'] = None
    
    # Free Cash Flow for P/FCF
    free_cashflow = _safe_float(info.get('freeCashflow') or info.get('freeCashFlow'))
    out['freeCashflow'] = free_cashflow
    
    # Calculate P/FCF (Price to Free Cash Flow)
    if out['currentPrice'] is not None and free_cashflow is not None:
        shares_outstanding = _safe_float(info.get('sharesOutstanding'))
        if shares_outstanding and shares_outstanding > 0:
            # P/FCF = Price per share / (Free Cash Flow per share)
            fcf_per_share = free_cashflow / shares_outstanding
            if fcf_per_share > 0:
                out['priceToFreeCashFlow'] = out['currentPrice'] / fcf_per_share
            else:
                out['priceToFreeCashFlow'] = None
        else:
            # Fallback: use marketCap if available
            if out['marketCap'] is not None and free_cashflow > 0:
                out['priceToFreeCashFlow'] = out['marketCap'] / free_cashflow
            else:
                out['priceToFreeCashFlow'] = None
    else:
        out['priceToFreeCashFlow'] = None

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

    # ROIC (Return on Invested Capital)
    roic = _safe_float(info.get('returnOnInvestedCapital') or info.get('roic'))
    if roic is None:
        # Fallback: Calculate from operatingIncome / (totalAssets - currentLiabilities)
        operating_income = _safe_float(info.get('operatingIncome') or info.get('operatingCashflow'))
        total_assets = _safe_float(info.get('totalAssets'))
        current_liabilities = _safe_float(info.get('currentLiabilities'))
        if operating_income is not None and total_assets is not None and current_liabilities is not None:
            invested_capital = total_assets - current_liabilities
            if invested_capital and invested_capital > 0:
                roic = operating_income / invested_capital
    if roic is not None:
        # Normalize percent format (>1 -> /100)
        if roic > 1:
            roic = roic / 100.0
        out['returnOnInvestedCapital'] = roic
    else:
        out['returnOnInvestedCapital'] = None

    # Profit Margins
    def _normalize_margin(margin_val):
        if margin_val is None:
            return None
        try:
            m = float(margin_val)
            # Normalize percent format (>1 -> /100)
            if m > 1:
                m = m / 100.0
            return m
        except Exception:
            return None

    gross_margin = _normalize_margin(info.get('grossMargins'))
    operating_margin = _normalize_margin(info.get('operatingMargins'))
    profit_margin = _normalize_margin(info.get('profitMargins'))
    
    out['grossMargins'] = gross_margin
    out['operatingMargins'] = operating_margin
    out['profitMargins'] = profit_margin
    
    # Calculate average margin from available margins
    margins = [m for m in [gross_margin, operating_margin, profit_margin] if m is not None]
    if margins:
        out['avgMargin'] = sum(margins) / len(margins)
    else:
        out['avgMargin'] = None

    # EV/EBITDA
    ev_ebitda = _safe_float(info.get('enterpriseToEbitda') or info.get('enterpriseValueMultiple'))
    if ev_ebitda is None:
        # Fallback: Calculate from enterpriseValue / ebitda
        enterprise_value = _safe_float(info.get('enterpriseValue'))
        ebitda = _safe_float(info.get('ebitda'))
        if enterprise_value is not None and ebitda is not None and ebitda > 0:
            ev_ebitda = enterprise_value / ebitda
    out['enterpriseToEbitda'] = ev_ebitda

    return out


def _fetch_single(tk: str) -> tuple[str, dict]:
    try:
        t = yf.Ticker(tk)
        info = _safe_info(t)
        norm = _normalize_info(info)
        
        # Calculate 12M Price Momentum
        try:
            hist = t.history(period="1y")[['Close']].dropna()
            if not hist.empty and len(hist) > 0:
                current_price = norm.get('currentPrice') or (hist['Close'].iloc[-1] if not hist.empty else None)
                price_12m_ago = hist['Close'].iloc[0] if len(hist) > 0 else None
                if current_price is not None and price_12m_ago is not None and price_12m_ago > 0:
                    momentum = ((current_price / price_12m_ago) - 1) * 100  # in percent
                    norm['priceMomentum12M'] = momentum
                else:
                    norm['priceMomentum12M'] = None
            else:
                norm['priceMomentum12M'] = None
        except Exception:
            norm['priceMomentum12M'] = None
        
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

from cache import get as cache_get, set as cache_set
from http_utils import get_text


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
