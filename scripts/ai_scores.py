# ai_scores.py
"""
KI-basierte Bewertungen für Moat, Quality und Performance-Vorhersagen.
Nutzt LLM-APIs (OpenAI/Anthropic/Groq) mit Fallback auf heuristische Berechnungen.
"""
import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path

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
        load_dotenv(env_path, override=True)  # override=True überschreibt vorhandene env vars
except ImportError:
    pass  # python-dotenv nicht installiert, verwende nur env vars
except Exception:
    pass  # Fehler beim Laden ignorieren

# Logger initialisieren (nach dotenv import)
logger = logging.getLogger(__name__)

# API-Konfiguration (aus Umgebungsvariablen)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")

# Log welche API verwendet wird (nur beim ersten Import, wenn logging konfiguriert ist)
if GROQ_API_KEY:
    try:
        logger.info("Groq API key found - will use Groq for AI scores")
    except:
        pass
elif OPENAI_API_KEY:
    try:
        logger.info("OpenAI API key found - will use OpenAI for AI scores")
    except:
        pass
elif ANTHROPIC_API_KEY:
    try:
        logger.info("Anthropic API key found - will use Anthropic for AI scores")
    except:
        pass
else:
    try:
        logger.warning("No LLM API key found - AI scores will use heuristic fallback")
    except:
        pass

# Cache für API-Antworten
from cache import get as cache_get, set as cache_set

# Developer Tier: Bis zu 500 RPM - keine Rate-Limiting-Delays mehr nötig!
import time


def _get_llm_response(prompt: str, ticker: str, cache_key_suffix: str) -> Optional[str]:
    """
    Ruft LLM-API auf (Groq/OpenAI/Anthropic) mit Caching.
    Priorität: Groq (kostenlos, schnell) > OpenAI > Anthropic
    Returns: JSON-String mit Antwort oder None bei Fehler
    
    WICHTIG: Das Rate-Limiting-Delay wird in der aufrufenden Schleife (monthly_update.py) 
    gehandhabt, nicht hier. Diese Funktion prüft nur den Cache und macht API-Calls.
    """
    cache_key = f"ai_score:{ticker}:{cache_key_suffix}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    
    try:
        # Versuche Groq zuerst (kostenlos, schnell)
        if GROQ_API_KEY:
            try:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                # Verwende das in AI_MODEL spezifizierte Modell oder Standard
                # Verfügbare Modelle: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
                if "llama" in AI_MODEL.lower() or "mixtral" in AI_MODEL.lower():
                    model = AI_MODEL
                    # Automatisch 3.1 auf 3.3 aktualisieren (3.1 ist dekommissioniert)
                    if "3.1" in model and "70b" in model:
                        model = model.replace("3.1", "3.3")
                        logger.debug(f"Updated deprecated model {AI_MODEL} to {model}")
                else:
                    model = "llama-3.3-70b-versatile"  # Aktuelles Standard-Modell
                
                # Developer Tier: Bis zu 500 RPM - keine Delays mehr nötig!
                # Einfache Retry-Logik für seltene Rate-Limits
                max_retries = 3
                retry_delay = 1.0  # Kurzes Delay nur bei Retries
                
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a financial analyst expert. Respond only with valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        result = response.choices[0].message.content
                        cache_set(cache_key, result, ttl_seconds=30 * 24 * 3600)  # Cache 30 Tage
                        logger.debug(f"Groq API success for {ticker}")
                        return result
                    except Exception as api_error:
                        error_str = str(api_error).lower()
                        # Prüfe auf Rate-Limiting (429) - sollte selten sein mit Developer Tier
                        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                            if attempt < max_retries - 1:
                                # Kurzes Delay nur bei Retries (Developer Tier hat hohe Limits)
                                wait_time = retry_delay * (attempt + 1)  # 1s, 2s, 3s
                                logger.debug(f"Groq rate limit hit for {ticker}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                                time.sleep(wait_time)
                                continue
                            else:
                                logger.warning(f"Groq rate limit exceeded for {ticker} after {max_retries} attempts")
                                raise
                        else:
                            # Andere Fehler: direkt weiterwerfen
                            raise
                
            except ImportError:
                logger.warning("groq package not installed. Install with: pip install groq")
            except Exception as e:
                logger.debug(f"Groq API error for {ticker}: {e}")
                # Try to use fallback model if model name is wrong
                if "model" in str(e).lower() or "decommissioned" in str(e).lower():
                    try:
                        # Rate-Limiting auch für Fallback
                        _wait_for_rate_limit()
                        # Try with current model
                        model = "llama-3.3-70b-versatile"
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a financial analyst expert. Respond only with valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        result = response.choices[0].message.content
                        cache_set(cache_key, result, ttl_seconds=30 * 24 * 3600)
                        logger.debug(f"Groq API success for {ticker} (with fallback model)")
                        return result
                    except Exception as e2:
                        logger.debug(f"Groq API error with fallback model for {ticker}: {e2}")
        
        # Versuche OpenAI
        if OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                # Rate-Limiting auch für OpenAI (konservativ)
                _wait_for_rate_limit()
                response = client.chat.completions.create(
                    model=AI_MODEL if "gpt" in AI_MODEL.lower() else "gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst expert. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                result = response.choices[0].message.content
                cache_set(cache_key, result, ttl_seconds=30 * 24 * 3600)  # Cache 30 Tage
                logger.debug(f"OpenAI API success for {ticker}")
                return result
            except ImportError:
                logger.warning("openai package not installed")
            except Exception as e:
                logger.debug(f"OpenAI API error for {ticker}: {e}")
        
        # Versuche Anthropic
        if ANTHROPIC_API_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                # Rate-Limiting auch für Anthropic
                _wait_for_rate_limit()
                response = client.messages.create(
                    model=AI_MODEL if "claude" in AI_MODEL.lower() else "claude-3-opus-20240229",
                    max_tokens=500,
                    temperature=0.3,
                    system="You are a financial analyst expert. Respond only with valid JSON.",
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
                cache_set(cache_key, result, ttl_seconds=30 * 24 * 3600)
                logger.debug(f"Anthropic API success for {ticker}")
                return result
            except ImportError:
                logger.warning("anthropic package not installed")
            except Exception as e:
                logger.debug(f"Anthropic API error: {e}")
        
        logger.warning(f"No LLM API key found. Using heuristic fallback for {ticker}")
        return None
    except Exception as e:
        logger.error(f"LLM API error for {ticker}: {e}")
        return None


def _heuristic_moat_score(financial_data: dict, scores: dict) -> float:
    """
    Heuristische Moat-Bewertung basierend auf Finanzdaten.
    Returns: Score 0.0-1.0
    """
    score = 0.5  # Base score
    
    # High margins indicate moat
    margins = financial_data.get('grossMargins', 0) or financial_data.get('operatingMargins', 0) or financial_data.get('profitMargins', 0) or 0
    if margins > 0.3:
        score += 0.15
    elif margins > 0.2:
        score += 0.10
    elif margins > 0.1:
        score += 0.05
    
    # High ROE indicates quality/moat
    roe = financial_data.get('returnOnEquity', 0) or 0
    if roe > 0.2:
        score += 0.15
    elif roe > 0.15:
        score += 0.10
    elif roe > 0.1:
        score += 0.05
    
    # Low debt indicates financial strength
    debt_to_equity = financial_data.get('debtToEquity', 1.0) or 1.0
    if debt_to_equity < 0.5:
        score += 0.10
    elif debt_to_equity < 1.0:
        score += 0.05
    
    # Community sentiment as proxy
    community = scores.get('community_score', 0.5) or 0.5
    score += (community - 0.5) * 0.1
    
    return max(0.0, min(1.0, score))


def _heuristic_quality_score(financial_data: dict, scores: dict) -> float:
    """
    Heuristische Quality-Bewertung basierend auf Finanzdaten.
    Returns: Score 0.0-1.0
    """
    score = 0.5  # Base score
    
    # ROE
    roe = financial_data.get('returnOnEquity', 0) or 0
    if roe > 0.2:
        score += 0.20
    elif roe > 0.15:
        score += 0.15
    elif roe > 0.1:
        score += 0.10
    
    # ROIC
    roic = financial_data.get('returnOnInvestedCapital', 0) or 0
    if roic > 0.15:
        score += 0.15
    elif roic > 0.10:
        score += 0.10
    
    # Margins
    avg_margin = (
        (financial_data.get('grossMargins', 0) or 0) +
        (financial_data.get('operatingMargins', 0) or 0) +
        (financial_data.get('profitMargins', 0) or 0)
    ) / 3
    if avg_margin > 0.25:
        score += 0.10
    elif avg_margin > 0.15:
        score += 0.05
    
    # Low debt
    debt_to_equity = financial_data.get('debtToEquity', 1.0) or 1.0
    if debt_to_equity < 0.5:
        score += 0.10
    elif debt_to_equity < 1.0:
        score += 0.05
    
    return max(0.0, min(1.0, score))


def _heuristic_performance_prediction(financial_data: dict, scores: dict) -> Dict[str, float]:
    """
    Heuristische Performance-Vorhersage basierend auf Finanzdaten.
    Returns: Dict mit CAGR für 1, 2, 5, 10 Jahre (in Prozent)
    """
    base_cagr = 8.0  # Market average
    
    # Adjust based on ROE (more granular)
    roe = financial_data.get('returnOnEquity', 0) or 0
    if roe > 0.25:
        base_cagr += 7.0
    elif roe > 0.20:
        base_cagr += 5.0
    elif roe > 0.15:
        base_cagr += 3.0
    elif roe > 0.10:
        base_cagr += 1.5
    elif roe < 0.05:
        base_cagr -= 2.0
    
    # Adjust based on ROIC (additional quality metric)
    roic = financial_data.get('returnOnInvestedCapital', 0) or 0
    if roic > 0.15:
        base_cagr += 2.0
    elif roic > 0.10:
        base_cagr += 1.0
    elif roic < 0.05:
        base_cagr -= 1.0
    
    # Adjust based on value (lower P/E = higher expected return, more granular)
    pe = financial_data.get('trailingPE', 20) or 20
    if pe < 12:
        base_cagr += 5.0
    elif pe < 15:
        base_cagr += 3.5
    elif pe < 20:
        base_cagr += 2.0
    elif pe < 25:
        base_cagr += 0.5
    elif pe > 35:
        base_cagr -= 3.0
    elif pe > 30:
        base_cagr -= 1.5
    
    # Adjust based on PEG (growth at reasonable price)
    peg = financial_data.get('pegRatio', 2.0) or 2.0
    if peg is not None and peg > 0:
        if peg < 1.0:
            base_cagr += 3.0
        elif peg < 1.5:
            base_cagr += 2.0
        elif peg < 2.0:
            base_cagr += 1.0
        elif peg > 3.0:
            base_cagr -= 2.0
    
    # Adjust based on margins (profitability)
    avg_margin = (
        (financial_data.get('grossMargins', 0) or 0) +
        (financial_data.get('operatingMargins', 0) or 0) +
        (financial_data.get('profitMargins', 0) or 0)
    ) / 3
    if avg_margin > 0.30:
        base_cagr += 2.5
    elif avg_margin > 0.20:
        base_cagr += 1.5
    elif avg_margin > 0.10:
        base_cagr += 0.5
    elif avg_margin < 0.05:
        base_cagr -= 1.5
    
    # Adjust based on quality score (more impact)
    quality = scores.get('quality_score', 0.5) or 0.5
    base_cagr += (quality - 0.5) * 12.0  # Increased from 10.0
    
    # Adjust based on value score if available
    value_score = scores.get('value_score', 0.5) or 0.5
    base_cagr += (value_score - 0.5) * 8.0
    
    # Add some variation based on sector (realistic variation)
    sector = financial_data.get('sector', 'Unknown')
    sector_adjustments = {
        'Technology': 1.5,
        'Healthcare': 1.0,
        'Financial Services': 0.5,
        'Consumer Cyclical': 0.0,
        'Consumer Defensive': -0.5,
        'Energy': -1.0,
        'Utilities': -1.5,
    }
    base_cagr += sector_adjustments.get(sector, 0.0)
    
    # Time horizon adjustments (longer = more uncertainty, but also more compounding)
    return {
        'cagr_1y': max(0.0, min(30.0, base_cagr + 2.0)),  # Short term: more optimistic, cap at 30%
        'cagr_2y': max(0.0, min(25.0, base_cagr + 1.0)),
        'cagr_5y': max(0.0, min(20.0, base_cagr)),  # Medium term
        'cagr_10y': max(0.0, min(18.0, base_cagr - 0.5))  # Long term: slight reduction but still good
    }


def get_ai_moat_score(ticker: str, financial_data: dict, scores: dict) -> float:
    """
    Bewertet den Moat (Wettbewerbsvorteil) einer Aktie.
    Returns: Score 0.0-1.0
    """
    try:
        # Build prompt
        prompt = f"""Evaluate the competitive moat (sustainable competitive advantage) of {ticker} stock.

Financial Data:
- P/E Ratio: {financial_data.get('trailingPE', 'N/A')}
- Gross Margin: {financial_data.get('grossMargins', 'N/A')}
- Operating Margin: {financial_data.get('operatingMargins', 'N/A')}
- ROE: {financial_data.get('returnOnEquity', 'N/A')}
- Debt/Equity: {financial_data.get('debtToEquity', 'N/A')}
- Sector: {financial_data.get('sector', 'Unknown')}

Respond with JSON: {{"moat_score": 0.0-1.0, "reasoning": "brief explanation"}}
Score 1.0 = exceptional moat, 0.5 = average, 0.0 = weak/no moat."""

        response = _get_llm_response(prompt, ticker, "moat")
        
        if response:
            try:
                result = json.loads(response)
                score = float(result.get('moat_score', 0.5))
                return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.debug(f"Failed to parse LLM response for {ticker} moat")
        
        # Fallback to heuristic
        return _heuristic_moat_score(financial_data, scores)
        
    except Exception as e:
        logger.error(f"Error calculating moat score for {ticker}: {e}")
        return _heuristic_moat_score(financial_data, scores)


def get_ai_quality_score(ticker: str, financial_data: dict, scores: dict) -> float:
    """
    Bewertet die Qualität des Unternehmens.
    Returns: Score 0.0-1.0
    """
    try:
        # Build prompt
        prompt = f"""Evaluate the overall business quality of {ticker} stock.

Financial Data:
- ROE: {financial_data.get('returnOnEquity', 'N/A')}
- ROIC: {financial_data.get('returnOnInvestedCapital', 'N/A')}
- Gross Margin: {financial_data.get('grossMargins', 'N/A')}
- Operating Margin: {financial_data.get('operatingMargins', 'N/A')}
- Profit Margin: {financial_data.get('profitMargins', 'N/A')}
- Debt/Equity: {financial_data.get('debtToEquity', 'N/A')}
- Sector: {financial_data.get('sector', 'Unknown')}

Respond with JSON: {{"quality_score": 0.0-1.0, "reasoning": "brief explanation"}}
Score 1.0 = exceptional quality, 0.5 = average, 0.0 = poor quality."""

        response = _get_llm_response(prompt, ticker, "quality")
        
        if response:
            try:
                result = json.loads(response)
                score = float(result.get('quality_score', 0.5))
                return max(0.0, min(1.0, score))
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.debug(f"Failed to parse LLM response for {ticker} quality")
        
        # Fallback to heuristic
        return _heuristic_quality_score(financial_data, scores)
        
    except Exception as e:
        logger.error(f"Error calculating quality score for {ticker}: {e}")
        return _heuristic_quality_score(financial_data, scores)


def get_ai_predicted_performance(ticker: str, financial_data: dict, scores: dict) -> Dict[str, float]:
    """
    Vorhersage der Performance für 1, 2, 5, 10 Jahre.
    Returns: Dict mit CAGR für 1, 2, 5, 10 Jahre (in Prozent, z.B. 15.5 für 15.5%)
    """
    try:
        # Build prompt
        prompt = f"""Predict the expected annual return (CAGR) for {ticker} stock over different time horizons.

Financial Data:
- Current P/E: {financial_data.get('trailingPE', 'N/A')}
- Forward P/E: {financial_data.get('forwardPE', 'N/A')}
- ROE: {financial_data.get('returnOnEquity', 'N/A')}
- ROIC: {financial_data.get('returnOnInvestedCapital', 'N/A')}
- Margins: Gross {financial_data.get('grossMargins', 'N/A')}, Operating {financial_data.get('operatingMargins', 'N/A')}
- Sector: {financial_data.get('sector', 'Unknown')}
- Beta: {financial_data.get('beta', 'N/A')}

Respond with JSON: {{"cagr_1y": X.X, "cagr_2y": X.X, "cagr_5y": X.X, "cagr_10y": X.X, "reasoning": "brief explanation"}}
Values in percentage (e.g., 15.5 for 15.5% annual return). Be realistic and conservative."""

        response = _get_llm_response(prompt, ticker, "performance")
        
        if response:
            try:
                result = json.loads(response)
                return {
                    'cagr_1y': max(0.0, float(result.get('cagr_1y', 8.0))),
                    'cagr_2y': max(0.0, float(result.get('cagr_2y', 8.0))),
                    'cagr_5y': max(0.0, float(result.get('cagr_5y', 8.0))),
                    'cagr_10y': max(0.0, float(result.get('cagr_10y', 8.0)))
                }
            except (json.JSONDecodeError, ValueError, KeyError):
                logger.debug(f"Failed to parse LLM response for {ticker} performance")
        
        # Fallback to heuristic
        return _heuristic_performance_prediction(financial_data, scores)
        
    except Exception as e:
        logger.error(f"Error calculating performance prediction for {ticker}: {e}")
        return _heuristic_performance_prediction(financial_data, scores)

