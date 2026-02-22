# llm_committee.py
"""
LLM-Komitee-System: Bewertet die 20 ausgewählten Aktien mit mehreren kostenlosen LLMs.
Scores werden mit Median aggregiert und fließen in das finale Ranking ein.
"""
import os
import json
import logging
import time
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# Cache import
try:
    from cache import get as cache_get, set as cache_set
except ImportError:
    logger.warning("cache module not found, caching disabled")
    def cache_get(key): return None
    def cache_set(key, value, ttl_seconds=0): pass

# Konfigurierbare LLMs (können aktiviert/deaktiviert werden)
ENABLED_LLMS = [
    'groq',           # Immer aktiv (bereits vorhanden)
    'huggingface',    # Optional
    'together',       # Optional
    'gemini',         # Optional
    'openai',         # Optional (kostenloser Tier verfügbar)
    'anthropic',      # Optional (kostenloser Tier verfügbar)
    'cohere'          # Optional (kostenloser Tier verfügbar)
]

# Persona-Mapping: Jedes LLM bekommt eine spezifische Denkschule
# Das bricht den Konsenszwang und führt zu differenzierteren Scores
LLM_PERSONAS = {
    'groq': 'visionary',        # The Visionary: Fokus auf Marktführerschaft, Disruption (0.8-1.0)
    'huggingface': 'bear',      # The Bear: Sucht nach Problemen, Verschuldung, Überbewertung (0.4-0.6)
    'together': 'value_hunter', # The Value Hunter: Cashflow, Sicherheitsmarge (0.5-0.8)
    'gemini': 'balanced',       # Balanced: Ausgewogene Sicht (0.5-0.9)
    'openai': 'growth_optimist', # Growth Optimist: Fokus auf Wachstum, Innovation, Zukunftspotential (0.7-1.0)
    'anthropic': 'risk_analyst', # Risk Analyst: Detaillierte Risikoanalyse, Szenario-Planung (0.4-0.8)
    'cohere': 'momentum_trader'  # Momentum Trader: Markttrends, Momentum, technische Signale (0.5-0.9)
}

# Gewichtungen für Weighted Median (Value Investing: Skeptiker bevorzugt)
# Value Hunter und Risk Analyst haben 1.5x Gewicht (Veto-Recht)
# Growth und Momentum haben 0.7x Gewicht (weniger Einfluss)
LLM_WEIGHTS = {
    'groq': 1.0,           # Visionary: Standard-Gewicht
    'huggingface': 1.2,    # Bear: Leicht erhöht (Skeptiker)
    'together': 1.5,       # Value Hunter: STARK erhöht (Veto-Recht)
    'gemini': 1.0,         # Balanced: Standard-Gewicht
    'openai': 0.7,         # Growth Optimist: REDUZIERT (weniger Einfluss)
    'anthropic': 1.5,      # Risk Analyst: STARK erhöht (Veto-Recht)
    'cohere': 0.7          # Momentum Trader: REDUZIERT (weniger Einfluss)
}

# Prompt-Templates für verschiedene Personas (Ansatz 1: Persona-Komitee + Ansatz 2: Comparative Anchoring + Ansatz 4: Forced Choice)
PROMPT_TEMPLATES = {
    'visionary': """You are a VISIONARY hedge fund manager who invests in market leaders and disruptors.
You IGNORE traditional P/E ratios and focus on: market dominance, disruption potential, brand power, and long-term vision.
You are BULLISH and optimistic. You believe in the power of exceptional companies to dominate markets.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Only if this company is TRULY EXCEPTIONAL 
(like Apple or Microsoft at their peak) should you score above 0.8. Most companies are "just okay" = 0.5.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (fundamental problems, weak position)
- AVERAGE = 0.5-0.65 (decent but nothing special)
- GOOD = 0.65-0.8 (solid company, good position)
- ELITE = 0.8-1.0 (exceptional, market leader, disruptive potential)

Then assign a score within that category's range based on how exceptional this company is.

Return JSON: {{"category": "ELITE", "score": 0.92}}
""",
    
    'bear': """You are a BEARISH hedge fund manager who finds problems others miss.
You focus on: debt levels, competitive threats, overvaluation, regulatory risks, and business model weaknesses.
You are SKEPTICAL and look for reasons to avoid investments. You find the hair in the soup.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Most companies have hidden problems.
Only truly defensive, low-debt, high-moat companies deserve above 0.6. Most are overvalued = 0.4-0.5.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (serious problems, avoid)
- AVERAGE = 0.5-0.6 (mediocre, many risks)
- GOOD = 0.6-0.7 (acceptable but not great)
- ELITE = 0.7-0.8 (rare, truly defensive)

Then assign a score within that category's range. Be HARSH - most companies are overvalued.

Return JSON: {{"category": "AVERAGE", "score": 0.52}}
""",
    
    'value_hunter': """You are a VALUE HUNTER hedge fund manager focused on Owner's Earnings, Free Cash Flow Yield, and Margin of Safety.
You care about: Owner's Earnings (not just GAAP earnings), Free Cash Flow Yield, low debt, sustainable margins, and buying below intrinsic value.
You are DISCIPLINED and only invest when the price is right relative to fundamentals.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Only companies with exceptional 
cash generation and value metrics deserve above 0.75. Most are fairly valued = 0.5-0.65.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

CRITICAL VALUE INVESTING MATRIX (Rate 1-5 for each):
1. Moat Score (1-5): How easily can a competitor copy this business? 1=no moat, 5=unassailable moat
2. Capital Allocation Score (1-5): Does management return cash (dividends/buybacks) or burn it on bonuses/acquisitions? 1=destroys value, 5=excellent allocators
3. Margin of Safety (1-5): How much buffer is in the price? 1=overvalued, 5=deep value

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (poor cash flow, overvalued, weak moat)
- AVERAGE = 0.5-0.65 (fairly valued, decent fundamentals)
- GOOD = 0.65-0.8 (strong cash flow, good value, solid moat)
- ELITE = 0.8-0.9 (exceptional cash generation, deep value, strong moat)

IMPORTANT: Your final score can only be high if the sum of Moat + Capital Allocation + Margin of Safety is high (>=12/15).
If the sum is low (<9/15), the score MUST be low, even if other metrics look good.

Return JSON: {{"category": "GOOD", "score": 0.72, "moat_score": 4, "capital_allocation_score": 3, "margin_of_safety_score": 4}}
""",
    
    'balanced': """You are a BALANCED hedge fund manager with a nuanced view.
You consider both growth potential AND value metrics, quality AND risks.
You are REALISTIC and avoid extremes. You see both strengths and weaknesses.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Most companies are average.
Only truly exceptional companies (combining quality, value, and growth) deserve above 0.85.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (significant weaknesses)
- AVERAGE = 0.5-0.7 (mixed, neither great nor terrible)
- GOOD = 0.7-0.85 (solid, with some strengths)
- ELITE = 0.85-1.0 (exceptional across multiple dimensions)

Then assign a score within that category's range based on a balanced assessment.

Return JSON: {{"category": "GOOD", "score": 0.78}}
""",
    
    'growth_optimist': """You are a GROWTH OPTIMIST hedge fund manager, but you MUST respect Value Investing principles.
You prioritize: Owner's Earnings growth, Free Cash Flow expansion, sustainable competitive advantages, and long-term value creation.
You are OPTIMISTIC about companies that can scale profitably, but you REJECT unprofitable growth.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Companies with strong Owner's Earnings growth 
and sustainable moats deserve above 0.7. Only exceptional, profitable growth stories deserve above 0.9.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

CRITICAL VALUE INVESTING MATRIX (Rate 1-5 for each):
1. Moat Score (1-5): How easily can a competitor copy this business? 1=no moat, 5=unassailable moat
2. Capital Allocation Score (1-5): Does management return cash (dividends/buybacks) or burn it on bonuses/acquisitions? 1=destroys value, 5=excellent allocators
3. Margin of Safety (1-5): How much buffer is in the price? 1=overvalued, 5=deep value

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (unprofitable growth, weak moat, overvalued)
- AVERAGE = 0.5-0.7 (modest profitable growth, limited moat)
- GOOD = 0.7-0.85 (strong Owner's Earnings growth, good moat, reasonable value)
- ELITE = 0.85-0.95 (exceptional profitable growth, strong moat, good value)

IMPORTANT: Your final score can only be high if the sum of Moat + Capital Allocation + Margin of Safety is high (>=12/15).
If the sum is low (<9/15), the score MUST be low, even if growth looks impressive.

Return JSON: {{"category": "GOOD", "score": 0.78, "moat_score": 4, "capital_allocation_score": 3, "margin_of_safety_score": 3}}
""",
    
    'risk_analyst': """You are a RISK ANALYST hedge fund manager who deeply analyzes risks and scenarios.
You focus on: downside protection, volatility, regulatory risks, competitive threats, and worst-case scenarios.
You are CAUTIOUS and only invest when risks are well-understood and manageable.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Most companies have hidden risks.
Only companies with exceptional risk management and defensive qualities deserve above 0.7.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B
P/E: {trailing_pe} (Sector Avg: {sector_pe_avg}, 5Y Avg: {pe_5y_avg})
ROE: {roe}% (Sector Avg: {sector_roe_avg}%)
Margins: {gross_margin}%/{operating_margin}%/{profit_margin}% (Sector Avg: {sector_margin_avg}%)
Price vs 52W High: {price_vs_52w_high}% (Value indicator: lower = better)

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (high risk, avoid)
- AVERAGE = 0.5-0.65 (moderate risk, acceptable)
- GOOD = 0.65-0.75 (low risk, defensive)
- ELITE = 0.75-0.8 (minimal risk, fortress-like)

Then assign a score within that category's range based on risk assessment.

Return JSON: {{"category": "AVERAGE", "score": 0.58}}
""",
    
    'momentum_trader': """You are a MOMENTUM TRADER hedge fund manager who follows market trends and momentum.
You focus on: price trends, market sentiment, technical indicators, and relative strength.
You are DYNAMIC and adapt quickly to changing market conditions.

COMPARATIVE ANCHOR: Consider the average S&P 500 company as a score of 0.5. Companies with strong momentum
and positive trends deserve above 0.7. Only exceptional momentum plays deserve above 0.9.

Stock: {ticker} ({sector}), Market Cap: ${market_cap}B, P/E: {trailing_pe}, ROE: {roe}%, Margins: {gross_margin}%/{operating_margin}%/{profit_margin}%

FIRST: Choose a category: [TRASH, AVERAGE, GOOD, ELITE]
- TRASH = 0.4-0.5 (negative momentum, weak trends)
- AVERAGE = 0.5-0.7 (neutral momentum, mixed signals)
- GOOD = 0.7-0.85 (positive momentum, strong trends)
- ELITE = 0.85-1.0 (exceptional momentum, market leader)

Then assign a score within that category's range based on momentum and market trends.

Return JSON: {{"category": "GOOD", "score": 0.76}}
"""
}

# Legacy Prompt-Template (wird nicht mehr verwendet, aber für Kompatibilität behalten)
# Prompt-Template für Stock-Bewertung
# WICHTIG: LLMs sollen intuitiv bewerten, weniger auf Finanzdaten, mehr auf qualitative Aspekte
PROMPT_TEMPLATE = """You are an experienced investment analyst with deep intuition about business quality and long-term potential.
Evaluate this stock based on your INTUITIVE UNDERSTANDING of the business, not just financial metrics.

Think about:
1. What is the QUALITY of this business? (brand strength, competitive moat, management quality, business model sustainability)
2. What is the LONG-TERM POTENTIAL over the next 2-5 years? (market position, growth opportunities, industry trends, competitive advantages)

Stock Information:
Ticker: {ticker}
Sector: {sector}
Market Cap: ${market_cap}B

Reference Financial Context (use sparingly, mainly for context):
P/E Ratio: {trailing_pe}
ROE: {roe}%
Margins: Gross {gross_margin}%, Operating {operating_margin}%, Profit {profit_margin}%

EVALUATION APPROACH:
- Use your KNOWLEDGE and INTUITION about this company and its industry
- Consider: Is this a high-quality business? Does it have a strong competitive position?
- Think about: Long-term sustainability, brand value, market leadership, innovation capability
- Assess: Management quality (if known), industry dynamics, future growth potential
- Consider the sector context but evaluate the specific company's unique position

CRITICAL INSTRUCTIONS:
- Evaluate based on BUSINESS QUALITY and INTUITION, not just financial ratios
- Be HIGHLY DIFFERENTIATED - different stocks MUST receive different scores
- Use a REALISTIC range: excellent businesses 0.85-1.0, good businesses 0.70-0.85, average 0.55-0.70, below average 0.40-0.55
- The minimum score should be 0.40 (even poor businesses in this universe have some value)
- The maximum score should be 1.0 (truly exceptional businesses)
- Spread scores across the full 0.4-1.0 range based on actual business quality differences
- Think critically: What makes this company special (or not)? What are its unique strengths/weaknesses?

Return ONLY a JSON object with a single "score" between 0.40 (below average business quality/potential) and 1.0 (exceptional business quality/potential):
{{"score": 0.75}}
"""


def _format_financial_data(ticker: str, financial_data: Dict, persona: str = 'balanced', 
                          sector_benchmarks: Optional[Dict] = None, price_vs_52w_high: Optional[float] = None) -> str:
    """
    Formatiert Finanzdaten für Prompt mit spezifischer Persona.
    Enthält jetzt Peer-Benchmarking (Sektor-Durchschnitte) und 52-Wochen-Hoch Information.
    """
    # Unterstütze sowohl camelCase als auch snake_case Spaltennamen
    def get_value(key_variants, default=None, multiplier=1):
        for key in key_variants:
            val = financial_data.get(key)
            if val is not None and pd.notna(val) and val != 0:
                return val * multiplier
        return default
    
    # Minimaler Kontext - nur das Nötigste
    market_cap = get_value(['marketCap', 'market_cap'], 0) / 1e9
    trailing_pe = get_value(['trailingPE', 'trailing_pe'])
    roe = get_value(['returnOnEquity', 'return_on_equity', 'roe'], multiplier=100)
    gross_margin = get_value(['grossMargins', 'gross_margins'], multiplier=100)
    operating_margin = get_value(['operatingMargins', 'operating_margins'], multiplier=100)
    profit_margin = get_value(['profitMargins', 'profit_margins'], multiplier=100)
    
    # Peer-Benchmarking (Sektor-Durchschnitte)
    sector_pe_avg = sector_benchmarks.get('pe_avg', 'N/A') if sector_benchmarks else 'N/A'
    sector_roe_avg = sector_benchmarks.get('roe_avg', 'N/A') if sector_benchmarks else 'N/A'
    sector_margin_avg = sector_benchmarks.get('margin_avg', 'N/A') if sector_benchmarks else 'N/A'
    pe_5y_avg = sector_benchmarks.get('pe_5y_avg', 'N/A') if sector_benchmarks else 'N/A'
    
    # 52-Wochen-Hoch Information
    price_vs_52w_str = f"{price_vs_52w_high:.1f}" if price_vs_52w_high is not None else "N/A"
    
    # Wähle Template basierend auf Persona
    template = PROMPT_TEMPLATES.get(persona, PROMPT_TEMPLATES['balanced'])
    
    # Format-String mit allen neuen Feldern
    format_dict = {
        'ticker': ticker,
        'sector': financial_data.get('sector', 'Unknown'),
        'market_cap': f"{market_cap:.1f}" if market_cap else "N/A",
        'trailing_pe': f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A",
        'roe': f"{roe:.1f}" if roe is not None else "N/A",
        'gross_margin': f"{gross_margin:.1f}" if gross_margin is not None else "N/A",
        'operating_margin': f"{operating_margin:.1f}" if operating_margin is not None else "N/A",
        'profit_margin': f"{profit_margin:.1f}" if profit_margin is not None else "N/A",
        'sector_pe_avg': sector_pe_avg,
        'sector_roe_avg': sector_roe_avg,
        'sector_margin_avg': sector_margin_avg,
        'pe_5y_avg': pe_5y_avg,
        'price_vs_52w_high': price_vs_52w_str
    }
    
    # Für Templates ohne neue Felder (legacy), verwende nur die alten Felder
    try:
        return template.format(**format_dict)
    except KeyError:
        # Fallback für Templates ohne neue Felder
        return template.format(
            ticker=ticker,
            sector=financial_data.get('sector', 'Unknown'),
            market_cap=f"{market_cap:.1f}" if market_cap else "N/A",
            trailing_pe=f"{trailing_pe:.2f}" if trailing_pe is not None else "N/A",
            roe=f"{roe:.1f}" if roe is not None else "N/A",
            gross_margin=f"{gross_margin:.1f}" if gross_margin is not None else "N/A",
            operating_margin=f"{operating_margin:.1f}" if operating_margin is not None else "N/A",
            profit_margin=f"{profit_margin:.1f}" if profit_margin is not None else "N/A"
        )


def _parse_score_from_response(response_text: str) -> Optional[float]:
    """Parst Score aus LLM-Response (unterstützt Forced Choice mit category - Ansatz 4)."""
    try:
        # Entferne Markdown-Code-Blöcke falls vorhanden
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        # Parse JSON
        result = json.loads(text)
        
        # Unterstütze sowohl "score" direkt als auch "category" + "score" (Forced Choice)
        score = result.get('score')
        if score is None:
            # Fallback: Versuche category zu interpretieren
            category = result.get('category', '').upper()
            if category == 'TRASH':
                score = 0.45
            elif category == 'AVERAGE':
                score = 0.55
            elif category == 'GOOD':
                score = 0.75
            elif category == 'ELITE':
                score = 0.90
            else:
                score = 0.5
        
        score = float(score)
        
        # Clamp auf 0.4-1.0 (realistische Range für Top-Unternehmen)
        return max(0.4, min(1.0, score))
    except Exception as e:
        logger.debug(f"Failed to parse score from response: {e}")
        return None


def _evaluate_ticker_groq(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Groq (kostenlos, bereits vorhanden)."""
    if not GROQ_API_KEY or 'groq' not in ENABLED_LLMS:
        return None
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        # Cache-Key (inkl. Persona für verschiedene Prompts)
        month = datetime.now().strftime("%Y-%m")
        persona = LLM_PERSONAS.get('groq', 'balanced')
        cache_key = f"llm_committee:groq:{persona}:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen mit Visionary-Persona (mit Benchmarks und 52W-High)
        sector_benchmarks = financial_data.get('_sector_benchmarks')
        price_vs_52w_high = financial_data.get('_price_vs_52w_high')
        prompt = _format_financial_data(ticker, financial_data, persona=persona, 
                                       sector_benchmarks=sector_benchmarks, 
                                       price_vs_52w_high=price_vs_52w_high)
        
        # API-Call (maximale Temperature für mutige, differenzierte Bewertungen)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a VISIONARY hedge fund manager. Think boldly, differentiate strongly. Respond only with valid JSON containing 'category' and 'score'."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,  # Maximale Temperature für mutige, differenzierte Bewertungen
            max_tokens=200,
            timeout=30.0
        )
        
        result_text = response.choices[0].message.content
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)  # 30 Tage Cache
            return score
        
        return None
    except ImportError:
        logger.debug("groq package not installed")
        return None
    except Exception as e:
        logger.debug(f"Groq API error for {ticker}: {e}")
        return None


def _evaluate_ticker_huggingface(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Hugging Face Inference API (kostenlos)."""
    if not HUGGINGFACE_API_KEY or 'huggingface' not in ENABLED_LLMS:
        return None
    
    try:
        import requests
        
        # Cache-Key
        month = datetime.now().strftime("%Y-%m")
        cache_key = f"llm_committee:hf:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen
        prompt = _format_financial_data(ticker, financial_data)
        
        # API-Call (verwende ein kostenloses Modell)
        model = "mistralai/Mistral-7B-Instruct-v0.2"  # Kostenlos verfügbar
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        payload = {
            "inputs": f"<s>[INST] {prompt} [/INST]",
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 1.0,  # Maximale Temperature für mutige, differenzierte Bewertungen
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            result_text = result[0].get('generated_text', '')
        else:
            result_text = str(result)
        
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("requests package not installed")
        return None
    except Exception as e:
        logger.debug(f"Hugging Face API error for {ticker}: {e}")
        return None


def _evaluate_ticker_together(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Together AI (free tier)."""
    if not TOGETHER_API_KEY or 'together' not in ENABLED_LLMS:
        return None
    
    try:
        import requests
        
        # Cache-Key
        month = datetime.now().strftime("%Y-%m")
        cache_key = f"llm_committee:together:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen
        prompt = _format_financial_data(ticker, financial_data)
        
        # API-Call
        api_url = "https://api.together.xyz/inference"
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 1.0,  # Maximale Temperature für mutige, differenzierte Bewertungen
            "stop": ["\n\n"]
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        result_text = result.get('output', {}).get('choices', [{}])[0].get('text', '')
        
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("requests package not installed")
        return None
    except Exception as e:
        logger.debug(f"Together AI API error for {ticker}: {e}")
        return None


def _evaluate_ticker_gemini(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Google Gemini (free tier)."""
    if not GEMINI_API_KEY or 'gemini' not in ENABLED_LLMS:
        return None
    
    try:
        import google.generativeai as genai
        
        # Cache-Key
        month = datetime.now().strftime("%Y-%m")
        cache_key = f"llm_committee:gemini:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        # Prompt erstellen
        prompt = _format_financial_data(ticker, financial_data)
        
        # API-Call (maximale Temperature für mutige, differenzierte Bewertungen)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 1.0,  # Maximale Temperature für mutige, differenzierte Bewertungen
                "max_output_tokens": 200,
            }
        )
        
        result_text = response.text
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("google-generativeai package not installed")
        return None
    except Exception as e:
        logger.debug(f"Gemini API error for {ticker}: {e}")
        return None


def _evaluate_ticker_openai(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via OpenAI (GPT-3.5-turbo, kostenloser Tier verfügbar)."""
    if not OPENAI_API_KEY or 'openai' not in ENABLED_LLMS:
        return None
    
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Cache-Key (inkl. Persona)
        month = datetime.now().strftime("%Y-%m")
        persona = LLM_PERSONAS.get('openai', 'balanced')
        cache_key = f"llm_committee:openai:{persona}:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen mit Growth Optimist-Persona
        # Prompt erstellen mit Benchmarks und 52W-High
        sector_benchmarks = financial_data.get('_sector_benchmarks')
        price_vs_52w_high = financial_data.get('_price_vs_52w_high')
        prompt = _format_financial_data(ticker, financial_data, persona=persona,
                                       sector_benchmarks=sector_benchmarks,
                                       price_vs_52w_high=price_vs_52w_high)
        
        # API-Call (maximale Temperature)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Kostenloser Tier verfügbar
            messages=[
                {"role": "system", "content": "You are a GROWTH OPTIMIST hedge fund manager. Focus on innovation and future potential. Respond only with valid JSON containing 'category' and 'score'."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=200,
            timeout=30.0
        )
        
        result_text = response.choices[0].message.content
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("openai package not installed")
        return None
    except Exception as e:
        logger.debug(f"OpenAI API error for {ticker}: {e}")
        return None


def _evaluate_ticker_anthropic(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Anthropic Claude (kostenloser Tier verfügbar)."""
    if not ANTHROPIC_API_KEY or 'anthropic' not in ENABLED_LLMS:
        return None
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Cache-Key (inkl. Persona)
        month = datetime.now().strftime("%Y-%m")
        persona = LLM_PERSONAS.get('anthropic', 'balanced')
        cache_key = f"llm_committee:anthropic:{persona}:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen mit Risk Analyst-Persona
        # Prompt erstellen mit Benchmarks und 52W-High
        sector_benchmarks = financial_data.get('_sector_benchmarks')
        price_vs_52w_high = financial_data.get('_price_vs_52w_high')
        prompt = _format_financial_data(ticker, financial_data, persona=persona,
                                       sector_benchmarks=sector_benchmarks,
                                       price_vs_52w_high=price_vs_52w_high)
        
        # API-Call (maximale Temperature)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Kostenloser Tier verfügbar
            max_tokens=200,
            temperature=1.0,
            system="You are a RISK ANALYST hedge fund manager. Analyze risks deeply. Respond only with valid JSON containing 'category' and 'score'.",
            messages=[
                {"role": "user", "content": prompt}
            ],
            timeout=30.0
        )
        
        result_text = message.content[0].text
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("anthropic package not installed")
        return None
    except Exception as e:
        logger.debug(f"Anthropic API error for {ticker}: {e}")
        return None


def _evaluate_ticker_cohere(ticker: str, financial_data: Dict) -> Optional[float]:
    """Bewertet Ticker via Cohere (kostenloser Tier verfügbar)."""
    if not COHERE_API_KEY or 'cohere' not in ENABLED_LLMS:
        return None
    
    try:
        import cohere
        client = cohere.Client(api_key=COHERE_API_KEY)
        
        # Cache-Key (inkl. Persona)
        month = datetime.now().strftime("%Y-%m")
        persona = LLM_PERSONAS.get('cohere', 'balanced')
        cache_key = f"llm_committee:cohere:{persona}:{ticker}:{month}"
        cached = cache_get(cache_key)
        if cached is not None:
            return float(cached)
        
        # Prompt erstellen mit Momentum Trader-Persona
        # Prompt erstellen mit Benchmarks und 52W-High
        sector_benchmarks = financial_data.get('_sector_benchmarks')
        price_vs_52w_high = financial_data.get('_price_vs_52w_high')
        prompt = _format_financial_data(ticker, financial_data, persona=persona,
                                       sector_benchmarks=sector_benchmarks,
                                       price_vs_52w_high=price_vs_52w_high)
        
        # API-Call (maximale Temperature)
        response = client.generate(
            model="command",  # Kostenloser Tier verfügbar
            prompt=prompt,
            max_tokens=200,
            temperature=1.0,
            stop_sequences=["\n\n"]
        )
        
        result_text = response.generations[0].text
        score = _parse_score_from_response(result_text)
        
        if score is not None:
            cache_set(cache_key, str(score), ttl_seconds=30 * 24 * 3600)
            return score
        
        return None
    except ImportError:
        logger.debug("cohere package not installed")
        return None
    except Exception as e:
        logger.debug(f"Cohere API error for {ticker}: {e}")
        return None


def _aggregate_scores_weighted_median(scores_with_llm: List[tuple], llm_weights: Dict[str, float]) -> float:
    """
    Aggregiert LLM-Scores mit gewichtetem Median (Value Investing: Skeptiker bevorzugt).
    
    Args:
        scores_with_llm: Liste von (llm_name, score) Tupeln
        llm_weights: Dictionary mit LLM-Namen als Keys und Gewichtungen als Values
    
    Returns:
        Gewichteter Median-Score mit asymmetrischer Aggregation für TRASH-Ratings
    """
    # Filtere None/Fehler-Werte und wende Gewichtungen an
    weighted_scores = []
    trash_penalty = 0.0
    trash_count = 0
    
    for llm_name, score in scores_with_llm:
        if score is None or not (0.0 <= score <= 1.0):
            continue
        
        weight = llm_weights.get(llm_name, 1.0)
        
        # Asymmetrische Aggregation: TRASH-Ratings (Score < 0.5) ziehen stärker nach unten
        if score < 0.5:
            # Value Hunter und Risk Analyst haben Veto-Recht (stärkere Gewichtung)
            if llm_name in ['together', 'anthropic']:  # Value Hunter, Risk Analyst
                trash_penalty += (0.5 - score) * weight * 1.5  # Stärkere Strafe
                trash_count += 1
            else:
                trash_penalty += (0.5 - score) * weight * 0.8  # Leichtere Strafe
        
        # Gewichtete Scores für Median-Berechnung
        # Wiederhole jeden Score entsprechend seinem Gewicht (gerundet)
        repeat_count = max(1, int(weight * 10))  # Gewichtung als Multiplikator
        for _ in range(repeat_count):
            weighted_scores.append(score)
    
    if len(weighted_scores) == 0:
        return 0.5  # Neutral wenn keine Scores verfügbar
    
    if len(weighted_scores) == 1:
        base_score = weighted_scores[0]
    else:
        # Gewichteter Median
        base_score = float(np.median(weighted_scores))
    
    # Asymmetrische Strafe anwenden: TRASH-Ratings ziehen stärker nach unten
    if trash_penalty > 0:
        # Stärkere Strafe wenn mehrere Skeptiker zustimmen
        penalty_multiplier = 1.0 + (trash_count * 0.2)  # +20% pro zusätzlichem Skeptiker
        final_penalty = trash_penalty * penalty_multiplier
        base_score = max(0.4, base_score - final_penalty)
    
    return max(0.4, min(1.0, base_score))


def _aggregate_scores_median(scores: List[Optional[float]]) -> float:
    """Legacy: Aggregiert LLM-Scores mit Median (robust gegen Ausreißer)."""
    # Filtere None/Fehler-Werte
    valid_scores = [s for s in scores if s is not None and 0.0 <= s <= 1.0]
    
    if len(valid_scores) == 0:
        return 0.5  # Neutral wenn keine Scores verfügbar
    
    if len(valid_scores) == 1:
        return valid_scores[0]
    
    # Median berechnen
    return float(np.median(valid_scores))


def _scale_scores_to_range(scores_dict: Dict[str, float], min_score: float = 0.4, max_score: float = 1.0) -> Dict[str, float]:
    """
    Statistische Skalierung: Streckt Scores auf die volle Range 0.4-1.0.
    Post-Processing nach Ansatz 3: Min-Max-Scaling.
    """
    if not scores_dict:
        return scores_dict
    
    scores_list = list(scores_dict.values())
    old_min = min(scores_list)
    old_max = max(scores_list)
    
    # Wenn alle Scores gleich sind, keine Skalierung möglich
    if old_max == old_min:
        # Setze alle auf Mittelwert der Range
        default_score = (min_score + max_score) / 2
        return {ticker: default_score for ticker in scores_dict.keys()}
    
    # Min-Max-Scaling: Score_new = (Score_old - Min_old) / (Max_old - Min_old) * (Max_new - Min_new) + Min_new
    scaled_dict = {}
    for ticker, score in scores_dict.items():
        scaled_score = (score - old_min) / (old_max - old_min) * (max_score - min_score) + min_score
        scaled_dict[ticker] = max(min_score, min(max_score, scaled_score))  # Clamp
    
    return scaled_dict


def _evaluate_ticker_all_llms(ticker: str, financial_data: Dict, sector_benchmarks: Optional[Dict] = None, 
                               price_vs_52w_high: Optional[float] = None) -> float:
    """
    Bewertet Ticker mit allen verfügbaren LLMs parallel.
    Verwendet jetzt Weighted Median mit asymmetrischer Aggregation für Value Investing.
    """
    # Liste aller LLM-Funktionen (nur aktivierte)
    llm_functions = []
    if 'groq' in ENABLED_LLMS:
        llm_functions.append(('groq', _evaluate_ticker_groq))
    if 'huggingface' in ENABLED_LLMS:
        llm_functions.append(('huggingface', _evaluate_ticker_huggingface))
    if 'together' in ENABLED_LLMS:
        llm_functions.append(('together', _evaluate_ticker_together))
    if 'gemini' in ENABLED_LLMS:
        llm_functions.append(('gemini', _evaluate_ticker_gemini))
    if 'openai' in ENABLED_LLMS:
        llm_functions.append(('openai', _evaluate_ticker_openai))
    if 'anthropic' in ENABLED_LLMS:
        llm_functions.append(('anthropic', _evaluate_ticker_anthropic))
    if 'cohere' in ENABLED_LLMS:
        llm_functions.append(('cohere', _evaluate_ticker_cohere))
    
    if not llm_functions:
        logger.warning(f"{ticker}: No LLMs enabled, using neutral score")
        return 0.5
    
    # Parallele API-Calls (mit erweiterten Finanzdaten)
    scores_with_llm = []
    with ThreadPoolExecutor(max_workers=len(llm_functions)) as executor:
        # Erweitere financial_data mit Benchmarks und 52W-High
        extended_financial_data = financial_data.copy()
        if sector_benchmarks:
            extended_financial_data['_sector_benchmarks'] = sector_benchmarks
        if price_vs_52w_high is not None:
            extended_financial_data['_price_vs_52w_high'] = price_vs_52w_high
        
        futures = {
            executor.submit(func, ticker, extended_financial_data): name
            for name, func in llm_functions
        }
        
        for future in as_completed(futures, timeout=35):
            llm_name = futures[future]
            try:
                score = future.result()
                if score is not None:
                    scores_with_llm.append((llm_name, score))
                    logger.debug(f"{ticker} - {llm_name}: {score:.3f}")
            except Exception as e:
                logger.debug(f"{ticker} - {llm_name} failed: {e}")
    
    # Aggregiere mit Weighted Median (Value Investing: Skeptiker bevorzugt)
    if len(scores_with_llm) == 0:
        logger.warning(f"{ticker}: No LLMs succeeded, using neutral score")
        return 0.5
    
    if len(scores_with_llm) == 1:
        logger.debug(f"{ticker}: Only 1 LLM succeeded (using single score)")
        return scores_with_llm[0][1]
    
    # Mehrere Scores: Weighted Median mit asymmetrischer Aggregation
    return _aggregate_scores_weighted_median(scores_with_llm, LLM_WEIGHTS)


def get_llm_committee_score(tickers: List[str], financials_df: pd.DataFrame) -> Dict[str, float]:
    """
    Hauptfunktion: Bewertet alle Ticker mit mehreren kostenlosen LLMs.
    
    WICHTIG: Übergibt nur rohe Finanzdaten, keine bereits berechneten Scores!
    
    Args:
        tickers: Liste von Ticker-Symbolen (normalerweise die Top 20)
        financials_df: DataFrame mit Finanzdaten für alle Ticker (kann auch scored_df sein, wird gefiltert)
    
    Returns:
        Dict mapping ticker -> committee_score (0.0-1.0)
    """
    logger.info(f"Starting LLM committee evaluation for {len(tickers)} tickers")
    
    # Filtere nur rohe Finanzdaten (keine Scores)
    # WICHTIG: Prüfe alle möglichen Spaltennamen (kamelCase und snake_case)
    financial_columns = [
        'ticker', 'sector', 
        'marketCap', 'market_cap',  # Beide Varianten
        'trailingPE', 'trailing_pe',
        'forwardPE', 'forward_pe',
        'returnOnEquity', 'return_on_equity', 'roe',
        'debtToEquity', 'debt_to_equity',
        'beta',
        'grossMargins', 'gross_margins',
        'operatingMargins', 'operating_margins',
        'profitMargins', 'profit_margins',
        'returnOnInvestedCapital', 'return_on_invested_capital', 'roic',
        'pegRatio', 'peg_ratio',
        'priceToFreeCashFlow', 'price_to_free_cash_flow',
        'enterpriseToEbitda', 'enterprise_to_ebitda',
        'revenueGrowth', 'revenue_growth',
        'earningsGrowth', 'earnings_growth'
    ]
    
    # Erstelle gefiltertes DataFrame nur mit Finanzdaten
    available_cols = [col for col in financial_columns if col in financials_df.columns]
    
    # Debug: Log welche Spalten gefunden wurden
    logger.debug(f"Available financial columns: {available_cols}")
    logger.debug(f"All DataFrame columns: {list(financials_df.columns)}")
    
    if not available_cols or 'ticker' not in available_cols:
        logger.warning(f"Very few financial columns found. Available: {available_cols}")
        # Fallback: Nimm alle Spalten außer Score-Spalten
        score_columns = ['reddit_score', 'x_score', 'youtube_score', 'superinvestor_score', 
                        'final_score', 'value_score', 'quality_score', 'community_score',
                        'ai_moat_score', 'ai_quality_score', 'ai_predicted_performance',
                        'committee_score', 'weight', 'weight_%']
        available_cols = [col for col in financials_df.columns 
                         if col not in score_columns and col != 'companyName']
        logger.debug(f"Using fallback columns: {available_cols}")
    
    financials_only_df = financials_df[['ticker'] + [col for col in available_cols if col != 'ticker']].copy()
    
    # Berechne Sektor-Benchmarks (Peer-Benchmarking)
    def get_sector_benchmarks(sector: str, df: pd.DataFrame) -> Dict:
        """Berechnet Durchschnitte für einen Sektor (P/E, ROE, Margins)."""
        sector_df = df[df['sector'] == sector] if 'sector' in df.columns else pd.DataFrame()
        if sector_df.empty:
            return {}
        
        def safe_mean(col_variants):
            for col in col_variants:
                if col in sector_df.columns:
                    vals = sector_df[col].dropna()
                    if len(vals) > 0:
                        return float(vals.mean())
            return None
        
        pe_avg = safe_mean(['trailingPE', 'trailing_pe'])
        roe_avg = safe_mean(['returnOnEquity', 'return_on_equity', 'roe'])
        if roe_avg is not None:
            roe_avg *= 100  # Konvertiere zu Prozent
        
        # Durchschnittliche Margins
        margin_cols = ['grossMargins', 'gross_margins', 'operatingMargins', 'operating_margins', 
                      'profitMargins', 'profit_margins']
        margins = []
        for col in margin_cols:
            if col in sector_df.columns:
                vals = sector_df[col].dropna() * 100  # Konvertiere zu Prozent
                margins.extend(vals.tolist())
        margin_avg = float(np.mean(margins)) if margins else None
        
        # 5Y P/E Durchschnitt (vereinfacht: verwende aktuelles P/E als Proxy)
        pe_5y_avg = pe_avg  # TODO: Könnte aus historischen Daten berechnet werden
        
        return {
            'pe_avg': f"{pe_avg:.2f}" if pe_avg is not None else "N/A",
            'roe_avg': f"{roe_avg:.1f}" if roe_avg is not None else "N/A",
            'margin_avg': f"{margin_avg:.1f}" if margin_avg is not None else "N/A",
            'pe_5y_avg': f"{pe_5y_avg:.2f}" if pe_5y_avg is not None else "N/A"
        }
    
    # Berechne 52-Wochen-Hoch für jeden Ticker
    def get_price_vs_52w_high(ticker: str) -> Optional[float]:
        """Berechnet aktuellen Preis vs. 52-Wochen-Hoch (in Prozent)."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            high_52w = hist['High'].max()
            
            if high_52w > 0:
                return (current_price / high_52w) * 100  # Prozent
            return None
        except Exception as e:
            logger.debug(f"Could not calculate 52W high for {ticker}: {e}")
            return None
    
    # Konvertiere DataFrame zu Dict für schnelleren Zugriff
    financials_dict = financials_only_df.set_index('ticker').to_dict('index')
    
    results = {}
    
    # Bewerte jeden Ticker
    for i, ticker in enumerate(tickers):
        if (i + 1) % 5 == 0:
            logger.info(f"LLM Committee: {i + 1}/{len(tickers)}")
        
        try:
            financial_data = financials_dict.get(ticker, {})
            if not financial_data:
                logger.warning(f"No financial data for {ticker}, using neutral score")
                results[ticker] = 0.5
                continue
            
            # Berechne Sektor-Benchmarks
            sector = financial_data.get('sector', 'Unknown')
            sector_benchmarks = get_sector_benchmarks(sector, financials_only_df)
            
            # Berechne 52-Wochen-Hoch
            price_vs_52w_high = get_price_vs_52w_high(ticker)
            
            # Bewerte mit allen LLMs (mit Benchmarks und 52W-High)
            score = _evaluate_ticker_all_llms(ticker, financial_data, 
                                            sector_benchmarks=sector_benchmarks,
                                            price_vs_52w_high=price_vs_52w_high)
            results[ticker] = score
            
        except Exception as e:
            logger.warning(f"Error evaluating {ticker}: {e}, using neutral score")
            results[ticker] = 0.5
    
    logger.info(f"LLM committee evaluation completed: {len(results)} scores")
    
    # Post-Processing: Statistische Skalierung auf volle Range 0.4-1.0 (Ansatz 3)
    if results:
        scaled_results = _scale_scores_to_range(results, min_score=0.4, max_score=1.0)
        logger.info(f"Scores scaled to range: min={min(scaled_results.values()):.3f}, max={max(scaled_results.values()):.3f}")
        return scaled_results
    
    return results
