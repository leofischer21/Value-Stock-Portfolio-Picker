"""
Test script to verify AI scores work with real API
"""
import sys
from pathlib import Path
import pandas as pd

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Load financials
financials_path = Path("data/financials/latest.csv")
if not financials_path.exists():
    print("ERROR: Financials data not found. Run monthly_update.py first.")
    sys.exit(1)

financials_df = pd.read_csv(financials_path)
print(f"Loaded {len(financials_df)} tickers from financials")

# Test with a few well-known tickers
test_tickers = ['GOOGL', 'AAPL', 'JPM']

# Import AI scores
from ai_scores import get_ai_moat_score, get_ai_quality_score, get_ai_predicted_performance

# Load scores
scores_path = Path("data/scores/latest.json")
if scores_path.exists():
    import json
    with open(scores_path, 'r') as f:
        scores_dict = json.load(f)
else:
    scores_dict = {'reddit_score': {}, 'x_score': {}, 'superinvestor_score': {}}

print("\n" + "=" * 60)
print("TESTING AI SCORES WITH REAL API")
print("=" * 60)

for ticker in test_tickers:
    print(f"\n{ticker}:")
    print("-" * 60)
    
    # Get financial data
    fin_data = financials_df[financials_df['ticker'] == ticker]
    if len(fin_data) == 0:
        print(f"  [SKIP] Ticker not in financials")
        continue
    
    fin_row = fin_data.iloc[0]
    financial_data = {
        'trailingPE': fin_row.get('trailingPE'),
        'forwardPE': fin_row.get('forwardPE'),
        'grossMargins': fin_row.get('grossMargins'),
        'operatingMargins': fin_row.get('operatingMargins'),
        'profitMargins': fin_row.get('profitMargins'),
        'returnOnEquity': fin_row.get('returnOnEquity'),
        'returnOnInvestedCapital': fin_row.get('returnOnInvestedCapital'),
        'debtToEquity': fin_row.get('debtToEquity'),
        'sector': fin_row.get('sector', 'Unknown'),
        'beta': fin_row.get('beta'),
    }
    
    # Get scores
    reddit = scores_dict.get('reddit_score', {}).get(ticker, 0.5)
    x = scores_dict.get('x_score', {}).get(ticker, 0.5)
    community_score = (reddit * 0.5 + x * 0.5)
    
    scores = {
        'community_score': community_score,
        'value_score': 0.5,  # Placeholder
        'quality_score': 0.5,  # Placeholder
    }
    
    print(f"  Testing AI Moat Score...")
    try:
        moat_score = get_ai_moat_score(ticker, financial_data, scores)
        print(f"    Moat Score: {moat_score:.3f}")
        if moat_score > 0.6:
            print(f"    [OK] High moat score (using LLM)")
        elif moat_score < 0.4:
            print(f"    [OK] Low moat score (using LLM)")
        else:
            print(f"    [OK] Medium moat score (using LLM)")
    except Exception as e:
        print(f"    [ERROR] {e}")
    
    print(f"  Testing AI Quality Score...")
    try:
        quality_score = get_ai_quality_score(ticker, financial_data, scores)
        print(f"    Quality Score: {quality_score:.3f}")
    except Exception as e:
        print(f"    [ERROR] {e}")
    
    print(f"  Testing AI Predicted Performance...")
    try:
        perf = get_ai_predicted_performance(ticker, financial_data, scores)
        print(f"    1Y CAGR: {perf.get('cagr_1y', 0):.1f}%")
        print(f"    2Y CAGR: {perf.get('cagr_2y', 0):.1f}%")
        print(f"    5Y CAGR: {perf.get('cagr_5y', 0):.1f}%")
        print(f"    10Y CAGR: {perf.get('cagr_10y', 0):.1f}%")
    except Exception as e:
        print(f"    [ERROR] {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nIf you see realistic scores (not all 0.5), the LLM API is working!")
print("If all scores are around 0.5, check the API key and model configuration.")

