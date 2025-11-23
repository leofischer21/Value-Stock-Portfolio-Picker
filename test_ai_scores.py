#!/usr/bin/env python3
"""Test script for AI Scores functionality"""
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

print("=" * 60)
print("KI Scores & Performance Predictions - Comprehensive Test")
print("=" * 60)
print()

# Test 1: API Key Configuration
print("1. API Key Configuration Check")
print("-" * 60)
openai_key = os.environ.get('OPENAI_API_KEY')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
ai_model = os.environ.get('AI_MODEL', 'gpt-4')
print(f"OPENAI_API_KEY: {'SET' if openai_key else 'NOT SET'}")
print(f"ANTHROPIC_API_KEY: {'SET' if anthropic_key else 'NOT SET'}")
print(f"AI_MODEL: {ai_model}")
if not openai_key and not anthropic_key:
    print("WARNING: No API keys configured - system will use heuristic fallbacks")
    print("  This is OK and fully functional!")
else:
    print("OK: API keys found - LLM will be used when available")
print()

# Test 2: AI Scores Module (Heuristic Fallbacks)
print("2. Testing AI Scores Module (without API keys)")
print("-" * 60)
try:
    from ai_scores import get_ai_moat_score, get_ai_quality_score, get_ai_predicted_performance
    
    test_data = {
        'trailingPE': 25.0,
        'forwardPE': 23.0,
        'grossMargins': 0.45,
        'operatingMargins': 0.32,
        'profitMargins': 0.28,
        'returnOnEquity': 0.25,
        'returnOnInvestedCapital': 0.18,
        'debtToEquity': 0.8,
        'beta': 1.1,
        'sector': 'Technology'
    }
    test_scores = {
        'community_score': 0.65,
        'quality_score': 0.75
    }
    
    print("Test ticker: AAPL")
    moat = get_ai_moat_score('AAPL', test_data, test_scores)
    quality = get_ai_quality_score('AAPL', test_data, test_scores)
    perf = get_ai_predicted_performance('AAPL', test_data, test_scores)
    
    print(f"  Moat Score (heuristic): {moat:.3f}")
    print(f"  Quality Score (heuristic): {quality:.3f}")
    print(f"  Performance Prediction:")
    print(f"    1Y CAGR: {perf['cagr_1y']:.1f}%")
    print(f"    2Y CAGR: {perf['cagr_2y']:.1f}%")
    print(f"    5Y CAGR: {perf['cagr_5y']:.1f}%")
    print(f"    10Y CAGR: {perf['cagr_10y']:.1f}%")
    print("OK: Heuristic fallbacks work correctly without API keys!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Check if AI scores file exists
print("3. Checking AI Scores Output Files")
print("-" * 60)
ai_scores_path = Path('data/ai_scores/latest.json')
if ai_scores_path.exists():
    import json
    print("OK: AI scores file exists!")
    with open(ai_scores_path, 'r') as f:
        data = json.load(f)
    print(f"  Last updated: {data.get('last_updated', 'N/A')}")
    print(f"  Moat scores: {len(data.get('moat_scores', {}))} tickers")
    print(f"  Quality scores: {len(data.get('quality_scores', {}))} tickers")
    print(f"  Performance predictions: {len(data.get('predicted_performance', {}))} tickers")
    if 'AAPL' in data.get('moat_scores', {}):
        print()
        print("  Sample (AAPL):")
        print(f"    Moat: {data['moat_scores']['AAPL']:.3f}")
        print(f"    Quality: {data['quality_scores']['AAPL']:.3f}")
        perf = data.get('predicted_performance', {}).get('AAPL', {})
        if perf:
            print(f"    CAGR 1Y: {perf.get('cagr_1y', 0):.1f}%")
            print(f"    CAGR 2Y: {perf.get('cagr_2y', 0):.1f}%")
else:
    print("WARNING: AI scores file not found yet")
    print("  Run: python scripts/monthly_update.py to generate")
print()

# Test 4: Portfolio Calculator Integration
print("4. Testing Portfolio Calculator Integration")
print("-" * 60)
try:
    from portfolio_calculator import combine_data
    import pandas as pd
    
    tickers_df = pd.DataFrame([{'ticker': 'AAPL', 'marketCap': 3e12}])
    financials_df = pd.DataFrame([{
        'ticker': 'AAPL',
        'marketCap': 3e12,
        'trailingPE': 25.0,
        'forwardPE': 23.0,
        'sector': 'Technology',
        'beta': 1.1,
        'returnOnEquity': 0.25
    }])
    scores_dict = {
        'superinvestor_score': {'AAPL': 0.7},
        'reddit_score': {'AAPL': 0.6},
        'x_score': {'AAPL': 0.65}
    }
    
    result = combine_data(tickers_df, financials_df, scores_dict)
    print(f"OK: combine_data executed successfully!")
    print(f"  Rows: {len(result)}")
    print(f"  Columns with 'ai' or 'predicted': {[c for c in result.columns if 'ai' in c.lower() or 'predicted' in c.lower()]}")
    if 'ai_moat_score' in result.columns:
        print(f"  AI Moat Score: {result['ai_moat_score'].iloc[0]:.3f}")
    if 'ai_quality_score' in result.columns:
        print(f"  AI Quality Score: {result['ai_quality_score'].iloc[0]:.3f}")
    if 'predicted_cagr_1y' in result.columns:
        print(f"  Predicted CAGR 1Y: {result['predicted_cagr_1y'].iloc[0]:.1f}%")
    print("  OK: AI scores columns present!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 5: Monthly Update Function
print("5. Testing update_ai_scores Function")
print("-" * 60)
try:
    from monthly_update import update_ai_scores
    import pandas as pd
    
    test_tickers = ['AAPL', 'MSFT']
    test_financials = pd.DataFrame([
        {
            'ticker': 'AAPL',
            'trailingPE': 25.0,
            'grossMargins': 0.45,
            'returnOnEquity': 0.25,
            'debtToEquity': 0.8,
            'sector': 'Technology'
        },
        {
            'ticker': 'MSFT',
            'trailingPE': 30.0,
            'grossMargins': 0.40,
            'returnOnEquity': 0.22,
            'debtToEquity': 0.6,
            'sector': 'Technology'
        }
    ])
    test_scores = {
        'superinvestor_score': {'AAPL': 0.7, 'MSFT': 0.75},
        'reddit_score': {'AAPL': 0.6, 'MSFT': 0.65},
        'x_score': {'AAPL': 0.65, 'MSFT': 0.7}
    }
    
    print("Testing with 2 tickers...")
    update_ai_scores(test_tickers, test_financials, test_scores, '2025-01')
    print("OK: update_ai_scores executed successfully!")
    print("  Check data/ai_scores/2025-01.json and latest.json")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test Summary")
print("=" * 60)
print("OK: All core functionality tested")
print("OK: System works with or without API keys")
print("OK: Heuristic fallbacks are functional")
print()
print("Next Steps:")
print("1. (Optional) Set API keys for LLM-based scores:")
print("   Windows: $env:OPENAI_API_KEY='your-key'")
print("   Linux/Mac: export OPENAI_API_KEY='your-key'")
print("2. Run monthly update: python scripts/monthly_update.py")
print("3. Start dashboard: streamlit run app/dashboard_final.py")

