"""
Test script to verify that community_signals.json is being used correctly
and that simulated data has the right mean (0.3-0.35).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add scripts to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "scripts"))

# Import functions
from reddit import get_reddit_mentions
from twitter import get_x_sentiment_score
from community import load_community_signals

def test_community_signals_integration():
    """Test that community_signals.json is used correctly"""
    
    # Load community_signals.json to see what real data we have
    print("Loading community_signals.json...")
    try:
        _, reddit_real, x_real = load_community_signals()
        print(f"  Found {len(reddit_real)} real Reddit scores")
        print(f"  Found {len(x_real)} real X/Twitter scores")
        print(f"\n  Real Reddit scores (sample):")
        for ticker, score in list(reddit_real.items())[:5]:
            print(f"    {ticker}: {score}")
        print(f"\n  Real X scores (sample):")
        for ticker, score in list(x_real.items())[:5]:
            print(f"    {ticker}: {score}")
    except Exception as e:
        print(f"  ERROR loading community_signals.json: {e}")
        return
    
    # Load financials for fallback generation
    financials_path = root_dir / "data/financials/latest.csv"
    if financials_path.exists():
        financials_df = pd.read_csv(financials_path)
        tickers = financials_df['ticker'].tolist()
        print(f"\nLoaded {len(tickers)} tickers from financials")
    else:
        print("\nWARNING: No financials data found, using sample tickers")
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JPM', 'COST', 'WMT', 'KO']
    
    # Test Reddit scores
    print("\n" + "=" * 60)
    print("TESTING REDDIT SCORES")
    print("=" * 60)
    reddit_scores = get_reddit_mentions(tickers, financials_df=financials_df if financials_path.exists() else None)
    
    # Check which tickers use real data vs simulated
    reddit_real_tickers = []
    reddit_simulated_tickers = []
    reddit_real_values = []
    reddit_simulated_values = []
    
    for ticker in tickers:
        if ticker in reddit_real:
            reddit_real_tickers.append(ticker)
            reddit_real_values.append(reddit_scores.get(ticker))
        else:
            reddit_simulated_tickers.append(ticker)
            reddit_simulated_values.append(reddit_scores.get(ticker))
    
    print(f"\nReddit Scores:")
    print(f"  Real data (from community_signals.json): {len(reddit_real_tickers)} tickers")
    if reddit_real_tickers:
        print(f"    Sample: {reddit_real_tickers[:5]}")
        print(f"    Values: {[reddit_scores.get(t) for t in reddit_real_tickers[:5]]}")
    
    print(f"\n  Simulated data: {len(reddit_simulated_tickers)} tickers")
    if reddit_simulated_values:
        sim_mean = np.mean(reddit_simulated_values)
        sim_std = np.std(reddit_simulated_values)
        sim_min = np.min(reddit_simulated_values)
        sim_max = np.max(reddit_simulated_values)
        print(f"    Mean: {sim_mean:.3f} (target: 0.3-0.35)")
        print(f"    Std Dev: {sim_std:.3f}")
        print(f"    Min: {sim_min:.3f}, Max: {sim_max:.3f}")
        if 0.30 <= sim_mean <= 0.40:
            print(f"    [OK] Mean is in target range")
        else:
            print(f"    [WARNING] Mean is outside target range")
    
    # Test X/Twitter scores
    print("\n" + "=" * 60)
    print("TESTING X/TWITTER SCORES")
    print("=" * 60)
    x_scores = get_x_sentiment_score(tickers, financials_df=financials_df if financials_path.exists() else None)
    
    # Check which tickers use real data vs simulated
    x_real_tickers = []
    x_simulated_tickers = []
    x_real_values = []
    x_simulated_values = []
    
    for ticker in tickers:
        if ticker in x_real:
            x_real_tickers.append(ticker)
            x_real_values.append(x_scores.get(ticker))
        else:
            x_simulated_tickers.append(ticker)
            x_simulated_values.append(x_scores.get(ticker))
    
    print(f"\nX/Twitter Scores:")
    print(f"  Real data (from community_signals.json): {len(x_real_tickers)} tickers")
    if x_real_tickers:
        print(f"    Sample: {x_real_tickers[:5]}")
        print(f"    Values: {[x_scores.get(t) for t in x_real_tickers[:5]]}")
    
    print(f"\n  Simulated data: {len(x_simulated_tickers)} tickers")
    if x_simulated_values:
        sim_mean = np.mean(x_simulated_values)
        sim_std = np.std(x_simulated_values)
        sim_min = np.min(x_simulated_values)
        sim_max = np.max(x_simulated_values)
        print(f"    Mean: {sim_mean:.3f} (target: 0.3-0.35)")
        print(f"    Std Dev: {sim_std:.3f}")
        print(f"    Min: {sim_min:.3f}, Max: {sim_max:.3f}")
        if 0.30 <= sim_mean <= 0.40:
            print(f"    [OK] Mean is in target range")
        else:
            print(f"    [WARNING] Mean is outside target range")
    
    # Verify that real data is actually used
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Check a few known tickers from community_signals.json
    test_tickers = ['JPM', 'GOOGL', 'BRK-B', 'KO', 'META']
    print("\nChecking if real data is used for known tickers:")
    for ticker in test_tickers:
        if ticker in tickers:
            reddit_val = reddit_scores.get(ticker)
            x_val = x_scores.get(ticker)
            expected_reddit = reddit_real.get(ticker)
            expected_x = x_real.get(ticker)
            
            reddit_match = abs(reddit_val - expected_reddit) < 0.01 if expected_reddit else False
            x_match = abs(x_val - expected_x) < 0.01 if expected_x else False
            
            print(f"  {ticker}:")
            print(f"    Reddit: {reddit_val:.3f} (expected: {expected_reddit:.3f}) {'[OK]' if reddit_match else '[FAIL]'}")
            print(f"    X: {x_val:.3f} (expected: {expected_x:.3f}) {'[OK]' if x_match else '[FAIL]'}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_community_signals_integration()

