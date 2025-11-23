"""
Test script to verify the new fallback score generation for Reddit and X/Twitter.
Checks that simulated scores are penalized (mean ~0.35-0.4 instead of ~0.65).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(Path(__file__).parent))

# Change to scripts directory for imports
os.chdir(scripts_dir)

from reddit import _generate_fallback_scores as reddit_fallback
from twitter import _generate_fallback_scores as twitter_fallback

# Change back
os.chdir(Path(__file__).parent)

def test_fallback_scores():
    """Test the new fallback score generation"""
    
    # Load financials data if available
    financials_path = Path(__file__).parent / "data/financials/latest.csv"
    if financials_path.exists():
        print("Loading financials data...")
        financials_df = pd.read_csv(financials_path)
        print(f"Loaded {len(financials_df)} tickers from financials data")
    else:
        print("WARNING: financials data not found, using None")
        financials_df = None
    
    # Get universe tickers
    if financials_df is not None and not financials_df.empty:
        universe_tickers = financials_df['ticker'].tolist()
    else:
        # Fallback: use a sample list
        universe_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JPM', 'COST', 'WMT', 'KO']
        print(f"Using sample tickers: {universe_tickers}")
    
    print(f"\nTesting with {len(universe_tickers)} tickers...")
    print("=" * 60)
    
    # Test Reddit fallback scores
    print("\n1. REDDIT FALLBACK SCORES:")
    print("-" * 60)
    reddit_scores = reddit_fallback(universe_tickers, financials_df=financials_df)
    
    if reddit_scores:
        reddit_values = list(reddit_scores.values())
        reddit_mean = np.mean(reddit_values)
        reddit_std = np.std(reddit_values)
        reddit_min = np.min(reddit_values)
        reddit_max = np.max(reddit_values)
        reddit_median = np.median(reddit_values)
        
        # Count neutral scores (0.3 ± 0.05)
        neutral_count = sum(1 for v in reddit_values if abs(v - 0.3) < 0.05)
        neutral_pct = (neutral_count / len(reddit_values)) * 100
        
        print(f"  Mean: {reddit_mean:.3f} (target: 0.35-0.4)")
        print(f"  Median: {reddit_median:.3f}")
        print(f"  Std Dev: {reddit_std:.3f}")
        print(f"  Min: {reddit_min:.3f}")
        print(f"  Max: {reddit_max:.3f}")
        print(f"  Neutral (0.3±0.05): {neutral_count}/{len(reddit_values)} ({neutral_pct:.1f}%)")
        
        # Check if mean is in target range
        if 0.30 <= reddit_mean <= 0.45:
            print(f"  [OK] Mean is in acceptable range (0.30-0.45)")
        else:
            print(f"  [WARNING] Mean is outside target range (0.30-0.45)")
        
        # Show sample scores
        print(f"\n  Sample scores (first 10):")
        for i, (ticker, score) in enumerate(list(reddit_scores.items())[:10]):
            print(f"    {ticker}: {score:.3f}")
    else:
        print("  [ERROR] No scores generated!")
    
    # Test Twitter fallback scores
    print("\n2. X/TWITTER FALLBACK SCORES:")
    print("-" * 60)
    twitter_scores = twitter_fallback(universe_tickers, financials_df=financials_df)
    
    if twitter_scores:
        twitter_values = list(twitter_scores.values())
        twitter_mean = np.mean(twitter_values)
        twitter_std = np.std(twitter_values)
        twitter_min = np.min(twitter_values)
        twitter_max = np.max(twitter_values)
        twitter_median = np.median(twitter_values)
        
        # Count neutral scores (0.5 ± 0.05)
        neutral_count = sum(1 for v in twitter_values if abs(v - 0.5) < 0.05)
        neutral_pct = (neutral_count / len(twitter_values)) * 100
        
        print(f"  Mean: {twitter_mean:.3f} (target: 0.35-0.4)")
        print(f"  Median: {twitter_median:.3f}")
        print(f"  Std Dev: {twitter_std:.3f}")
        print(f"  Min: {twitter_min:.3f}")
        print(f"  Max: {twitter_max:.3f}")
        print(f"  Neutral (0.5±0.05): {neutral_count}/{len(twitter_values)} ({neutral_pct:.1f}%)")
        
        # Check if mean is in target range
        if 0.30 <= twitter_mean <= 0.45:
            print(f"  [OK] Mean is in acceptable range (0.30-0.45)")
        else:
            print(f"  [WARNING] Mean is outside target range (0.30-0.45)")
        
        # Show sample scores
        print(f"\n  Sample scores (first 10):")
        for i, (ticker, score) in enumerate(list(twitter_scores.items())[:10]):
            print(f"    {ticker}: {score:.3f}")
    else:
        print("  [ERROR] No scores generated!")
    
    # Compare with old behavior (would have been ~0.65 mean)
    print("\n3. COMPARISON:")
    print("-" * 60)
    print(f"  Old behavior: Mean ~0.65 (too high for simulated data)")
    print(f"  New Reddit:   Mean {reddit_mean:.3f} (penalized)")
    print(f"  New Twitter:  Mean {twitter_mean:.3f} (penalized)")
    
    if reddit_mean < 0.50 and twitter_mean < 0.50:
        print(f"  [OK] Both means are below 0.50 (simulated data is penalized)")
    else:
        print(f"  [WARNING] One or both means are still too high")
    
    # Check value ranges
    print("\n4. VALUE RANGES:")
    print("-" * 60)
    reddit_in_range = sum(1 for v in reddit_values if 0.15 <= v <= 0.65)
    twitter_in_range = sum(1 for v in twitter_values if 0.15 <= v <= 0.65)
    
    print(f"  Reddit scores in range [0.15, 0.65]: {reddit_in_range}/{len(reddit_values)} ({reddit_in_range/len(reddit_values)*100:.1f}%)")
    print(f"  Twitter scores in range [0.15, 0.65]: {twitter_in_range}/{len(twitter_values)} ({twitter_in_range/len(twitter_values)*100:.1f}%)")
    
    if reddit_in_range == len(reddit_values) and twitter_in_range == len(twitter_values):
        print(f"  [OK] All scores are within expected range")
    else:
        print(f"  [WARNING] Some scores are outside expected range")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    
    return reddit_scores, twitter_scores

if __name__ == "__main__":
    test_fallback_scores()

