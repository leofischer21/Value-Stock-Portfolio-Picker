"""
Quick test to verify X/Twitter scores are now lower
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts to Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "scripts"))

# Import directly from scripts directory
import importlib.util
spec_reddit = importlib.util.spec_from_file_location("reddit", root_dir / "scripts" / "reddit.py")
reddit_module = importlib.util.module_from_spec(spec_reddit)
sys.modules['cache'] = __import__('scripts.cache', fromlist=['get', 'set'])
spec_reddit.loader.exec_module(reddit_module)

spec_twitter = importlib.util.spec_from_file_location("twitter", root_dir / "scripts" / "twitter.py")
twitter_module = importlib.util.module_from_spec(spec_twitter)
sys.modules['cache'] = __import__('scripts.cache', fromlist=['get', 'set'])
spec_twitter.loader.exec_module(twitter_module)

# Load financials
financials_path = root_dir / "data/financials/latest.csv"
if financials_path.exists():
    financials_df = pd.read_csv(financials_path)
    tickers = financials_df['ticker'].tolist()
    print(f"Loaded {len(tickers)} tickers")
else:
    print("No financials data found")
    sys.exit(1)

# Test X/Twitter scores
print("\nTesting X/Twitter fallback scores...")
x_scores = twitter_module._generate_fallback_scores(tickers, financials_df=financials_df)

x_values = list(x_scores.values())
x_mean = np.mean(x_values)
x_std = np.std(x_values)
x_min = np.min(x_values)
x_max = np.max(x_values)
x_median = np.median(x_values)

print(f"\nX/Twitter Scores:")
print(f"  Mean: {x_mean:.3f} (target: <0.35, ideally ~0.32)")
print(f"  Median: {x_median:.3f}")
print(f"  Std Dev: {x_std:.3f}")
print(f"  Min: {x_min:.3f}")
print(f"  Max: {x_max:.3f}")

if x_mean < 0.35:
    print(f"  [OK] Mean is below 0.35")
else:
    print(f"  [WARNING] Mean is still >= 0.35")

if x_max <= 0.60:
    print(f"  [OK] Max is <= 0.60")
else:
    print(f"  [WARNING] Max is > 0.60")

print(f"\nSample scores (first 15):")
for i, (ticker, score) in enumerate(list(x_scores.items())[:15]):
    print(f"  {ticker}: {score:.3f}")

