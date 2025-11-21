"""
Quick test for fallback score generation
"""
import sys
from pathlib import Path
import pandas as pd

# Add scripts to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

def test_fallback_generation():
    """Test if fallback generation works correctly"""
    print("=" * 60)
    print("Testing Fallback Score Generation")
    print("=" * 60)
    
    # Load financials
    financials_path = ROOT_DIR / "data/financials/latest.csv"
    if not financials_path.exists():
        print(f"[SKIP] Financials file not found: {financials_path}")
        print("       Run monthly_update.py first to generate financials")
        return False
    
    financials_df = pd.read_csv(financials_path)
    print(f"[OK] Loaded financials with {len(financials_df)} tickers")
    
    # Test Reddit fallback generation
    print("\n1. Testing Reddit fallback generation...")
    try:
        from reddit import _generate_fallback_scores
        
        # Test with sample tickers
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'BRK-B', 'JPM', 'COST', 'UNH', 'XOM', 'NVDA', 'TSLA']
        reddit_scores = _generate_fallback_scores(test_tickers, financials_df=financials_df)
        
        print(f"   [OK] Generated {len(reddit_scores)} Reddit fallback scores")
        print(f"   Sample scores:")
        for ticker in test_tickers[:5]:
            score = reddit_scores.get(ticker, 'N/A')
            print(f"     {ticker}: {score}")
        
        # Check if scores are differentiated (not all 0.3)
        neutral_count = sum(1 for v in reddit_scores.values() if abs(v - 0.3) < 0.01)
        neutral_ratio = neutral_count / len(reddit_scores) if len(reddit_scores) > 0 else 1.0
        print(f"   Neutral ratio: {neutral_count}/{len(reddit_scores)} ({neutral_ratio*100:.1f}%)")
        
        if neutral_ratio < 0.5:
            print(f"   [OK] Good differentiation (less than 50% neutral)")
        else:
            print(f"   [WARN] High neutral ratio - might need adjustment")
    except Exception as e:
        print(f"   [FAIL] Reddit fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Twitter fallback generation
    print("\n2. Testing Twitter fallback generation...")
    try:
        from twitter import _generate_fallback_scores
        
        twitter_scores = _generate_fallback_scores(test_tickers, financials_df=financials_df)
        
        print(f"   [OK] Generated {len(twitter_scores)} Twitter fallback scores")
        print(f"   Sample scores:")
        for ticker in test_tickers[:5]:
            score = twitter_scores.get(ticker, 'N/A')
            print(f"     {ticker}: {score}")
        
        # Check if scores are differentiated (not all 0.5)
        neutral_count = sum(1 for v in twitter_scores.values() if abs(v - 0.5) < 0.01)
        neutral_ratio = neutral_count / len(twitter_scores) if len(twitter_scores) > 0 else 1.0
        print(f"   Neutral ratio: {neutral_count}/{len(twitter_scores)} ({neutral_ratio*100:.1f}%)")
        
        if neutral_ratio < 0.5:
            print(f"   [OK] Good differentiation (less than 50% neutral)")
        else:
            print(f"   [WARN] High neutral ratio - might need adjustment")
    except Exception as e:
        print(f"   [FAIL] Twitter fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with all tickers
    print("\n3. Testing with all tickers from financials...")
    try:
        all_tickers = financials_df['ticker'].tolist()
        print(f"   Testing with {len(all_tickers)} tickers...")
        
        reddit_all = _generate_fallback_scores(all_tickers, financials_df=financials_df)
        twitter_all = _generate_fallback_scores(all_tickers, financials_df=financials_df)
        
        reddit_neutral = sum(1 for v in reddit_all.values() if abs(v - 0.3) < 0.01)
        twitter_neutral = sum(1 for v in twitter_all.values() if abs(v - 0.5) < 0.01)
        
        reddit_ratio = reddit_neutral / len(reddit_all) if len(reddit_all) > 0 else 1.0
        twitter_ratio = twitter_neutral / len(twitter_all) if len(twitter_all) > 0 else 1.0
        
        print(f"   Reddit: {reddit_neutral}/{len(reddit_all)} neutral ({reddit_ratio*100:.1f}%)")
        print(f"   Twitter: {twitter_neutral}/{len(twitter_all)} neutral ({twitter_ratio*100:.1f}%)")
        
        if reddit_ratio < 0.3 and twitter_ratio < 0.3:
            print(f"   [OK] Excellent! Both below 30% neutral")
            return True
        elif reddit_ratio < 0.5 and twitter_ratio < 0.5:
            print(f"   [OK] Good! Both below 50% neutral")
            return True
        else:
            print(f"   [WARN] Still high neutral ratios - might need further adjustment")
            return False
    except Exception as e:
        print(f"   [FAIL] Full ticker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fallback_generation()
    sys.exit(0 if success else 1)

