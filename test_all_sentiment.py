"""
Quick test for all sentiment improvements
"""
import sys
from pathlib import Path

# Add scripts to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

def test_all():
    """Test all sentiment functions"""
    print("=" * 60)
    print("Testing All Sentiment Score Improvements")
    print("=" * 60)
    
    test_tickers = ['BRK-B', 'JPM', 'COST', 'AAPL', 'MSFT', 'GOOGL', 'META', 'UNH']
    
    # Test Dataroma
    print("\n1. Testing Dataroma...")
    try:
        from dataroma import get_superinvestor_data
        dataroma_result = get_superinvestor_data(universe=test_tickers)
        print(f"   [OK] Dataroma returned {len(dataroma_result)} scores")
        print(f"   Sample: BRK-B={dataroma_result.get('BRK-B'):.3f}, JPM={dataroma_result.get('JPM'):.3f}")
        
        # Check if values are reasonable (not all 0.5)
        neutral_count = sum(1 for v in dataroma_result.values() if abs(v - 0.5) < 0.01)
        if neutral_count < len(dataroma_result) * 0.5:
            print(f"   [OK] Only {neutral_count}/{len(dataroma_result)} neutral values (good)")
        else:
            print(f"   [WARN] {neutral_count}/{len(dataroma_result)} neutral values (might need improvement)")
    except Exception as e:
        print(f"   [FAIL] Dataroma test failed: {e}")
        return False
    
    # Test Reddit
    print("\n2. Testing Reddit...")
    try:
        from reddit import get_reddit_mentions
        reddit_result = get_reddit_mentions(test_tickers, days_back=120)
        print(f"   [OK] Reddit returned {len(reddit_result)} scores")
        print(f"   Sample: BRK-B={reddit_result.get('BRK-B'):.3f}, COST={reddit_result.get('COST'):.3f}, AAPL={reddit_result.get('AAPL'):.3f}")
        
        # Check if values are reasonable (not all 0.3)
        neutral_count = sum(1 for v in reddit_result.values() if abs(v - 0.3) < 0.01)
        if neutral_count < len(reddit_result) * 0.5:
            print(f"   [OK] Only {neutral_count}/{len(reddit_result)} neutral values (good)")
        else:
            print(f"   [WARN] {neutral_count}/{len(reddit_result)} neutral values (might need improvement)")
        
        # Check if values are averaged (should be between scraped and fallback)
        brk_score = reddit_result.get('BRK-B', 0)
        if 0.4 <= brk_score <= 0.8:
            print(f"   [OK] BRK-B score ({brk_score:.3f}) is in reasonable range (averaged)")
        else:
            print(f"   [WARN] BRK-B score ({brk_score:.3f}) might be too extreme")
    except Exception as e:
        print(f"   [FAIL] Reddit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Twitter
    print("\n3. Testing Twitter...")
    try:
        from twitter import get_x_sentiment_score
        twitter_result = get_x_sentiment_score(test_tickers)
        print(f"   [OK] Twitter returned {len(twitter_result)} scores")
        print(f"   Sample: GOOGL={twitter_result.get('GOOGL'):.3f}, BRK-B={twitter_result.get('BRK-B'):.3f}, META={twitter_result.get('META'):.3f}")
        
        # Check if values are reasonable (not all 0.5)
        neutral_count = sum(1 for v in twitter_result.values() if abs(v - 0.5) < 0.01)
        if neutral_count < len(twitter_result) * 0.5:
            print(f"   [OK] Only {neutral_count}/{len(twitter_result)} neutral values (good)")
        else:
            print(f"   [WARN] {neutral_count}/{len(twitter_result)} neutral values (might need improvement)")
    except Exception as e:
        print(f"   [FAIL] Twitter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)

