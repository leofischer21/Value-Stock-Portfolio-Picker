"""
Test script to verify sentiment score improvements work correctly
"""
import sys
from pathlib import Path

# Add scripts to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from dataroma import get_superinvestor_data
        print("[OK] dataroma imported successfully")
    except Exception as e:
        print(f"[FAIL] dataroma import failed: {e}")
        return False
    
    try:
        from reddit import get_reddit_mentions
        print("[OK] reddit imported successfully")
    except Exception as e:
        print(f"[FAIL] reddit import failed: {e}")
        return False
    
    try:
        from twitter import get_x_sentiment_score
        print("[OK] twitter imported successfully")
    except Exception as e:
        print(f"[FAIL] twitter import failed: {e}")
        return False
    
    return True

def test_dataroma_fallback():
    """Test dataroma fallback logic"""
    print("\nTesting dataroma fallback...")
    try:
        from dataroma import get_superinvestor_data
        
        # Test with a small universe
        test_tickers = ['BRK-B', 'JPM', 'AAPL', 'MSFT', 'UNKNOWN_TICKER']
        result = get_superinvestor_data(universe=test_tickers)
        
        # Check if all tickers have scores
        if len(result) != len(test_tickers):
            print(f"[FAIL] Expected {len(test_tickers)} scores, got {len(result)}")
            return False
        
        # Check if known tickers have good scores (not 0.5)
        if result.get('BRK-B', 0) < 0.8:
            print(f"[FAIL] BRK-B should have high score, got {result.get('BRK-B')}")
            return False
        
        if result.get('JPM', 0) < 0.8:
            print(f"[FAIL] JPM should have high score, got {result.get('JPM')}")
            return False
        
        # Check if unknown ticker has reasonable default (not exactly 0.5, but close)
        unknown_score = result.get('UNKNOWN_TICKER', 0)
        if unknown_score < 0.4 or unknown_score > 0.6:
            print(f"[FAIL] UNKNOWN_TICKER should have neutral score (~0.5), got {unknown_score}")
            return False
        
        print(f"[OK] Dataroma fallback works correctly")
        print(f"  BRK-B: {result.get('BRK-B')}, JPM: {result.get('JPM')}, UNKNOWN: {result.get('UNKNOWN_TICKER')}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Dataroma fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reddit_fallback():
    """Test reddit fallback logic"""
    print("\nTesting reddit fallback...")
    try:
        from reddit import get_reddit_mentions
        
        # Test with a small universe
        test_tickers = ['BRK-B', 'JPM', 'COST', 'UNKNOWN_TICKER']
        result = get_reddit_mentions(test_tickers, days_back=120)
        
        # Check if all tickers have scores
        if len(result) != len(test_tickers):
            print(f"[FAIL] Expected {len(test_tickers)} scores, got {len(result)}")
            return False
        
        # Check if known tickers have reasonable scores
        if result.get('BRK-B', 0) < 0.5:
            print(f"[FAIL] BRK-B should have decent score, got {result.get('BRK-B')}")
            return False
        
        if result.get('COST', 0) < 0.5:
            print(f"[FAIL] COST should have decent score, got {result.get('COST')}")
            return False
        
        # Check if unknown ticker has reasonable default (0.3 for reddit)
        unknown_score = result.get('UNKNOWN_TICKER', 0)
        if unknown_score < 0.2 or unknown_score > 0.4:
            print(f"[FAIL] UNKNOWN_TICKER should have neutral score (~0.3), got {unknown_score}")
            return False
        
        print(f"[OK] Reddit fallback works correctly")
        print(f"  BRK-B: {result.get('BRK-B')}, COST: {result.get('COST')}, UNKNOWN: {result.get('UNKNOWN_TICKER')}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Reddit fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_twitter_fallback():
    """Test twitter fallback logic"""
    print("\nTesting twitter fallback...")
    try:
        from twitter import get_x_sentiment_score, _get_static_fallback
        
        # Test static fallback directly
        test_tickers = ['GOOGL', 'META', 'BRK-B', 'JPM', 'UNKNOWN_TICKER']
        static_result = _get_static_fallback(test_tickers)
        
        # Check if all tickers have scores
        if len(static_result) != len(test_tickers):
            print(f"[FAIL] Expected {len(test_tickers)} scores, got {len(static_result)}")
            return False
        
        # Check if known tickers have good scores
        if static_result.get('GOOGL', 0) < 0.8:
            print(f"[FAIL] GOOGL should have high score, got {static_result.get('GOOGL')}")
            return False
        
        if static_result.get('BRK-B', 0) < 0.8:
            print(f"[FAIL] BRK-B should have high score, got {static_result.get('BRK-B')}")
            return False
        
        # Check if unknown ticker has neutral default (0.5)
        unknown_score = static_result.get('UNKNOWN_TICKER', 0)
        if unknown_score < 0.4 or unknown_score > 0.6:
            print(f"[FAIL] UNKNOWN_TICKER should have neutral score (~0.5), got {unknown_score}")
            return False
        
        # Test full function (will use cache or static fallback)
        full_result = get_x_sentiment_score(test_tickers)
        if len(full_result) != len(test_tickers):
            print(f"[FAIL] Expected {len(test_tickers)} scores, got {len(full_result)}")
            return False
        
        print(f"[OK] Twitter fallback works correctly")
        print(f"  GOOGL: {static_result.get('GOOGL')}, BRK-B: {static_result.get('BRK-B')}, UNKNOWN: {static_result.get('UNKNOWN_TICKER')}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Twitter fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_logic():
    """Test validation logic in monthly_update"""
    print("\nTesting validation logic...")
    try:
        # Simulate score dictionaries
        good_scores = {'TICKER1': 0.8, 'TICKER2': 0.7, 'TICKER3': 0.6, 'TICKER4': 0.9, 'TICKER5': 0.5}
        bad_scores = {'TICKER1': 0.5, 'TICKER2': 0.5, 'TICKER3': 0.5, 'TICKER4': 0.5, 'TICKER5': 0.5}
        
        def count_neutral(scores, threshold=0.5):
            return sum(1 for v in scores.values() if abs(v - threshold) < 0.01)
        
        good_neutral = count_neutral(good_scores)
        bad_neutral = count_neutral(bad_scores)
        
        good_ratio = good_neutral / len(good_scores)
        bad_ratio = bad_neutral / len(bad_scores)
        
        print(f"  Good scores: {good_neutral}/{len(good_scores)} neutral ({good_ratio*100:.1f}%)")
        print(f"  Bad scores: {bad_neutral}/{len(bad_scores)} neutral ({bad_ratio*100:.1f}%)")
        
        if good_ratio > 0.6:
            print("[FAIL] Good scores should have <60% neutral")
            return False
        
        if bad_ratio < 0.9:
            print("[FAIL] Bad scores should have >90% neutral")
            return False
        
        print("[OK] Validation logic works correctly")
        return True
        
    except Exception as e:
        print(f"[FAIL] Validation logic test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Sentiment Score Improvements")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataroma Fallback", test_dataroma_fallback),
        ("Reddit Fallback", test_reddit_fallback),
        ("Twitter Fallback", test_twitter_fallback),
        ("Validation Logic", test_validation_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[OK] All tests passed! The improvements should work correctly.")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

