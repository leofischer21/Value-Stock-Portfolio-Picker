"""Analyze sentiment scores after update"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent
scores = json.load(open(ROOT_DIR / 'data/scores/latest.json'))
si = scores['superinvestor_score']
reddit = scores['reddit_score']
x = scores['x_score']

print('=== SENTIMENT SCORES ANALYSIS ===')
print(f'\nSuperinvestor: {len(si)} tickers')
si_neutral = sum(1 for v in si.values() if abs(v - 0.5) < 0.01)
print(f'  Neutral (0.5): {si_neutral}/{len(si)} ({si_neutral/len(si)*100:.1f}%)')
print(f'  Sample: BRK-B={si.get("BRK-B", 0):.3f}, JPM={si.get("JPM", 0):.3f}, AAPL={si.get("AAPL", 0):.3f}')

print(f'\nReddit: {len(reddit)} tickers')
reddit_neutral = sum(1 for v in reddit.values() if abs(v - 0.3) < 0.01)
print(f'  Neutral (0.3): {reddit_neutral}/{len(reddit)} ({reddit_neutral/len(reddit)*100:.1f}%)')
print(f'  Sample: BRK-B={reddit.get("BRK-B", 0):.3f}, JPM={reddit.get("JPM", 0):.3f}, AAPL={reddit.get("AAPL", 0):.3f}, COST={reddit.get("COST", 0):.3f}')

print(f'\nX/Twitter: {len(x)} tickers')
x_neutral = sum(1 for v in x.values() if abs(v - 0.5) < 0.01)
print(f'  Neutral (0.5): {x_neutral}/{len(x)} ({x_neutral/len(x)*100:.1f}%)')
print(f'  Sample: GOOGL={x.get("GOOGL", 0):.3f}, BRK-B={x.get("BRK-B", 0):.3f}, META={x.get("META", 0):.3f}, AAPL={x.get("AAPL", 0):.3f}')

# Show score distribution
print('\n=== SCORE DISTRIBUTION ===')
print('\nReddit Score Ranges:')
ranges = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
for v in reddit.values():
    if v < 0.3:
        ranges['0.0-0.3'] += 1
    elif v < 0.5:
        ranges['0.3-0.5'] += 1
    elif v < 0.7:
        ranges['0.5-0.7'] += 1
    elif v < 0.9:
        ranges['0.7-0.9'] += 1
    else:
        ranges['0.9-1.0'] += 1
for k, v in ranges.items():
    print(f'  {k}: {v} ({v/len(reddit)*100:.1f}%)')

print('\nX/Twitter Score Ranges:')
x_ranges = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
for v in x.values():
    if v < 0.3:
        x_ranges['0.0-0.3'] += 1
    elif v < 0.5:
        x_ranges['0.3-0.5'] += 1
    elif v < 0.7:
        x_ranges['0.5-0.7'] += 1
    elif v < 0.9:
        x_ranges['0.7-0.9'] += 1
    else:
        x_ranges['0.9-1.0'] += 1
for k, v in x_ranges.items():
    print(f'  {k}: {v} ({v/len(x)*100:.1f}%)')

