import sys, os
sys.path.insert(0, os.path.abspath('.'))
from tests.test_scoring import make_sample_df
from run_picker_2 import compute_scores

df = make_sample_df()
scored = compute_scores(df, min_market_cap=0)
print(scored[['ticker','value_score','quality_score','community_score','final_score']])
