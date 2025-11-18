# src/portfolio.py
import pandas as pd


def construct_portfolio(df, n=20, max_weight=0.06, sector_cap=0.30):
    """Simple construction: pick top-n by final_score, apply naive sector cap and equal-weighting with clipping.
    Returns a DataFrame of selected tickers with target weights.
    """
    df_sorted = df.sort_values('final_score', ascending=False).head(n).copy()
    # initial equal weight
    k = len(df_sorted)
    df_sorted['weight'] = 1.0 / k

    # enforce max weight via iterative clipping & renormalization
    while True:
        over = df_sorted[df_sorted['weight'] > max_weight]
        if over.empty:
            break
        # clip those and redistribute
        excess = (df_sorted['weight'] - max_weight).clip(lower=0).sum()
        df_sorted.loc[df_sorted['weight'] > max_weight, 'weight'] = max_weight
        rest = df_sorted['weight'].sum()
        if rest >= 1 - excess:
            df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum() * (1 - excess)
        else:
            # fallback equal weight among non-clipped
            non_clipped = df_sorted['weight'] < max_weight
            n_non = non_clipped.sum()
            if n_non == 0:
                break
            df_sorted.loc[non_clipped, 'weight'] = (1 - max_weight*len(over)) / n_non

    # simple sector cap enforcement (soft): if sector > sector_cap, downweight proportionally
    sectors = df_sorted.groupby('sector')['weight'].sum()
    for sec, total in sectors.items():
        if total > sector_cap:
            factor = sector_cap / total
            mask = df_sorted['sector'] == sec
            df_sorted.loc[mask, 'weight'] *= factor
            # renormalize remaining
            df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()

    return df_sorted.reset_index(drop=True)


def should_replace(old_pos, new_candidate, threshold=0.10):
    """Decide whether to replace old_pos (row with 'final_score') with new_candidate.
    Replace only if new_candidate.final_score > old_pos.final_score * (1 + threshold)
    """
    return new_candidate['final_score'] > old_pos['final_score'] * (1 + threshold)


def save_portfolio(df, path):
    df.to_csv(path, index=False)