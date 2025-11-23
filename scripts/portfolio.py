# src/portfolio.py
import pandas as pd


def construct_portfolio(df, n=20, max_weight=0.06, sector_cap=0.30, beta_min=0.7, beta_max=1.3):
    """Simple construction: pick top-n by final_score, apply naive sector cap, equal-weighting with clipping, and beta constraint.
    Returns a DataFrame of selected tickers with target weights.
    """
    df_sorted = df.sort_values('final_score', ascending=False).head(n).copy()
    
    k = len(df_sorted)
    
    # 🌟 KORREKTUR: ZeroDivisionError vermeiden 🌟
    if k == 0:
        print("Warnung: Leeres Portfolio (0 Aktien) nach Scoring.")
        # Gibt ein leeres DataFrame mit den erwarteten Spalten zurück
        return pd.DataFrame(columns=df_sorted.columns.tolist() + ['weight'])
    # 🌟 ENDE KORREKTUR 🌟
    
    # initial equal weight
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

    # Improved sector diversification: enforce sector cap and improve diversification
    max_sector_iterations = 15
    sector_iteration = 0
    
    while sector_iteration < max_sector_iterations:
        sectors = df_sorted.groupby('sector')['weight'].sum()
        over_cap_sectors = sectors[sectors > sector_cap]
        
        if over_cap_sectors.empty:
            # All sectors are within cap, check if we can improve diversification
            # If one sector dominates (>50% of portfolio), try to diversify
            if len(sectors) > 0 and sectors.max() > 0.5:
                # Find the dominant sector
                dominant_sector = sectors.idxmax()
                dominant_weight = sectors.max()
                
                # Find candidates from other sectors
                selected_tickers = set(df_sorted['ticker'].values)
                candidates = df[~df['ticker'].isin(selected_tickers)].copy()
                
                # Filter candidates from other sectors
                other_sector_candidates = candidates[candidates['sector'] != dominant_sector].copy()
                
                if len(other_sector_candidates) > 0:
                    # Replace one stock from dominant sector with best candidate from other sectors
                    dominant_stocks = df_sorted[df_sorted['sector'] == dominant_sector]
                    if len(dominant_stocks) > 1:  # Keep at least one from dominant sector
                        # Replace the lowest scoring stock from dominant sector
                        to_replace_idx = dominant_stocks['final_score'].idxmin()
                        replacement = other_sector_candidates.sort_values('final_score', ascending=False).iloc[0]
                        
                        # Replace
                        df_sorted.loc[to_replace_idx, 'ticker'] = replacement['ticker']
                        for col in df_sorted.columns:
                            if col != 'ticker' and col != 'weight':
                                df_sorted.loc[to_replace_idx, col] = replacement.get(col, df_sorted.loc[to_replace_idx, col])
                        
                        # Renormalize weights
                        df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
                        sector_iteration += 1
                        continue
            break  # Diversification is good enough
        
        # Handle sectors over cap
        for sec, total in over_cap_sectors.items():
            # First try: downweight proportionally
            factor = sector_cap / total
            mask = df_sorted['sector'] == sec
            df_sorted.loc[mask, 'weight'] *= factor
            df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
            
            # Second try: if still over cap after downweighting, replace one stock
            sectors_after = df_sorted.groupby('sector')['weight'].sum()
            if sectors_after.get(sec, 0) > sector_cap:
                # Find candidates from other sectors
                selected_tickers = set(df_sorted['ticker'].values)
                candidates = df[~df['ticker'].isin(selected_tickers)].copy()
                
                # Filter candidates from other sectors
                other_sector_candidates = candidates[candidates['sector'] != sec].copy()
                
                if len(other_sector_candidates) > 0:
                    # Replace one stock from over-cap sector with best candidate from other sectors
                    over_cap_stocks = df_sorted[df_sorted['sector'] == sec]
                    if len(over_cap_stocks) > 1:  # Keep at least one from this sector
                        # Replace the lowest scoring stock from over-cap sector
                        to_replace_idx = over_cap_stocks['final_score'].idxmin()
                        replacement = other_sector_candidates.sort_values('final_score', ascending=False).iloc[0]
                        
                        # Replace
                        df_sorted.loc[to_replace_idx, 'ticker'] = replacement['ticker']
                        for col in df_sorted.columns:
                            if col != 'ticker' and col != 'weight':
                                df_sorted.loc[to_replace_idx, col] = replacement.get(col, df_sorted.loc[to_replace_idx, col])
                        
                        # Renormalize weights
                        df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
        
        sector_iteration += 1

    # Beta constraint enforcement
    if 'beta' in df_sorted.columns:
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Calculate portfolio beta
            df_sorted['beta_filled'] = df_sorted['beta'].fillna(1.0)  # Default to market beta if missing
            portfolio_beta = (df_sorted['beta_filled'] * df_sorted['weight']).sum()
            
            if beta_min <= portfolio_beta <= beta_max:
                # Beta is in target range
                df_sorted = df_sorted.drop(columns=['beta_filled'])
                break
            
            # Find candidates for replacement
            selected_tickers = set(df_sorted['ticker'].values)
            candidates = df[~df['ticker'].isin(selected_tickers)].copy()
            
            if len(candidates) == 0:
                # No candidates available, break
                df_sorted = df_sorted.drop(columns=['beta_filled'])
                break
            
            # Sort candidates by final_score (descending)
            candidates = candidates.sort_values('final_score', ascending=False)
            
            if portfolio_beta < beta_min:
                # Need higher beta: replace lowest beta stock with higher beta candidate
                df_sorted = df_sorted.sort_values('beta_filled', ascending=True)
                lowest_beta_idx = df_sorted.index[0]
                lowest_beta_ticker = df_sorted.loc[lowest_beta_idx, 'ticker']
                
                # Find candidate with higher beta and similar score
                candidates['beta_filled'] = candidates['beta'].fillna(1.0)
                higher_beta_candidates = candidates[candidates['beta_filled'] > df_sorted.loc[lowest_beta_idx, 'beta_filled']]
                
                if len(higher_beta_candidates) > 0:
                    # Take best scoring candidate with higher beta
                    replacement = higher_beta_candidates.iloc[0]
                    df_sorted.loc[lowest_beta_idx, 'ticker'] = replacement['ticker']
                    for col in df_sorted.columns:
                        if col != 'ticker' and col != 'weight' and col != 'beta_filled':
                            df_sorted.loc[lowest_beta_idx, col] = replacement.get(col, df_sorted.loc[lowest_beta_idx, col])
                    # Renormalize weights
                    df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
                else:
                    # No suitable replacement found
                    df_sorted = df_sorted.drop(columns=['beta_filled'])
                    break
                    
            elif portfolio_beta > beta_max:
                # Need lower beta: replace highest beta stock with lower beta candidate
                df_sorted = df_sorted.sort_values('beta_filled', ascending=False)
                highest_beta_idx = df_sorted.index[0]
                highest_beta_ticker = df_sorted.loc[highest_beta_idx, 'ticker']
                
                # Find candidate with lower beta and similar score
                candidates['beta_filled'] = candidates['beta'].fillna(1.0)
                lower_beta_candidates = candidates[candidates['beta_filled'] < df_sorted.loc[highest_beta_idx, 'beta_filled']]
                
                if len(lower_beta_candidates) > 0:
                    # Take best scoring candidate with lower beta
                    replacement = lower_beta_candidates.iloc[0]
                    df_sorted.loc[highest_beta_idx, 'ticker'] = replacement['ticker']
                    for col in df_sorted.columns:
                        if col != 'ticker' and col != 'weight' and col != 'beta_filled':
                            df_sorted.loc[highest_beta_idx, col] = replacement.get(col, df_sorted.loc[highest_beta_idx, col])
                    # Renormalize weights
                    df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
                else:
                    # No suitable replacement found
                    df_sorted = df_sorted.drop(columns=['beta_filled'])
                    break
            
            iteration += 1
        
        # Clean up temporary column if still present
        if 'beta_filled' in df_sorted.columns:
            df_sorted = df_sorted.drop(columns=['beta_filled'])

    # Final sector cap enforcement: Ensure no sector exceeds 30% after all constraints
    # This is a hard constraint - if a sector is over cap, we must reduce it
    max_final_iterations = 20
    final_iteration = 0
    
    while final_iteration < max_final_iterations:
        sectors = df_sorted.groupby('sector')['weight'].sum()
        over_cap_sectors = sectors[sectors > sector_cap]
        
        if over_cap_sectors.empty:
            break  # All sectors are within cap
        
        # Force reduction: replace stocks from over-cap sectors
        for sec, total in over_cap_sectors.items():
            # Find candidates from other sectors
            selected_tickers = set(df_sorted['ticker'].values)
            candidates = df[~df['ticker'].isin(selected_tickers)].copy()
            
            # Filter candidates from other sectors
            other_sector_candidates = candidates[candidates['sector'] != sec].copy()
            
            if len(other_sector_candidates) > 0:
                # Replace one stock from over-cap sector with best candidate from other sectors
                over_cap_stocks = df_sorted[df_sorted['sector'] == sec]
                if len(over_cap_stocks) > 1:  # Keep at least one from this sector
                    # Replace the lowest scoring stock from over-cap sector
                    to_replace_idx = over_cap_stocks['final_score'].idxmin()
                    replacement = other_sector_candidates.sort_values('final_score', ascending=False).iloc[0]
                    
                    # Replace
                    df_sorted.loc[to_replace_idx, 'ticker'] = replacement['ticker']
                    for col in df_sorted.columns:
                        if col != 'ticker' and col != 'weight':
                            df_sorted.loc[to_replace_idx, col] = replacement.get(col, df_sorted.loc[to_replace_idx, col])
                    
                    # Renormalize weights
                    df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
                else:
                    # Only one stock in this sector, must downweight it
                    factor = sector_cap / total
                    mask = df_sorted['sector'] == sec
                    df_sorted.loc[mask, 'weight'] *= factor
                    df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
            else:
                # No candidates available, must downweight
                factor = sector_cap / total
                mask = df_sorted['sector'] == sec
                df_sorted.loc[mask, 'weight'] *= factor
                df_sorted['weight'] = df_sorted['weight'] / df_sorted['weight'].sum()
        
        final_iteration += 1

    return df_sorted.reset_index(drop=True)


def should_replace(old_pos, new_candidate, threshold=0.10):
    """Decide whether to replace old_pos (row with 'final_score') with new_candidate.
    Replace only if new_candidate.final_score > old_pos.final_score * (1 + threshold)
    """
    return new_candidate['final_score'] > old_pos['final_score'] * (1 + threshold)


def save_portfolio(df, path):
    df.to_csv(path, index=False)