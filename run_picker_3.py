# Archived duplicate runner — original logic moved into other files. Keep this file as a stub.
"""
This file was a duplicate runner and has been archived to avoid confusion.
Use `run_picker.py` (original) or `run_picker_2.py` (enhanced) as the active entrypoints.
"""

    rows = []
    for tk in universe:
        try:
            ticker = yf.Ticker(tk)
            info = ticker.info

            # 1. P/E vs. 5-Jahres-Durchschnitt
            pe_vs_history_score = fetch_historical_pe(tk) 

            # 2. Insider Käufe (OpenInsider – Robuste Logik)
            insider_score = 0.0
            cached_insider = get_cached_data(tk, "openinsider")
            if cached_insider:
                 insider_score = cached_insider.get('insider_score', 0.0)
            else:
                try:
                    url = f"http://openinsider.com/search?q={tk}"
                    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}, timeout=12)
                    
                    if r.status_code == 200:
                        text = r.text.upper()
                        if "P - " in text or "OPEN MARKET PURCHASE" in text:
                            if any(title in text for title in ["CEO", "CFO", "PRESIDENT", "COO", "DIRECTOR"]):
                                insider_score = 1.0 
                            else:
                                insider_score = 0.7 

                    save_data_to_cache(tk, "openinsider", {'insider_score': insider_score})
                except requests.exceptions.RequestException:
                    insider_score = 0.0
            
            # 3. Analysten-Rating
            rating = info.get('recommendationMean', 3.0) 
            analyst_score = max(0.0, (5 - rating) / 4)

            time.sleep(0.2)

            rows.append({
                'ticker': tk,
                'marketCap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'beta': info.get('beta'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
                
                # Community Scores
                'superinvestor_score': superinvestor.get(tk, 0.0), 
                'reddit_score': reddit.get(tk, 0.0),             
                'x_score': x_sent.get(tk, 0.0),                  
                
                # Neue Scores
                'pe_vs_history_score': pe_vs_history_score,
                'insider_score': insider_score,
                'analyst_score': analyst_score,
            })
        except Exception as e:
            print(f"Fehler beim Laden der Ticker-Daten für {tk}: {e}")
            
    return pd.DataFrame(rows)

# --- 6. COMPUTE SCORES (Mit NaN Fix und neuen Gewichten) ---
def compute_scores(df):
    # Filtern auf Mindest-MarketCap und vorhandene P/E-Daten
    df = df[df['marketCap'] >= MIN_MARKET_CAP].dropna(subset=['trailingPE','forwardPE']).copy()

    # 1. Value Score
    df['value_score'] = (
        df['trailingPE'].rank(ascending=True, pct=True) * 0.6 +
        df['forwardPE'].rank(ascending=True, pct=True) * 0.4
    )

    # 2. Quality Score (KORRIGIERT: Umgang mit NaN in debtToEquity für Finanzwerte)
    median_debt = df['debtToEquity'].median()
    df['debtToEquity_fixed'] = df['debtToEquity'].fillna(median_debt) 

    df['quality_score'] = (
        df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 +
        (1 - df['debtToEquity_fixed'].rank(pct=True)) * 0.3
    )
    df = df.drop(columns=['debtToEquity_fixed'])


    # 3. Community Score (gleichgewichtet)
    df['community_score'] = (
        df['superinvestor_score'] * 0.333 +
        df['reddit_score']        * 0.333 +
        df['x_score']             * 0.334
    )

    # 4. Final Score – (Neue Gewichtung aus den letzten Schritten)
    df['final_score'] = (
        df['value_score']           * 0.30 +
        df['quality_score']         * 0.20 +
        df['community_score']       * 0.30 +
        df['pe_vs_history_score']   * 0.08 +
        df['insider_score']         * 0.05 +
        df['analyst_score']         * 0.05 +
        # HINWEIS: KI-Moat Score ist hier 0, da die Quelle auskommentiert/entfernt wurde.
        # Es muss entweder Hardcoded oder eine funktionierende JSON-Quelle hinzugefügt werden.
        0 # KI-Moat Gewichtung ist 0.07 * 0 = 0, da wir die KI-Datenquelle entfernt haben
    )

    return df.sort_values('final_score', ascending=False)


# --- 7. MAIN FUNKTION (Mit verbesserter Konsolenausgabe) ---
# Import pathlib für get_cached_data/save_data_to_cache
from pathlib import Path
import json 

# --- 7. MAIN FUNKTION (Mit korrigierter Konsolenausgabe) ---
# Import pathlib für get_cached_data/save_data_to_cache
from pathlib import Path
import json 

def main():
    print("Starte erweitertes Value Portfolio (mit Insider, Analysten, P/E-History & Hardcoded Community Scores)\n")
    universe = load_universe()
    print(f"Universe: {len(universe)} Aktien\n")

    df = gather(universe)
    portfolio = compute_scores(df).head(PORTFOLIO_SIZE)
    portfolio = construct_portfolio(portfolio, n=PORTFOLIO_SIZE)
    
    # Füge die Prozentspalte hinzu (Muss VOR dem Kopieren passieren, falls 'weight' nicht im out DataFrame ist)
    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"examples/portfolio_{timestamp}.csv"

    # 1. Spaltenauswahl für die Konsole (enthält auch die berechneten Scores)
    cols_to_show = [
        'ticker', 'marketCap', 'trailingPE', 'forwardPE', 'beta', 'debtToEquity', 
        'value_score', 'quality_score', 'community_score', 
        'insider_score', 'pe_vs_history_score', 'analyst_score', 
        'final_score', 'weight_%'
    ]
    
    # 2. Score-Spalten, die der Pandas Styler auf 3 Nachkommastellen formatieren soll
    score_cols = [
        'value_score', 'quality_score', 'community_score', 
        'superinvestor_score', 'reddit_score', 'x_score',
        'insider_score', 'pe_vs_history_score', 'analyst_score', 
        'final_score'
    ]
    
    # Das DataFrame 'out' ist eine Kopie des 'portfolio' mit den ausgewählten Spalten
    # Wir stellen sicher, dass alle Spalten, die wir später formatieren wollen, vorhanden sind.
    # WICHTIG: Hier müssen die Spalten aus 'score_cols' im 'portfolio'-DF enthalten sein!
    out = portfolio[cols_to_show + [c for c in score_cols if c not in cols_to_show]].copy()
    
    # 3. Formatierung der Spalten (Umwandlung in Strings)
    out['marketCap'] = (out['marketCap']/1e9).round(1).astype(str) + ' Mrd'
    # out['returnOnEquity'] wird hier nicht gezeigt, daher nicht formatiert
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['debtToEquity'] = out['debtToEquity'].round(1)

    print("--- 🎯 Endausgabe Portfolio Scores (Top 20) ---")
    
    # 4. Ausgabe mit Styler-Formatierung für numerische Scores
    # Wir wenden .style.format nur auf die numerischen Spalten an, um den Fehler zu vermeiden
    # NOTE: Wir müssen sicherstellen, dass die Spalten 'superinvestor_score', 'reddit_score', 'x_score' 
    # im DataFrame 'out' vorhanden sind, wenn sie Teil von 'score_cols' sind.
    
    # Da 'superinvestor_score', 'reddit_score', 'x_score' nicht in 'cols_to_show' enthalten sind,
    # entferne ich sie aus 'score_cols' für die Konsolenausgabe, um den Fehler zu vermeiden.
    # WENN du diese Spalten in der Konsole sehen willst, füge sie zu 'cols_to_show' hinzu!
    
    # Reduzierte Score-Spalten, die in der Ausgabe angezeigt werden
    visible_score_cols = [c for c in score_cols if c in cols_to_show]
    
    # Ausgabe, jetzt sollte der KeyError behoben sein
    print(out[cols_to_show].style.format(precision=3, subset=visible_score_cols).to_string(index=False))
    
    print("------------------------------------------------")

    print(f"\nPortfolio-Beta: {portfolio['beta'].mean():.2f}")
    print(f"Durchschn. Forward P/E: {portfolio['forwardPE'].mean():.1f}")

    save_portfolio(portfolio, filename)
    print(f"\nGespeichert → {filename}")