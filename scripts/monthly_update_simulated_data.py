# monthly_update.py – 100 % offline, 100 % stabil, 100 % realistisch
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import random

ROOT_DIR = Path(__file__).parent.parent
np.random.seed(42)  # Reproduzierbar
random.seed(42)

# Deine echte Ticker-Liste (aus deiner CSV)
UNIVERSE_CSV = ROOT_DIR / "data" / "tickers_over_50B_full.csv"

def load_real_universe():
    
    if UNIVERSE_CSV.exists():
        df = pd.read_csv(UNIVERSE_CSV)
        return df['ticker'].dropna().tolist()
    else:
        print("large_cap_universe.csv nicht gefunden → fallback")
        return ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V"]

def simulate_financial_data(tickers):
    data = []
    for t in tickers:
        # Realistische Verteilung
        pe = np.random.lognormal(mean=2.8, sigma=0.6)  # 8–50
        fwd_pe = pe * np.random.uniform(0.7, 1.2)
        roe = np.random.beta(5, 2) * 0.4  # 0–40%
        debt = np.random.lognormal(3, 1)  # 0–1000+
        beta = np.random.beta(2, 3) + 0.3  # 0.3–2.0
        
        data.append({
            "ticker": t,
            "trailingPE": round(pe, 1),
            "forwardPE": round(fwd_pe, 1),
            "returnOnEquity": round(roe, 3),
            "debtToEquity": round(debt, 1),
            "beta": round(beta, 2),
        })
    return pd.DataFrame(data)

def simulate_community_scores(tickers):
    # Bekannte Favoriten bekommen hohe Scores
    superinvestor_base = {"BRK-B": 1.0, "JPM": 0.98, "GOOGL": 0.92, "META": 0.90, "UNH": 0.88}
    reddit_base = {"TGT": 0.95, "COF": 0.90, "SBUX": 0.88, "COST": 0.85, "JPM": 0.82}
    x_base = {"GOOGL": 0.96, "META": 0.94, "COST": 0.95, "BRK-B": 0.90, "KO": 0.88}
    
    superinvestor = {t: superinvestor_base.get(t, round(np.random.uniform(0.0, 0.7), 2)) for t in tickers}
    reddit = {t: reddit_base.get(t, round(np.random.uniform(0.0, 0.6), 2)) for t in tickers}
    x_sent = {t: x_base.get(t, round(np.random.uniform(0.3, 0.8), 2)) for t in tickers}
    
    return superinvestor, reddit, x_sent

def main():
    
    # <<< ADD THESE TWO LINES >>>
    Path("data/financials").mkdir(parents=True, exist_ok=True)
    Path("data/scores").mkdir(parents=True, exist_ok=True)
    # <<< END ADD >>>

    print("Starte monatlichen SIMULIERTEN Update (100 % offline & stabil)\n")
    
    # 1. Universe laden
    tickers = load_real_universe()
    print(f"Universe: {len(tickers)} Aktien (>50 Mrd)")

    # 2. Finanzdaten simulieren
    financial_df = simulate_financial_data(tickers)
    
    # 3. Community simulieren
    superinvestor, reddit, x_sent = simulate_community_scores(tickers)

    # 4. Speichern
    month = datetime.now().strftime("%Y-%m")
    
    # Finanzdaten
    Path("data/financials").mkdir(parents=True, exist_ok=True)
    financial_df.to_csv(f"data/financials/{month}.csv", index=False)
    financial_df.to_csv("data/financials/latest.csv", index=False)
    
    # Community
    scores = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "superinvestor_score": superinvestor,
        "reddit_score": reddit,
        "x_score": x_sent
    }
    Path("data/scores").mkdir(parents=True, exist_ok=True)
    with open(f"data/scores/{month}.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open("data/scores/latest.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    print(f"\nUpdate abgeschlossen für {month}")
    print(f"→ {len(tickers)} Aktien, Finanzdaten + Community-Scores gespeichert")

if __name__ == '__main__':
    main()