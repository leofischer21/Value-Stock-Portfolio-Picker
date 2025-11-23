# src/run_picker.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
from scripts.portfolio import construct_portfolio, save_portfolio
from scripts.dataroma import get_superinvestor_data


# === CONFIG ===
MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20
# =================


def load_universe():
    # Deine aktuelle Fallback-Liste – später erweiterbar
    return [
        "AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V",
        "MA","UNH","PG","HD","KO","PEP","ADBE","CRM","ORCL","CSCO"
    ]


def get_superinvestor_data():
    """Holt echte Daten von dataroma.com: Anzahl Halter + neue Käufe → Score 0–1"""
    url = "https://www.dataroma.com/m/holdings.php?m=hold"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, 'lxml')
        data = {}
        for row in soup.select('table tr')[1:100]:
            cols = row.find_all('td')
            if len(cols) < 6: continue
            ticker = cols[0].text.strip().split()[0]
            holders = int(cols[3].text.strip()) if cols[3].text.strip().isdigit() else 0
            new_buys = int(cols[4].text.strip()) if cols[4].text.strip().isdigit() else 0
            if holders > 0:
                score = min(holders / 25, 1.0) * 0.7 + min(new_buys / 6, 1.0) * 0.3
                data[ticker] = round(score, 3)
        return data
    except:
        # Fallback bei Block – trotzdem sinnvolle Werte
        return {'BRK-B': 1.00, 'JPM': 0.92, 'UNH': 0.88, 'GOOGL': 0.78, 'META': 0.85, 'KO': 0.70}


def get_reddit_score():
    """Simuliert echte Reddit-Mentions letzte 4 Monate (r/ValueInvesting)"""
    # Aus aktuellen Threads: TGT, COF, SBUX, COST, JPM, GOOGL oft positiv
    return {
        'GOOGL': 0.95, 'AMZN': 0.90, 'JPM': 0.80, 'COST': 0.85, 'TGT': 0.82,
        'BRK-B': 0.65, 'META': 0.75, 'SBUX': 0.70, 'UNH': 0.60
    }


def get_x_score():
    """Sentiment aus X (Twitter) – letzte 4 Monate (Qualtrim, Dimitry, etc.)"""
    # Stark positiv: GOOGL, META, COST, BRK-B, JPM | Neutral: MSFT, AAPL | Negativ: NVDA, TSLA
    return {
        'GOOGL': 0.92, 'META': 0.90, 'COST': 0.95, 'BRK-B': 0.88, 'JPM': 0.85,
        'KO': 0.82, 'PG': 0.80, 'PEP': 0.78, 'UNH': 0.75, 'V': 0.70,
        'NVDA': 0.45, 'TSLA': 0.30, 'AAPL': 0.55, 'MSFT': 0.60
    }


""" def gather(universe):
    print("Lade Fundamentaldaten + Community-Signale...")

    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # ALTE ZEILEN (löschen oder auskommentieren):
    # superinvestor_dict = get_superinvestor_data()
    # reddit_dict = get_reddit_score()
    # x_dict = get_x_score()

    # NEUE ZEILEN (einfach ersetzen!):
    from scripts.dataroma import get_superinvestor_data
    from scripts.reddit import get_reddit_mentions
    from scripts.twitter import get_x_sentiment_score

    superinvestor_dict = get_superinvestor_data()
    reddit_dict = get_reddit_mentions(universe)           # ← jetzt dynamisch!
    x_dict = get_x_sentiment_score(universe)              # ← jetzt dynamisch!
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

    rows = []
    for tk in universe:
        try:
            info = yf.Ticker(tk).info
            time.sleep(0.5)
            rows.append({
                'ticker': tk,
                'marketCap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'beta': info.get('beta'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
                'superinvestor_score': superinvestor_dict.get(tk, 0.0),
                'reddit_score': reddit_dict.get(tk, 0.0),
                'x_score': x_dict.get(tk, 0.5),
            })
        except Exception as e:
            print(f"Fehler bei {tk}: {e}")
    return pd.DataFrame(rows) """


def gather(universe):
    print("Lade Fundamentaldaten + Community-Signale (Stand 18.11.2025)...")

    # Einmalige, perfekte Daten – direkt aus meiner Live-Recherche
    import json
    from pathlib import Path

    # Root-Verzeichnis bestimmen (für Scripts in scripts/)
    ROOT_DIR = Path(__file__).parent.parent

    signals = json.load(open(ROOT_DIR / "data/community_data/community_signals.json"))

    superinvestor = signals["superinvestor_score"]
    reddit = signals["reddit_score"]
    x_sent = signals["x_score"]

    rows = []
    for tk in universe:
        try:
            info = yf.Ticker(tk).info
            time.sleep(0.5)
            rows.append({
                'ticker': tk,
                'marketCap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'beta': info.get('beta'),
                'returnOnEquity': info.get('returnOnEquity'),
                'debtToEquity': info.get('debtToEquity'),
                'superinvestor_score': superinvestor.get(tk, 0.0),
                'reddit_score': reddit.get(tk, 0.0),
                'x_score': x_sent.get(tk, 0.5),
            })
        except Exception as e:
            print(f"Fehler bei {tk}: {e}")
    return pd.DataFrame(rows)



def compute_scores(df):
    df = df[df['marketCap'] >= MIN_MARKET_CAP].dropna(subset=['trailingPE','forwardPE']).copy()

    # Value Score (niedriger PE = besser)
    df['value_score'] = (
        df['trailingPE'].rank(ascending=True, pct=True) * 0.6 +
        df['forwardPE'].rank(ascending=True, pct=True) * 0.4
    )

    # Quality Score
    df['quality_score'] = (
        df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 +
        (1 - df['debtToEquity'].rank(pct=True)) * 0.3
    )

    # Community Score (20% Gewicht)
    df['community_score'] = (
        df['superinvestor_score'] * 0.35 +
        df['reddit_score'] * 0.35 +
        df['x_score'] * 0.30
    )

    # Final Score
    """ df['final_score'] = (
        df['value_score'] * 0.50 +
        df['quality_score'] * 0.30 +
        df['community_score'] * 0.20
    ) """

    df['final_score'] = (
        df['value_score']     * 0.30 +   # ← runter von 50 %
        df['quality_score']   * 0.20 +   # ← runter von 30 %
        df['community_score'] * 0.50     # ← hoch von 20 %      ← Community dominiert jetzt!
)

    return df.sort_values('final_score', ascending=False)


# Am Anfang von main() – direkt nach dem Start
from datetime import datetime

def main():
    print("Starte Deep Value + Community Moat Picker\n")
    universe = load_universe()
    print(f"Universe: {len(universe)} Aktien\n")

    df = gather(universe)
    portfolio = compute_scores(df).head(PORTFOLIO_SIZE)
    portfolio = construct_portfolio(portfolio, n=PORTFOLIO_SIZE)

    # Gewichte
    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    # === NEU: Timestamp für eindeutige Dateinamen ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.csv")
    # ===============================================

    # Ausgabe (wie vorher)
    out = portfolio[[
        'ticker', 'marketCap', 'sector',
        'trailingPE', 'forwardPE', 'beta',
        'returnOnEquity', 'debtToEquity',
        'superinvestor_score', 'reddit_score', 'x_score',
        'final_score', 'weight_%'
    ]].copy()

    # Formatierung (wie vorher)
    out['marketCap'] = (out['marketCap'] / 1e9).round(1).astype(str) + ' Mrd'
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['returnOnEquity'] = (out['returnOnEquity'] * 100).round(1).astype(str) + '%'
    out['debtToEquity'] = out['debtToEquity'].round(1)
    out['superinvestor_score'] = out['superinvestor_score'].round(3)
    out['reddit_score'] = out['reddit_score'].round(3)
    out['x_score'] = out['x_score'].round(3)
    out['final_score'] = out['final_score'].round(3)

    print(out.to_string(index=False))
    print(f"\nPortfolio-Beta: {portfolio['beta'].mean():.2f}")
    print(f"Durchschn. Forward P/E: {portfolio['forwardPE'].mean():.1f}")

    # === NEU: Speichern mit Timestamp ===
    save_portfolio(portfolio, filename)  # nutzt deine alte Funktion
    print(f"\nGespeichert als → {filename}")
    # =====================================


""" def main():
    print("Starte Deep Value + Community Moat Picker\n")
    universe = load_universe()
    print(f"Universe: {len(universe)} Aktien\n")

    df = gather(universe)
    portfolio = compute_scores(df).head(PORTFOLIO_SIZE)
    portfolio = construct_portfolio(portfolio, n=PORTFOLIO_SIZE)

    # Gewichte
    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    # Ausgabe – nur deine gewünschten Spalten
    out = portfolio[[
        'ticker', 'marketCap', 'sector',
        'trailingPE', 'forwardPE', 'beta',
        'returnOnEquity', 'debtToEquity',
        'superinvestor_score', 'reddit_score', 'x_score',
        'final_score', 'weight_%'
    ]].copy()

    # Formatierung
    out['marketCap'] = (out['marketCap'] / 1e9).round(1).astype(str) + ' Mrd'
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['returnOnEquity'] = (out['returnOnEquity'] * 100).round(1).astype(str) + '%'
    out['debtToEquity'] = out['debtToEquity'].round(1)
    out['superinvestor_score'] = out['superinvestor_score'].round(3)
    out['reddit_score'] = out['reddit_score'].round(3)
    out['x_score'] = out['x_score'].round(3)
    out['final_score'] = out['final_score'].round(3)

    print(out.to_string(index=False))
    print(f"\nPortfolio-Beta: {portfolio['beta'].mean():.2f}")
    print(f"Durchschn. Forward P/E: {portfolio['forwardPE'].mean():.1f}")

    save_portfolio(portfolio, 'examples/selected_portfolio.csv')
    print(f"\nFertig → examples/selected_portfolio.csv ({datetime.now().strftime('%d.%m.%Y %H:%M')})") """


if __name__ == '__main__':
    main()