# src/run_picker.py
import yfinance as yf
import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from scripts.portfolio import construct_portfolio, save_portfolio
import webbrowser 

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20

def load_universe():
    csv_path = ROOT_DIR / "data/tickers/tickers_over_50B_full.csv"
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            tickers = df['ticker'].dropna().str.strip().tolist()
            print(f"Universe loaded: {len(tickers)} large-cap stocks (>50 bn USD)")
            return tickers
        except Exception as e:
            print(f"Error reading CSV: {e} → fallback to 20 stocks")
    
    # Fallback if file missing
    print("large_cap_universe.csv not found → fallback to 20 stocks")
    return ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V",
            "MA","UNH","PG","HD","KO","PEP","ADBE","CRM","ORCL","CSCO"]

def gather(universe):
    print("Lade Fundamentaldaten + erweiterte Signale...")

    # Community + KI-Moat laden
    community_path = ROOT_DIR / "data/community_data/community_signals.json"
    ki_path = ROOT_DIR / "data/community_data/ai_moat.json"
    community = json.load(open(community_path)) if community_path.exists() else {}
    ki_data = json.load(open(ki_path)) if ki_path.exists() else {"ki_moat_score": {}}

    superinvestor = community.get("superinvestor_score", {})
    reddit = community.get("reddit_score", {})
    x_sent = community.get("x_score", {})
    ki_moat = ki_data.get("ki_moat_score", {})

    rows = []
    for tk in universe:
        try:
            ticker = yf.Ticker(tk)
            info = ticker.info
            hist = ticker.history(period="5y")

            # 1. P/E vs. 5-Jahres-Durchschnitt
            current_pe = info.get('trailingPE')
            if current_pe and len(hist) > 1000:
                eps = info.get('trailingEps')
                if eps and eps > 0:
                    hist_price = hist['Close']
                    hist_pe = hist_price / eps
                    avg_5y_pe = hist_pe.mean()
                    pe_ratio = current_pe / avg_5y_pe
                    pe_vs_history_score = max(0.0, min(1.0, 1.5 - pe_ratio))  # 1.0 = extrem günstig
                else:
                    pe_vs_history_score = 0.5
            else:
                pe_vs_history_score = 0.5

            # 2. Insider Käufe (OpenInsider – leicht & legal)
            insider_score = 0.0
            try:
                url = f"http://openinsider.com/search?q={tk}"
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
                text = r.text.upper()
                if "P - " in text or "OPEN MARKET PURCHASE" in text:
                    if any(title in text for title in ["CEO", "CFO", "PRESIDENT", "COO"]):
                        insider_score = 1.0
                    else:
                        insider_score = 0.7
            except:
                pass

            # 3. Analysten-Rating
            rating = info.get('recommendationMean', 3.0)  # 1 = Strong Buy
            analyst_score = max(0.0, (5 - rating) / 4)

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
                'ki_moat_score': ki_moat.get(tk, 0.5),
                'pe_vs_history_score': pe_vs_history_score,
                'insider_score': insider_score,
                'analyst_score': analyst_score,
            })
        except Exception as e:
            print(f"Fehler bei {tk}: {e}")
    return pd.DataFrame(rows)

def compute_scores(df):
    df = df[df['marketCap'] >= MIN_MARKET_CAP].dropna(subset=['trailingPE','forwardPE']).copy()

    df['value_score'] = (
        df['trailingPE'].rank(ascending=True, pct=True) * 0.6 +
        df['forwardPE'].rank(ascending=True, pct=True) * 0.4
    )
    df['quality_score'] = (
        df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 +
        (1 - df['debtToEquity'].rank(pct=True)) * 0.3
    )
    df['community_score'] = (
        df['superinvestor_score'] * 0.333 +
        df['reddit_score']       * 0.333 +
        df['x_score']            * 0.334
    )

    df['final_score'] = (
        df['value_score']           * 0.30 +
        df['quality_score']         * 0.20 +
        df['community_score']       * 0.30 +
        df['pe_vs_history_score']   * 0.08 +
        df['insider_score']         * 0.05 +
        df['analyst_score']         * 0.05 +
        df['ki_moat_score']         * 0.07
    )

    return df.sort_values('final_score', ascending=False)

def main():
    print("Starte Deep Value Portfolio (mit Insider, Analysten, P/E-History & KI-Moat)\n")
    universe = load_universe()
    print(f"Universe: {len(universe)} Aktien\n")

    df = gather(universe)
    portfolio = compute_scores(df).head(PORTFOLIO_SIZE)
    portfolio = construct_portfolio(portfolio, n=PORTFOLIO_SIZE)
    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.csv")

    out = portfolio[['ticker','marketCap','sector','trailingPE','forwardPE','beta',
                     'returnOnEquity','debtToEquity','superinvestor_score',
                     'reddit_score','x_score','ki_moat_score',
                     'pe_vs_history_score','insider_score','analyst_score',
                     'final_score','weight_%']].copy()

    out['marketCap'] = (out['marketCap']/1e9).round(1).astype(str) + ' Mrd'
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['returnOnEquity'] = (out['returnOnEquity']*100).round(1).astype(str) + '%'
    out['debtToEquity'] = out['debtToEquity'].round(1)
    for col in ['superinvestor_score','reddit_score','x_score','ki_moat_score',
                'pe_vs_history_score','insider_score','analyst_score','final_score']:
        out[col] = out[col].round(3)

    print(out.to_string(index=False))
    print(f"\nPortfolio-Beta: {portfolio['beta'].mean():.2f}")
    print(f"Durchschn. Forward P/E: {portfolio['forwardPE'].mean():.1f}")

    save_portfolio(portfolio, filename)
    print(f"\nGespeichert → {filename}")

    # === VISUELLE HTML-ANSICHT GENERIEREN & IM BROWSER ÖFFNEN ===
    html_filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.html")
    html_path = Path(html_filename)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Value Portfolio – {timestamp}</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; color: #333; }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin: 25px 0; background: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #2c3e50; color: white; }}
            tr:hover {{ background: #f0f8ff; }}
            .weight {{ font-weight: bold; color: #27ae60; }}
            .metric {{ font-size: 1.2em; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Value Stock Portfolio</h1>
        <p><strong>Generiert am:</strong> {datetime.now().strftime("%d.%m.%Y %H:%M")}</p>
        <p class="metric"><strong>Portfolio-Beta:</strong> {portfolio['beta'].mean():.2f} | 
        <strong>Forward P/E:</strong> {portfolio['forwardPE'].mean():.1f}</p>

        {out.to_html(index=False, classes="table", border=0)}

        <p><em>Dein Deep Value + Community Moat Portfolio – made by Leo</em></p>
    </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")
    print(f"HTML-Ansicht erstellt → {html_filename}")

    # Browser öffnen
    import subprocess
    import os

    # Öffnet die HTML-Datei garantiert im Standard-Browser (funktioniert auf Windows, Mac, Linux)
    html_full_path = str(Path(html_filename).resolve())
    print(f"Öffne Portfolio im Browser...")
    os.startfile(html_full_path)
    print("Erfolg! Dein schönes Portfolio ist jetzt im Browser offen.")

if __name__ == '__main__':
    main()