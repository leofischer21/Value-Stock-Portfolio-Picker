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
import os
from scripts.portfolio import construct_portfolio, save_portfolio

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20

def load_universe():
    csv_path = ROOT_DIR / "data/tickers/tickers_over_50B_full.csv"  # <-- deine große CSV hier
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            tickers = df['ticker'].dropna().str.strip().tolist()
            print(f"Universe geladen: {len(tickers)} Large-Cap-Aktien (>30 Mrd USD)")
            return tickers
        except Exception as e:
            print(f"Fehler beim Lesen der CSV: {e}")
    
    # Fallback
    print("CSV nicht gefunden → fallback auf 20 Aktien")
    return ["AAPL","MSFT","AMZN","GOOGL","NVDA","META","BRK-B","TSLA","JPM","V",
            "MA","UNH","PG","HD","KO","PEP","ADBE","CRM","ORCL","CSCO"]

def gather(universe):
    print(f"Lade Daten für {len(universe)} Aktien...")

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

            # P/E vs. 5-Jahres-Durchschnitt
            current_pe = info.get('trailingPE')
            if current_pe and len(hist) > 1000:
                eps = info.get('trailingEps', 0)
                if eps > 0:
                    hist_pe = hist['Close'] / eps
                    avg_5y_pe = hist_pe.mean()
                    pe_ratio = current_pe / avg_5y_pe
                    pe_vs_history_score = max(0.0, min(1.0, 1.5 - pe_ratio))
                else:
                    pe_vs_history_score = 0.5
            else:
                pe_vs_history_score = 0.5

            # Insider Käufe
            insider_score = 0.0
            try:
                url = f"http://openinsider.com/search?q={tk}"
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
                if "P -" in r.text.upper() or "OPEN MARKET PURCHASE" in r.text.upper():
                    insider_score = 1.0 if any(x in r.text.upper() for x in ["CEO", "CFO", "PRESIDENT"]) else 0.7
            except:
                pass

            # Analysten-Rating
            rating = info.get('recommendationMean', 3.0)
            analyst_score = max(0.0, (5 - rating) / 4)

            time.sleep(0.4)
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

    df['value_score'] = df['trailingPE'].rank(ascending=True, pct=True) * 0.6 + df['forwardPE'].rank(ascending=True, pct=True) * 0.4
    df['quality_score'] = df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 + (1 - df['debtToEquity'].rank(pct=True)) * 0.3
    df['community_score'] = df['superinvestor_score'] * 0.333 + df['reddit_score'] * 0.333 + df['x_score'] * 0.334

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
    print("Starte Value Stock Portfolio Picker (400+ Aktien, alle Signale)\n")
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

    # === HTML + Browser öffnen ===
    html_filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.html")
    html = f"""
    <!DOCTYPE html>
    <html><head><title>Value Portfolio {timestamp}</title>
    <meta charset="utf-8">
    <style>
        body {{font-family:Arial;margin:40px;background:#f8f9fa;color:#333;}}
        h1 {{color:#2c3e50;}}
        table {{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
        th,td {{padding:12px;text-align:left;border-bottom:1px solid #ddd;}}
        th {{background:#2c3e50;color:white;}}
        tr:hover {{background:#f0f8ff;}}
    </style>
    </head><body>
    <h1>Value Stock Portfolio</h1>
    <p><strong>{datetime.now().strftime("%d.%m.%Y %H:%M")}</strong> – {len(universe)} Aktien gescreent</p>
    <p><strong>Beta:</strong> {portfolio['beta'].mean():.2f} | <strong>Fwd P/E:</strong> {portfolio['forwardPE'].mean():.1f}</p>
    {out.to_html(index=False, border=0)}
    </body></html>
    """
    Path(html_filename).write_text(html, encoding="utf-8")
    print(f"HTML erstellt → {html_filename}")

    # Browser öffnen (Windows-sicher)
    try:
        os.startfile(Path(html_filename).resolve())
        print("Browser geöffnet!")
    except:
        print("Browser konnte nicht geöffnet werden – öffne manuell die HTML-Datei")

if __name__ == '__main__':
    main()