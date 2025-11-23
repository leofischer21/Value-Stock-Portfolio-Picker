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

import sys
from pathlib import Path

# Füge das Projekt-Root-Verzeichnis (den Eltern-Ordner) zum Python-Pfad hinzu
# Dies erlaubt den Import von 'src.portfolio'
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Jetzt kann der Import von src.portfolio funktionieren
from scripts.portfolio import construct_portfolio, save_portfolio
# ... restlicher Code ...

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

MIN_MARKET_CAP = 30_000_000_000
PORTFOLIO_SIZE = 20

# Korrekte Pfade – relativ zum Projekt-Root
base_path = Path(__file__).parent.parent

def load_universe():
    csv_path = base_path / "src" / "data" / "tickers_over_50B_full.csv"
    
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
    print("Lade gespeicherte Daten aus monatlichem Update (offline & blitzschnell)...")

    base_path = Path(__file__).parent.parent
    financial_path = base_path / "data" / "financials" / "latest.csv"
    scores_path = base_path / "data" / "scores" / "latest.json"
    ki_path = base_path / "data" / "ki_moat.json"

    if not financial_path.exists():
        raise FileNotFoundError(f"{financial_path} fehlt – führe monthly_update_simulated_data.py aus!")
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} fehlt – führe monthly_update_simulated_data.py aus!")

    # Laden und Index setzen
    financials = pd.read_csv(financial_path)
    
    # === KORREKTUR TEIL 1: Normalisierung und Umbenennung ===
    # Erstellt eine Map zur Korrektur der Spaltennamen
    col_map = {c.lower(): c for c in financials.columns}
    
    # Korrigiere Ticker-Index und Marktkapitalisierung auf Standard
    if 'ticker' in col_map:
        financials = financials.set_index(col_map['ticker'])
    else:
        # Fallback, falls die Ticker-Spalte fehlt – sehr unwahrscheinlich
        print("Fehler: 'ticker' Spalte nicht in financials.csv gefunden!")
        return pd.DataFrame()
        
    if 'marketcap' in col_map:
        financials = financials.rename(columns={col_map['marketcap']: 'marketCap'}, errors='ignore')
    # Wenn financials keine marketCap-Spalte hat oder viele Werte fehlen,
    # nutze die vorhandene `data/tickers_over_50B_full.csv` als zuverlässige Quelle
    tickers_path = base_path / 'data' / 'tickers_over_50B_full.csv'
    try:
        if tickers_path.exists():
            tickers_df = pd.read_csv(tickers_path, index_col='ticker')
            if 'marketCap' in tickers_df.columns:
                tickers_df['marketCap'] = pd.to_numeric(tickers_df['marketCap'], errors='coerce')
                # join the marketCap from tickers_df into financials (prefer financials existing values)
                financials = financials.join(tickers_df['marketCap'].rename('marketCap_tick'), how='left')
                if 'marketCap' not in financials.columns:
                    financials['marketCap'] = financials['marketCap_tick']
                else:
                    financials['marketCap'] = financials['marketCap'].fillna(financials['marketCap_tick'])
                # drop temporary column
                if 'marketCap_tick' in financials.columns:
                    financials = financials.drop(columns=['marketCap_tick'])
    except Exception:
        # safe fallback — falls Lesen/Join fehlschlägt weiter ohne crash
        pass
    # ========================================================
    
    scores = json.load(open(scores_path))
    superinvestor = scores.get("superinvestor_score", {})
    reddit = scores.get("reddit_score", {})
    x_sent = scores.get("x_score", {})
    
    ki_moat = {}
    if ki_path.exists():
        try:
            ki_moat_data = json.load(open(ki_path))
            ki_moat = ki_moat_data.get("ki_moat_score", ki_moat_data) # Akzeptiert Top-Level oder Sub-Key
        except:
            pass
            
    rows = []
    # Stellen Sie sicher, dass 'marketCap' nach der Umbenennung im Index ist
    required_cols = ['marketCap', 'trailingPE', 'forwardPE']

    for tk in universe:
        # === KORREKTUR TEIL 2: Robuster Check und Datenextraktion ===
        if tk in financials.index:
            # Stellt sicher, dass die wichtigsten Spalten vorhanden sind (z.B. nach Umbenennung)
            missing_data = False
            for col in required_cols:
                 if col not in financials.columns:
                     # Wenn eine wichtige Spalte fehlt, brechen wir den Prozess ab oder loggen den Fehler.
                     # Da die Spalte in compute_scores benötigt wird, muss sie hier existieren.
                     break 
            
            row = financials.loc[tk].to_dict()
            
            # WICHTIG: Überspringe Ticker nur, wenn beide P/E-Daten fehlen
            # (frühere Logik hat zu viele Ticker verworfen)
            if row.get('trailingPE') is None and row.get('forwardPE') is None:
                continue

            row.update({
                'sector': row.get('sector', 'Unknown'),
                'ticker': tk,
                'superinvestor_score': superinvestor.get(tk, 0.0),
                'reddit_score': reddit.get(tk, 0.0),
                'x_score': x_sent.get(tk, 0.5),
                'ki_moat_score': ki_moat.get(tk, 0.5),
                'pe_vs_history_score': row.get('pe_vs_history_score', 0.5),
                'insider_score': row.get('insider_score', 0.0),
                'analyst_score': row.get('analyst_score', 0.5),
            })
            rows.append(row)
        # =============================================================

    df = pd.DataFrame(rows)
    print(f"{len(df)} Aktien geladen – in unter 1 Sekunde!")
    
    # Füge fehlende Scores (die in compute_scores benötigt werden) als Platzhalter hinzu, falls df leer ist
    for col in ['marketCap', 'trailingPE', 'forwardPE']:
        if col not in df.columns and len(df) > 0:
            # Sollte nicht passieren, aber falls doch, hinzufügen
            df[col] = np.nan 

    return df

def compute_scores(df):
    
    # 🌟 KORREKTUR: Datenkonvertierung sicherstellen 🌟
    # Konvertiere die kritischen Spalten in Zahlen. 'coerce' wandelt nicht-numerische Werte (z.B. Text, Kommas, 'N/A') in NaN um.
    for col in ['marketCap', 'trailingPE', 'forwardPE', 'returnOnEquity', 'debtToEquity']:
        if col in df.columns:
            # Stelle sicher, dass die Spalten als float interpretiert werden
            df[col] = pd.to_numeric(df[col], errors='coerce') 
    # 🌟 ENDE KORREKTUR 🌟

    # Debug: marketCap statistics before filtering so we can detect scaling issues
    if 'marketCap' in df.columns and not df['marketCap'].isna().all():
        try:
            stats = df['marketCap'].describe().to_dict()
            print("marketCap stats (pre-filter):", {k: float(v) for k, v in stats.items()})
        except Exception:
            print("marketCap stats (pre-filter): could not compute stats")
        try:
            sample_vals = df['marketCap'].dropna().sort_values(ascending=False).head(5).tolist()
            print("marketCap top samples:", sample_vals)
        except Exception:
            pass

        max_mc = df['marketCap'].max(skipna=True)
        if pd.notna(max_mc) and max_mc < MIN_MARKET_CAP:
            # Heuristik: Werte sind vermutlich in Milliarden (Mrd) — skaliere auf USD
            print(f"Hinweis: marketCap max {max_mc} < MIN_MARKET_CAP ({MIN_MARKET_CAP}); skaliere *1e9 angenommen Mrd-Einheit.")
            df['marketCap'] = df['marketCap'] * 1e9

    # Filtern: Marktkapitalisierung muss vorhanden sein, und P/E-Daten müssen vorhanden 
    # sein.

    # Apply marketCap filter only if marketCap column exists and contains values
    if 'marketCap' in df.columns and not df['marketCap'].isna().all():
        df = df[df['marketCap'] >= MIN_MARKET_CAP].copy()
    else:
        print("Hinweis: marketCap fehlt oder ist komplett NaN — überspringe MarketCap-Filter.")

    # df = df[df['marketCap'] >= MIN_MARKET_CAP].dropna(subset=['trailingPE','forwardPE']).copy()
    
    # WARNUNG: Wenn nach dem Filtern keine Zeilen mehr da sind (die Ursache für den ZeroDivisionError)
    if df.empty:
        print("WARNUNG: Keine Aktien erfüllen die Kriterien (Marktkapitalisierung > 30 Mrd & P/E-Daten vorhanden).")
        # Gibt ein leeres DataFrame mit den erwarteten Spalten zurück, um compute_scores abzuschließen.
        return pd.DataFrame(columns=df.columns.tolist() + ['value_score', 'quality_score', 'community_score', 'final_score'])


    # Robust value score: compute percentile ranks, fill missing ranks with 0.5 (neutral)
    t_rank = df['trailingPE'].rank(ascending=True, pct=True)
    f_rank = df['forwardPE'].rank(ascending=True, pct=True)
    t_rank = t_rank.fillna(0.5)
    f_rank = f_rank.fillna(0.5)
    df['value_score'] = t_rank * 0.6 + f_rank * 0.4

    # Quality score: fill missing ROE/Debt with median to avoid excluding banks
    if 'returnOnEquity' in df.columns:
        roe_med = df['returnOnEquity'].median()
        df['returnOnEquity'] = df['returnOnEquity'].fillna(roe_med)
    if 'debtToEquity' in df.columns:
        de_med = df['debtToEquity'].median()
        df['debtToEquity'] = df['debtToEquity'].fillna(de_med)

    df['quality_score'] = df['returnOnEquity'].rank(ascending=False, pct=True) * 0.7 + (1 - df['debtToEquity'].rank(pct=True)) * 0.3
    df['community_score'] = df['superinvestor_score'] * 0.333 + df['reddit_score'] * 0.333 + df['x_score'] * 0.334

    df['final_score'] = (
        df['value_score'] * 0.30 +
        df['quality_score'] * 0.20 +
        df['community_score'] * 0.30 +
        df.get('pe_vs_history_score', 0.5) * 0.08 +
        df.get('insider_score', 0.0) * 0.05 +
        df.get('analyst_score', 0.5) * 0.05 +
        df['ki_moat_score'] * 0.07
    )

    return df.sort_values('final_score', ascending=False)

def main():
    
    print("Starte Value Stock Portfolio Picker (400+ Aktien, alle Signale)\n")
    universe = load_universe()
    print(f"Universe: {len(universe)} Aktien\n")

    df = gather(universe)
    portfolio = compute_scores(df).head(PORTFOLIO_SIZE)
    portfolio = construct_portfolio(portfolio, n=PORTFOLIO_SIZE)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = str(ROOT_DIR / f"examples/portfolio_{timestamp}.csv")
    
    # 🌟 KORREKTUR: Check, ob das Portfolio leer ist 🌟
    if portfolio.empty:
        print("\n--- 🎯 Endausgabe Portfolio Scores (Top 20) ---")
        print("Das Portfolio ist leer, da keine Aktien die Mindestkriterien erfüllt haben.")
        print("Bitte überprüfe die Kriterien in compute_scores oder deine Daten.")
        return # Beende die Funktion hier, wenn das Portfolio leer ist
    # 🌟 ENDE KORREKTUR 🌟


    portfolio['weight_%'] = (portfolio['weight'] * 100).round(1).astype(str) + '%'

    out = portfolio[['ticker','marketCap','sector','trailingPE','forwardPE','beta',
                     'returnOnEquity','debtToEquity','superinvestor_score',
                     'reddit_score','x_score','ki_moat_score',
                     'pe_vs_history_score','insider_score','analyst_score',
                     'final_score','weight_%']].copy()

    # Formatierungen für die Ausgabe
    out['marketCap'] = (out['marketCap']/1e9).round(1).astype(str) + ' Mrd'
    out['trailingPE'] = out['trailingPE'].round(1)
    out['forwardPE'] = out['forwardPE'].round(1)
    out['beta'] = out['beta'].round(2)
    out['returnOnEquity'] = (out['returnOnEquity']*100).round(1).astype(str) + '%'
    out['debtToEquity'] = out['debtToEquity'].round(1)
    for col in ['superinvestor_score','reddit_score','x_score','ki_moat_score',
                 'pe_vs_history_score','insider_score','analyst_score','final_score']:
        out[col] = out[col].round(3)

    print("\n--- 🎯 Endausgabe Portfolio Scores (Top 20) ---")
    print(out.to_string(index=False))
    
    # Berechnung der Portfolio-Kennzahlen
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

# Muss am Ende deiner run_picker_simulated_data.py existieren:
if __name__ == '__main__':
    main()