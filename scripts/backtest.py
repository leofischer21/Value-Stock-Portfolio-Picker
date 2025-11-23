# backtest.py
import pandas as pd
import yfinance as yf
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# Root-Verzeichnis bestimmen (für Scripts in scripts/)
ROOT_DIR = Path(__file__).parent.parent

print("Starte monatlichen Backtest deines Value-Portfolios...\n")

# 1. Alle historischen Portfolios laden
files = sorted(glob.glob(str(ROOT_DIR / "examples/portfolio_*.csv")))
if len(files) < 2:
    print("Nicht genug historische Portfolios! Mindestens 2 Läufe nötig.")
    exit()

# 2. Für jedes Portfolio die monatliche Rendite berechnen
returns = []
dates = []
benchmark_returns = []

sp500 = yf.Ticker("^GSPC")

for i, file in enumerate(files):
    
    # --- Start des konsolidierten TRY-Blocks ---
    try:
        # A) Datums-Parsing korrigiert
        date_str = os.path.basename(file).replace("portfolio_", "").replace(".csv", "")
        # Nimmt den Teil vor dem ersten Unterstrich an, z.B. '2025-11-18' aus '2025-11-18_20-20'
        current_date = datetime.strptime(date_str.split("_")[0], "%Y-%m-%d")
        
        df = pd.read_csv(file)
        tickers = df['ticker'].tolist()
        weights = df['weight'].tolist()
        
        # Bestimme das Enddatum (Startdatum des nächsten Portfolios oder heute)
        if i == len(files) - 1:
            # Letztes Portfolio: Bis heute
            next_date = datetime.now()
        else:
            # Alle anderen: Bis zum Start des nächsten Portfolios
            next_file = files[i + 1]
            next_date_str = os.path.basename(next_file).split("_")[0]
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d")
        
        # Um den letzten Handelstag zu erfassen, müssen wir yf.download bis zum Tag NACH dem Enddatum aufrufen
        # Wir fügen 1 Tag hinzu, um den Start-Tag des nächsten Rebalance-Zeitpunktes NICHT zu inkludieren.
        download_end_date = next_date - timedelta(days=1)
        
        # B) Aktiendaten herunterladen
        # Wir nutzen nur 'Close' oder 'Adj Close'. 'Adj Close' ist besser für Renditen.
        prices = yf.download(tickers, start=current_date, end=next_date, progress=False)['Adj Close']
        
        # Entferne Ticker, für die keine Daten gefunden wurden (falls die Spalte fehlt)
        prices = prices.dropna(axis=1, how='all')
        
        # Prüfe, ob genügend Daten vorhanden sind
        if prices.shape[0] < 2:
             raise ValueError("Nicht genug Daten für die Renditenberechnung gefunden.")

        # C) Portfolio Rendite berechnen
        # Die monatliche Rendite ist die prozentuale Veränderung vom ersten zum letzten Tag.
        monthly_returns = prices.iloc[-1] / prices.iloc[0] - 1
        
        # Multipliziere Gewichte mit Renditen. Behandle NaN im monthly_returns (z.B. bei nicht gehandelten Aktien)
        portfolio_return = sum(w * r for w, r in zip(weights, monthly_returns) if not pd.isna(r))
        
        # D) Benchmark Rendite
        sp_data = sp500.history(start=current_date, end=next_date)['Close']
        sp_return = sp_data.iloc[-1] / sp_data.iloc[0] - 1
        
        # Füllen der Listen bei Erfolg
        dates.append(current_date.strftime("%Y-%m"))
        returns.append(portfolio_return)
        benchmark_returns.append(sp_return)
        
    except Exception as e:
        print(f"Fehler bei {file}: {e}")
        # WICHTIG: Stelle sicher, dass ALLE Listen mit Platzhaltern gefüllt werden,
        # damit sie am Ende die gleiche Länge haben. Wir verwenden 0 als Rendite.
        dates.append(current_date.strftime("%Y-%m") if 'current_date' in locals() else 'Fehlerdatum')
        returns.append(0)
        benchmark_returns.append(0)
        
# 3. Ergebnisse (Der Rest des Codes ist in Ordnung)
df_back = pd.DataFrame({
    "Monat": dates,
    "Portfolio": returns,
    "S&P 500": benchmark_returns
})
# ... (der Rest des Codes bleibt gleich)

if len(df_back) == 0:
    print("Backtest konnte keine gültigen Daten sammeln.")
    exit()

df_back["Portfolio_cum"] = (1 + df_back["Portfolio"]).cumprod()
df_back["SP500_cum"] = (1 + df_back["S&P 500"]).cumprod()

total_return = df_back["Portfolio_cum"].iloc[-1] - 1
cagr = (df_back["Portfolio_cum"].iloc[-1]) ** (12 / len(df_back)) - 1
max_dd = (df_back["Portfolio_cum"] / df_back["Portfolio_cum"].cummax() - 1).min()

print(f"\n--- Backtest Ergebnisse ---")
print(f"Backtest-Zeitraum: {df_back['Monat'].iloc[0]} bis {df_back['Monat'].iloc[-1]}")
print(f"Anzahl Monate: {len(df_back)}")
print(f"Gesamtrendite: {total_return:.1%}")
print(f"CAGR: {cagr:.1%}")
print(f"Max Drawdown: {max_dd:.1%}")
print(f"Outperformance vs. S&P 500: {(total_return - (df_back['SP500_cum'].iloc[-1]-1)):.1%}")

# Plot speichern
plt.figure(figsize=(12,6))
plt.plot(df_back["Monat"], df_back["Portfolio_cum"], label="Dein Value-Portfolio", linewidth=2)
plt.plot(df_back["Monat"], df_back["SP500_cum"], label="S&P 500", linewidth= 2, alpha=0.8)
plt.title("Backtest: Dein Portfolio vs. S&P 500")
plt.ylabel("Wachstum (1 = Startkapital)")
plt.legend()
plt.grid(True)
# Zeigt nur eine Untermenge der Datumsangaben an, um Lesbarkeit zu verbessern
step = max(1, len(df_back) // 15) 
plt.xticks(df_back["Monat"][::step], rotation=45) 
plt.tight_layout()
plt.savefig(str(ROOT_DIR / "examples/backtest_curve.png"), dpi=150)
print("\nKurve gespeichert → examples/backtest_curve.png")

# Ergebnis speichern
df_back.to_csv(str(ROOT_DIR / "examples/backtest_results.csv"), index=False)
print("Backtest-Daten gespeichert → examples/backtest_results.csv")



""" # backtest.py
import pandas as pd
import yfinance as yf
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt

print("Starte monatlichen Backtest deines Value-Portfolios...\n")

# 1. Alle historischen Portfolios laden
files = sorted(glob.glob("examples/portfolio_*.csv"))
if len(files) < 2:
    print("Nicht genug historische Portfolios! Mindestens 2 Läufe nötig.")
    exit()

# 2. Für jedes Portfolio die monatliche Rendite berechnen
returns = []
dates = []
benchmark_returns = []

sp500 = yf.Ticker("^GSPC")

for i, file in enumerate(files):
    date_str = os.path.basename(file).replace("portfolio_", "").replace(".csv", "")
    try:
        current_date = datetime.strptime(date_str.split("_")[0], "%Y-%m-%d")
    except:
        continue

    df = pd.read_csv(file)
    tickers = df['ticker'].tolist()
    weights = df['weight'].tolist()

    # Rendite dieses Monats
    try:
        prices = yf.download(tickers, start=current_date, end=current_date.replace(month=current_date.month % 12 + 1) if current_date.month < 12 else current_date.replace(year=current_date.year + 1, month=1), progress=False)['Adj Close']
        if i == len(files) - 1:
            end_date = datetime.now()
        else:
            next_file = files[i+1] if i+1 < len(files) else None
            next_date = datetime.strptime(os.path.basename(next_file).split("_")[0], "%Y-%m-%d") if next_file else datetime.now()
            prices = yf.download(tickers, start=current_date, end=next_date, progress=False)['Adj Close']

        monthly_returns = prices.pct_change().iloc[-1]  # letzte Zeile = Rendite bis zum nächsten Rebalance
        portfolio_return = sum(w * r for w, r in zip(weights, monthly_returns) if not pd.isna(r))
        returns.append(portfolio_return)
        dates.append(current_date.strftime("%Y-%m"))

        # S&P 500 Vergleich
        sp_return = sp500.history(start=current_date, end=next_date if 'next_date' in locals() else datetime.now())['Close'].pct_change().iloc[-1]
        benchmark_returns.append(sp_return)

    except Exception as e:
        print(f"Fehler bei {file}: {e}")
        returns.append(0)
        benchmark_returns.append(0)

# 3. Ergebnisse
df_back = pd.DataFrame({
    "Monat": dates,
    "Portfolio": returns,
    "S&P 500": benchmark_returns
})
df_back["Portfolio_cum"] = (1 + df_back["Portfolio"]).cumprod()
df_back["SP500_cum"] = (1 + df_back["S&P 500"]).cumprod()

total_return = df_back["Portfolio_cum"].iloc[-1] - 1
cagr = (df_back["Portfolio_cum"].iloc[-1]) ** (12 / len(df_back)) - 1
max_dd = (df_back["Portfolio_cum"] / df_back["Portfolio_cum"].cummax() - 1).min()

print(f"Backtest-Zeitraum: {df_back['Monat'].iloc[0]} bis {df_back['Monat'].iloc[-1] or 'heute'}")
print(f"Anzahl Monate: {len(df_back)}")
print(f"Gesamtrendite: {total_return:.1%}")
print(f"CAGR: {cagr:.1%}")
print(f"Max Drawdown: {max_dd:.1%}")
print(f"Outperformance vs. S&P 500: {(total_return - (df_back['SP500_cum'].iloc[-1]-1)):.1%}")

# Plot speichern
plt.figure(figsize=(12,6))
plt.plot(df_back["Monat"], df_back["Portfolio_cum"], label="Dein Value-Portfolio", linewidth=2)
plt.plot(df_back["Monat"], df_back["SP500_cum"], label="S&P 500", linewidth= 2, alpha=0.8)
plt.title("Backtest: Dein Portfolio vs. S&P 500")
plt.ylabel("Wachstum (1 = Startkapital)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("examples/backtest_curve.png", dpi=150)
print("\nKurve gespeichert → examples/backtest_curve.png")

# Ergebnis speichern
df_back.to_csv("examples/backtest_results.csv", index=False)
print("Backtest-Daten gespeichert → examples/backtest_results.csv") """