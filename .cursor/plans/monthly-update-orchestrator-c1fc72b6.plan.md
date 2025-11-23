<!-- c1fc72b6-9e26-496d-b8da-481eb6f5eedf ad96a85c-50e9-465f-b863-d7e92a7dc0aa -->
# Monatliches Update System & Streamlit App - Vollständige Implementierung

## Aktuelle Probleme

### 1. Monthly Update fehlt Finanzdaten

- ✅ Ticker-Liste wird geladen
- ✅ Sentiment-Daten werden geladen
- ❌ **Finanzdaten (Yahoo Finance) werden NICHT geladen und gespeichert**

### 2. Streamlit App ist nicht dynamisch

- ❌ Liest noch aus `examples/portfolio_*.csv` (alte, statische Daten)
- ❌ Berechnet Portfolio nicht dynamisch aus monatlichen Daten
- ❌ Anlagehorizont wird nicht verwendet (nur UI-Element, keine Logik)

### 3. Fehlende Integration

- ❌ Keine Funktion die monatliche Daten kombiniert und Portfolio berechnet
- ❌ App kann nicht auf `data/tickers/`, `data/financials/`, `data/scores/` zugreifen

## Lösung: 3 Hauptkomponenten

### Komponente 1: Monthly Update erweitern (Finanzdaten)

**Datei:** `scripts/monthly_update.py`

**Neue Funktion:** `update_financials(tickers)`

- Ruft `data_providers.fetch_fundamentals()` auf
- Speichert in `data/financials/YYYY-MM.csv` und `latest.csv`
- Format: DataFrame mit allen Fundamentaldaten (marketCap, trailingPE, forwardPE, beta, returnOnEquity, debtToEquity, sector, etc.)

**Ablauf:**

1. `update_universe()` → Ticker-Liste
2. `update_financials(tickers)` → **NEU** Finanzdaten
3. `update_sentiments(tickers)` → Sentiment-Daten
4. Alles speichern

### Komponente 2: Portfolio-Berechnung aus monatlichen Daten

**Neue Datei:** `scripts/portfolio_calculator.py`

**Funktion:** `calculate_portfolio_from_monthly_data(portfolio_size=20, horizon="5+ Jahre")`

- Lädt `data/tickers/latest.csv` → Ticker-Liste
- Lädt `data/financials/latest.csv` → Finanzdaten
- Lädt `data/scores/latest.json` → Sentiment-Daten
- Lädt `data/community_data/ai_moat.json` → KI-Moat (optional)
- Kombiniert alle Daten
- Ruft `compute_scores()` auf (aus run_picker_2.py)
- Ruft `construct_portfolio()` auf
- Gibt fertiges Portfolio zurück

**Anlagehorizont-Logik:**

- "1 Jahr": Mehr Gewicht auf Value (0.50), weniger auf Quality (0.15)
- "2 Jahre": Ausgewogen (0.40 Value, 0.20 Quality)
- "5+ Jahre": Mehr Gewicht auf Quality (0.30 Value, 0.30 Quality), weniger auf kurzfristige Signale

### Komponente 3: Streamlit App umbauen

**Datei:** `app/dashboard_final.py` (oder neue `app/main.py`)

**Neue Features:**

1. **Sidebar:**

   - Portfolio-Größe: 5, 10, 20 Aktien (Dropdown)
   - Anlagehorizont: 1 Jahr, 2 Jahre, 5+ Jahre (Dropdown)

2. **Hauptlogik:**

   - Ruft `portfolio_calculator.calculate_portfolio_from_monthly_data()` auf
   - Zeigt Portfolio dynamisch an
   - Keine CSV-Dateien mehr aus `examples/`

3. **UI:**

   - Portfolio-Tabelle
   - Sektor-Verteilung (Pie Chart)
   - Community vs. Final Score (Scatter)
   - Download-Button für CSV

## Implementierungsschritte

### Schritt 1: monthly_update.py erweitern

- Funktion `update_financials(tickers)` hinzufügen
- `data_providers.fetch_fundamentals()` aufrufen
- In `data/financials/YYYY-MM.csv` und `latest.csv` speichern
- In `main()` nach `update_universe()` aufrufen

### Schritt 2: portfolio_calculator.py erstellen

- Funktion `calculate_portfolio_from_monthly_data()` implementieren
- Alle monatlichen Daten laden und kombinieren
- `compute_scores()` und `construct_portfolio()` importieren und aufrufen
- Anlagehorizont-Logik implementieren (Gewichtungen anpassen)

### Schritt 3: dashboard_final.py umbauen

- Sidebar mit Portfolio-Größe und Anlagehorizont
- `portfolio_calculator` importieren
- Dynamische Portfolio-Berechnung statt CSV-Lesen
- UI anpassen

### Schritt 4: GitHub Actions erweitern

- `data/financials/*` auch committen
- Dependencies prüfen (yfinance, etc.)

## Dateien die geändert/erstellt werden

**Geändert:**

- `scripts/monthly_update.py` - Finanzdaten-Laden hinzufügen
- `app/dashboard_final.py` - Dynamische Berechnung
- `.github/workflows/monthly_update.yml` - data/financials/* committen

**Neu:**

- `scripts/portfolio_calculator.py` - Portfolio-Berechnung aus monatlichen Daten

## Anlagehorizont-Implementierung

```python
HORIZON_WEIGHTS = {
    "1 Jahr": {
        'value': 0.50, 'quality': 0.15, 'community': 0.20,
        'pe_vs_history': 0.10, 'insider': 0.03, 'analyst': 0.02
    },
    "2 Jahre": {
        'value': 0.40, 'quality': 0.20, 'community': 0.20,
        'pe_vs_history': 0.08, 'insider': 0.05, 'analyst': 0.05, 'ki_moat': 0.07
    },
    "5+ Jahre": {
        'value': 0.30, 'quality': 0.30, 'community': 0.20,
        'pe_vs_history': 0.05, 'insider': 0.03, 'analyst': 0.05, 'ki_moat': 0.12
    }
}
```

### To-dos

- [ ] monthly_update.py erweitern: update_universe() Funktion die generate_largecaps aufruft
- [ ] monthly_update.py: update_sentiments() Funktion die Reddit, X, Superinvestoren lädt
- [ ] dataroma.py: Scraper-Code aktivieren und universe-Parameter hinzufügen
- [ ] monthly_update.py: update_financials() Funktion hinzufügen die data_providers.fetch_fundamentals() aufruft und in data/financials/ speichert
- [ ] monthly_update.py: update_financials() in main() nach update_universe() aufrufen
- [ ] portfolio_calculator.py erstellen: calculate_portfolio_from_monthly_data() Funktion die alle monatlichen Daten lädt und kombiniert
- [ ] portfolio_calculator.py: compute_scores() und construct_portfolio() integrieren
- [ ] portfolio_calculator.py: Anlagehorizont-Logik implementieren (Gewichtungen anpassen)
- [ ] dashboard_final.py: Sidebar mit Portfolio-Größe (5,10,20) und Anlagehorizont (1,2,5+ Jahre) hinzufügen
- [ ] dashboard_final.py: Dynamische Portfolio-Berechnung mit portfolio_calculator statt CSV-Lesen
- [ ] GitHub Actions: data/financials/* auch committen