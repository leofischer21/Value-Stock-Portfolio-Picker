# Value Stock Portfolio Picker

Automatisches Screening großer, qualitativ hochwertiger Aktien mit starker Community-Unterstützung.  
Kombiniert klassische Value-/Quality-Kennzahlen mit realen Signalen von Superinvestoren, r/ValueInvesting und X (Twitter).

→ Ergebnis: ein diversifiziertes 20-Titel-Portfolio, das monatlich aktualisiert wird.

## Features (Stand 2025)

- Market-Cap-Filter ≥ 30 Mrd. USD  
- Value-Score (Trailing-PE + Forward-PE)  
- Quality-Score (ROE + Verschuldung)  
- Community-Score (gleich gewichtet)  
  - Superinvestoren (Dataroma-ähnliche Daten, manuell kuratiert)  
  - Reddit-Sentiment (r/ValueInvesting)  
  - X-Sentiment (Value-Twitter-Community)  
- Gewichtung: 50 % Fundamentaldaten + 50 % Community  
- Timestamp-basierte Historie (nie überschreiben)  
- Schickes Streamlit-Dashboard inkl. Sektorverteilung & Bubble-Charts  
- 100 % lokal lauffähig, voll reproduzierbar

## Quickstart

```bash
# 1. Repo klonen
git clone https://github.com/leofischer21/Value-Stock-Portfolio-Picker.git
cd Value-Stock-Portfolio-Picker

# 2. Conda-Umgebung erstellen & aktivieren
conda env create -f environment.yml
conda activate valuepicker

# 3. Portfolio berechnen (neue CSV mit Timestamp wird erstellt)
python scripts/run_picker.py

# 4. Dashboard starten
streamlit run app/dashboard_final.py
