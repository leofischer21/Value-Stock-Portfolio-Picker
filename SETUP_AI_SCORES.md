# KI Scores Setup & Verwendung

## Aktueller Status

✅ **System funktioniert OHNE API-Keys!**
- Heuristische Fallbacks sind voll funktionsfähig
- Alle Scores werden berechnet (Moat, Quality, Performance)
- System ist sofort einsatzbereit

## API-Keys: Optional aber Empfohlen

### Warum API-Keys?

**Ohne API-Keys:**
- System nutzt **heuristische Berechnungen** basierend auf Finanzdaten
- Funktioniert vollständig, aber weniger "intelligent"
- Scores basieren auf mathematischen Formeln

**Mit API-Keys:**
- System nutzt **LLM (GPT-4/Claude)** für intelligente Bewertungen
- Berücksichtigt Kontext, Branche, Marktlage
- Genauere und nuanciertere Bewertungen

### API-Keys konfigurieren

#### Option 1: Umgebungsvariablen (Empfohlen)

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:AI_MODEL="gpt-4"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-your-key-here
set AI_MODEL=gpt-4
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
export AI_MODEL="gpt-4"
```

**Oder für Anthropic Claude:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
$env:AI_MODEL="claude-3-opus-20240229"
```

#### Option 2: Permanente Konfiguration

Erstelle eine `.env` Datei im Root-Verzeichnis:
```
OPENAI_API_KEY=sk-your-key-here
AI_MODEL=gpt-4
```

**Hinweis:** `.env` Datei sollte in `.gitignore` sein!

### API-Keys erhalten

1. **OpenAI:**
   - Gehe zu https://platform.openai.com/api-keys
   - Erstelle neuen API-Key
   - Kosten: ~$0.01-0.03 pro 1000 Ticker (bei monatlicher Berechnung)

2. **Anthropic:**
   - Gehe zu https://console.anthropic.com/
   - Erstelle neuen API-Key
   - Kosten: Ähnlich wie OpenAI

## Wie funktioniert es?

### 1. Monatliche Berechnung

Wenn du `python scripts/monthly_update.py` ausführst:

1. **Ohne API-Keys:**
   - System nutzt heuristische Funktionen
   - Berechnet Scores basierend auf Finanzdaten
   - Sehr schnell, keine Kosten

2. **Mit API-Keys:**
   - System versucht LLM-API aufzurufen
   - Bei Erfolg: Intelligente Bewertungen
   - Bei Fehler: Automatischer Fallback zu Heuristik
   - Ergebnisse werden 30 Tage gecacht

### 2. Was wird berechnet?

Für jeden Ticker:
- **Moat Score** (0.0-1.0): Wettbewerbsvorteil
- **Quality Score** (0.0-1.0): Unternehmensqualität
- **Predicted Performance:**
  - CAGR 1 Jahr (z.B. 12.5%)
  - CAGR 2 Jahre (z.B. 15.3%)
  - CAGR 5 Jahre (z.B. 18.2%)
  - CAGR 10 Jahre (z.B. 16.8%)

### 3. Speicherung

Ergebnisse werden gespeichert in:
- `data/ai_scores/YYYY-MM.json` (monatliche Datei)
- `data/ai_scores/latest.json` (aktuelle Daten)

Diese Dateien werden dann vom Portfolio Calculator geladen.

## Kosten-Schätzung

**Ohne API-Keys:** Kostenlos ✅

**Mit API-Keys (OpenAI GPT-4):**
- ~400 Ticker pro Monat
- ~$0.50-1.50 pro Monat (je nach Modell)
- Caching reduziert Kosten bei wiederholten Berechnungen

**Tipp:** Starte ohne API-Keys, teste das System, und füge API-Keys später hinzu wenn gewünscht.

## Installation von API-Paketen

**Für OpenAI:**
```bash
pip install openai
```

**Für Anthropic:**
```bash
pip install anthropic
```

**Hinweis:** Diese Pakete sind **optional** - System funktioniert auch ohne!

## Testen

Führe aus:
```bash
python test_ai_scores.py
```

Dies testet:
- ✅ API-Key Konfiguration
- ✅ Heuristische Fallbacks
- ✅ Portfolio Calculator Integration
- ✅ Monatliche Update-Funktion

## Nächste Schritte

1. **Sofort starten (ohne API-Keys):**
   ```bash
   python scripts/monthly_update.py
   streamlit run app/dashboard_final.py
   ```

2. **Später API-Keys hinzufügen (optional):**
   - Setze Umgebungsvariablen
   - Führe `monthly_update.py` erneut aus
   - System nutzt automatisch LLM wenn verfügbar

## Troubleshooting

**Problem:** "openai package not installed"
- **Lösung:** `pip install openai` (optional, nur wenn API-Keys verwendet werden)

**Problem:** API-Fehler
- **Lösung:** System nutzt automatisch Fallbacks - kein Problem!

**Problem:** Keine AI Scores im Dashboard
- **Lösung:** Führe `python scripts/monthly_update.py` aus um Scores zu generieren

