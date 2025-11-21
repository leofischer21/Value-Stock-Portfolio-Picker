# KI Scores und Performance-Vorhersagen

## Übersicht

Das System nutzt KI (LLM-APIs) zur Bewertung von:
- **Moat Score**: Wettbewerbsvorteil einer Aktie (0.0-1.0)
- **Quality Score**: Qualität des Unternehmens (0.0-1.0)
- **Predicted Performance**: CAGR-Vorhersagen für 1, 2, 5, 10 Jahre (in Prozent)

## API-Konfiguration

### Option 1: Umgebungsvariablen (empfohlen)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
$env:AI_MODEL="gpt-4"

# Oder für Anthropic
$env:ANTHROPIC_API_KEY="your-api-key-here"
$env:AI_MODEL="claude-3-opus-20240229"
```

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export AI_MODEL="gpt-4"
```

### Option 2: .env Datei (optional)

Erstelle eine `.env` Datei im Root-Verzeichnis:
```
OPENAI_API_KEY=your-api-key-here
AI_MODEL=gpt-4
```

## Fallback-Strategie

Wenn keine API-Keys konfiguriert sind oder die API nicht verfügbar ist, nutzt das System **heuristische Berechnungen** basierend auf:
- Finanzdaten (P/E, Margins, ROE, ROIC, Debt/Equity)
- Bestehende Scores (Community, Quality)
- Markt-Durchschnitte

Die heuristischen Fallbacks sind voll funktionsfähig und liefern realistische Schätzungen.

## Monatliche Berechnung

KI-Scores werden **nur einmal pro Monat** berechnet (am Anfang des Monats) und in `data/ai_scores/` gespeichert:
- `YYYY-MM.json`: Monatliche Datei
- `latest.json`: Aktuelle Daten

Dies reduziert API-Kosten und verbessert die Performance.

## Integration

- **Portfolio Calculator**: Lädt KI-Scores automatisch aus gespeicherten Dateien
- **Dashboard**: Zeigt Predicted Performance je nach ausgewähltem Anlagehorizont
- **Final Score**: KI-Scores fließen in die Final Score Berechnung ein

## Kosten-Hinweis

Für 400+ Ticker können API-Calls teuer sein. Das System nutzt:
- **Caching**: 30 Tage Cache für API-Antworten
- **Monatliche Berechnung**: Nur 1x pro Monat
- **Fallbacks**: Heuristische Berechnungen wenn API nicht verfügbar

## Installation

Für OpenAI:
```bash
pip install openai
```

Für Anthropic:
```bash
pip install anthropic
```

## Verwendung

Die KI-Scores werden automatisch in `monthly_update.py` berechnet. Keine manuelle Konfiguration nötig!

