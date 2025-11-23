# API Setup für monthly_update_api.py

Diese Datei dokumentiert die benötigten API-Keys für die API-basierte Version von `monthly_update.py`.

## Neue API-Keys in .env

Füge folgende Variablen zu deiner `.env` Datei hinzu:

```env
# Apify API (für Reddit UND Twitter)
APIFY_API_TOKEN=your_apify_token_here

# Groq API (bereits vorhanden, wird für Sentiment-Analyse verwendet)
GROQ_API_KEY=your_groq_api_key_here
AI_MODEL=llama-3.3-70b-versatile
```

## API-Setup Anleitungen

### 1. Apify API Token

1. Gehe zu https://apify.com
2. Erstelle ein kostenloses Konto
3. Gehe zu Settings > Integrations > API tokens
4. Kopiere deinen API Token
5. Füge ihn als `APIFY_API_TOKEN` in `.env` ein

**Kosten**: ~3-5 €/Monat für Reddit (~10.000 Posts) + ~2-5 €/Monat für Twitter (~5.000 Tweets) = **~5-10 €/Monat gesamt** (Pay-per-Result)

**Verwendete Actors**:
- **Reddit**: `comchat/reddit-api-scraper`
- **Twitter**: `kaitoeasyapi/tweet-scraper`

### 3. Groq API Key

Bereits vorhanden - wird für Sentiment-Analyse verwendet.

## Fallback-Mechanismus

Falls eine API fehlschlägt oder kein API-Key gesetzt ist, greift das System automatisch auf die Legacy-Methoden zurück:
- `reddit.py` (community_signals.json + simulierte Daten)
- `twitter.py` (community_signals.json + simulierte Daten)
- `dataroma.py` (Scraping von dataroma.com)

## Verwendung

```bash
# Normale Ausführung (überspringt vorhandene Daten)
python scripts/monthly_update_api.py

# Force-Modus (überschreibt alle Daten)
python scripts/monthly_update_api.py --force
```

## Unterschiede zur Original-Version

- **Reddit**: Verwendet Apify API (`comchat/reddit-api-scraper`) statt community_signals.json
- **X/Twitter**: Verwendet Apify API (`kaitoeasyapi/tweet-scraper`) statt community_signals.json
- **Dataroma**: Verwendet SEC EDGAR API statt Scraping (kostenlos, nur User-Agent erforderlich)
- **Sentiment-Analyse**: Verwendet Groq für alle Posts/Tweets (mit spezieller Gewichtung für "undervalued"/"overvalued")

## Subreddit-Gewichtungen

- `r/ValueInvesting`: **3.0** (am stärksten)
- `r/stocks`: **1.5**
- `r/investing`: **1.5**
- `r/stockmarket`: **1.0**
- `r/wallstreetbets`: **0.3** (niedrig)

## X/Twitter-Account-Gewichtungen

Folgende Accounts werden stärker gewichtet (Gewicht 3.0):
- compounding quality
- oguz o. (x capitalist)
- shay boloor
- patient investor
- fiscal.ai
- data driven investing
- qualtrim
- bourbon capital
- mindset for money
- dimitry nakhla
- quality equities
- sam badawi
- antonio linares

## Keyword-Gewichtungen

- **"undervalued"**: +0.3 zum Sentiment-Score (sehr stark positiv)
- **"overvalued"**: -0.3 vom Sentiment-Score (sehr stark negativ)

