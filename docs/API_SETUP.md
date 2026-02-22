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

# YouTube Data API (für YouTube-Video-Sentiment)
YOUTUBE_API_KEY=your_youtube_api_key_here

# YouTube Channel IDs (optional, komma-separiert)
# Falls nicht gesetzt, wird Standard-Liste verwendet (z.B. Joseph Carlson)
YOUTUBE_CHANNEL_IDS=UCwDlyuX3Fkg5WNBufLnH6dw,UCxxxxx,UCyyyyy
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

### 4. YouTube Data API Key

1. Gehe zu https://console.cloud.google.com/apis/library/youtube.googleapis.com
2. Erstelle ein Projekt (falls noch nicht vorhanden)
3. Aktiviere die "YouTube Data API v3"
4. Gehe zu "Credentials" > "Create Credentials" > "API Key"
5. Kopiere den API Key
6. Füge ihn als `YOUTUBE_API_KEY` in `.env` ein

**Kosten**: Free (10.000 Queries/Tag) - ausreichend für monatliche Updates

**Optional**: Channel-IDs konfigurieren
- Standard: Joseph Carlson (`UCwDlyuX3Fkg5WNBufLnH6dw`)
- Weitere Kanäle: Füge `YOUTUBE_CHANNEL_IDS` in `.env` hinzu (komma-separiert)
- Oder bearbeite `scripts/youtube_sentiment.py` direkt

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
- **YouTube**: Neue Funktion - Fetcht Videos von konfigurierten Kanälen, transkribiert (Whisper) und analysiert Sentiment (Groq)
- **Sentiment-Analyse**: Verwendet Groq für alle Posts/Tweets/Videos (mit spezieller Gewichtung für "undervalued"/"overvalued")

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
- **"quality" / "moat"**: +0.1 zum Sentiment-Score (gilt für Reddit, X und YouTube)

## YouTube-Sentiment

- **Transkription**: Verwendet `yt-dlp` + OpenAI Whisper (lokal) für Video-Transkription
- **Analyse**: Groq Chat API extrahiert Ticker und Sentiment aus Transkripten
- **Zeitraum**: Nur Videos der letzten 6 Monate (180 Tage) werden berücksichtigt
- **Deduplizierung**: Verarbeitete Videos werden in `data/youtube/processed_videos.json` gespeichert
- **Kosten**: ~1-6€/Monat (abhängig von Video-Anzahl, ~0.10-0.50€ pro Video für Transkription)

