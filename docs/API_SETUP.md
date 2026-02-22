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

# LLM Committee APIs (optional, kostenlos)
# Für LLM-Komitee-Bewertung der Top 20 Aktien
HUGGINGFACE_API_KEY=your_hf_token_here
TOGETHER_API_KEY=your_together_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
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

### 5. LLM Committee APIs (Optional, kostenlos)

Das LLM-Komitee-System bewertet die Top 20 ausgewählten Aktien mit mehreren kostenlosen LLMs.

#### Hugging Face Inference API

1. Gehe zu https://huggingface.co/settings/tokens
2. Erstelle ein kostenloses Konto (falls noch nicht vorhanden)
3. Erstelle einen neuen Token (read access)
4. Füge ihn als `HUGGINGFACE_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, ~30 requests/min)

**Verwendete Modelle**: `mistralai/Mistral-7B-Instruct-v0.2`

#### Together AI

1. Gehe zu https://api.together.xyz/
2. Erstelle ein kostenloses Konto
3. Gehe zu API Keys
4. Kopiere deinen API Key
5. Füge ihn als `TOGETHER_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, ~100 requests/day)

**Verwendete Modelle**: `meta-llama/Llama-3-8b-chat-hf`

#### Google Gemini

1. Gehe zu https://makersuite.google.com/app/apikey
2. Erstelle ein kostenloses Konto (falls noch nicht vorhanden)
3. Erstelle einen neuen API Key
4. Füge ihn als `GEMINI_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, 60 requests/min)

**Verwendete Modelle**: `gemini-pro`

#### OpenAI (GPT-3.5-turbo)

1. Gehe zu https://platform.openai.com/api-keys
2. Erstelle ein kostenloses Konto (falls noch nicht vorhanden)
3. Erstelle einen neuen API Key
4. Füge ihn als `OPENAI_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, $5 Startguthaben, dann Pay-as-you-go)

**Verwendete Modelle**: `gpt-3.5-turbo`

**Persona**: Growth Optimist (Fokus auf Wachstum, Innovation, Zukunftspotential)

#### Anthropic Claude

1. Gehe zu https://console.anthropic.com/
2. Erstelle ein kostenloses Konto (falls noch nicht vorhanden)
3. Gehe zu API Keys
4. Erstelle einen neuen API Key
5. Füge ihn als `ANTHROPIC_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, $5 Startguthaben, dann Pay-as-you-go)

**Verwendete Modelle**: `claude-3-5-sonnet-20241022`

**Persona**: Risk Analyst (Detaillierte Risikoanalyse, Szenario-Planung)

#### Cohere

1. Gehe zu https://dashboard.cohere.com/api-keys
2. Erstelle ein kostenloses Konto (falls noch nicht vorhanden)
3. Erstelle einen neuen API Key
4. Füge ihn als `COHERE_API_KEY` in `.env` ein

**Kosten**: Kostenlos (free tier, ~100 requests/min)

**Verwendete Modelle**: `command`

**Persona**: Momentum Trader (Markttrends, Momentum, technische Signale)

**Hinweis**: Das LLM-Komitee funktioniert auch nur mit Groq (bereits vorhanden). Die anderen APIs sind optional und verbessern die Robustheit durch mehr LLMs. Mit allen 7 LLMs (Groq, Hugging Face, Together, Gemini, OpenAI, Anthropic, Cohere) erhältst du die beste Diversität und robusteste Scores.

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

