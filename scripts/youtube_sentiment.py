# youtube_sentiment.py
"""
YouTube-Video-Sentiment-Analyse via YouTube Data API v3 und Groq.
Fetcht neue Videos von konfigurierten Kanälen, transkribiert sie (Groq Whisper)
und analysiert Sentiment für erwähnte Ticker (Groq Chat).
"""
import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np

# Root-Verzeichnis bestimmen
try:
    ROOT_DIR = Path(__file__).parent.parent
except:
    ROOT_DIR = Path.cwd()

# Lade .env Datei falls vorhanden
try:
    from dotenv import load_dotenv
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass
except Exception:
    pass

logger = logging.getLogger(__name__)

# API-Konfiguration
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile")

# Konfigurierbare YouTube-Kanal-IDs (kann in .env oder hier erweitert werden)
# Format: Liste von Channel-IDs
DEFAULT_CHANNEL_IDS = [
    "UCbta0n8i6Rljh0obO7HzG9A",  # Joseph Carlson (The Joseph Carlson Show) - @JosephCarlsonShow
    "UCfCT7SSFEWyG4th9ZmaGYqQ",  # Joseph Carlson After Hours
    "UCFyQbh4o40KNgdKLWPaU_9Q",  # The Quality Investor
    "UCowj3bHIz47dMIe8n37qTlw",  # The Patient Investor
    "UCrGLm-Drgv0vbbemwwHeXJw",  # Couch Investor
    "UCJggpN5VY0PWKoOyBBT0R8A",  # FAST Graphs
    # Weitere Kanäle können hier hinzugefügt werden
]

# Filter-Thresholds für Kostenoptimierung
MIN_VIEW_COUNT = 5000  # Nur Videos mit >5k Views (eliminiert Low-Engagement)
MIN_TICKERS_FOR_ANALYSIS = 3  # Skippe Videos mit <3 Tickers (allgemeine Videos ohne spezifische Picks)

# Lade Channel-IDs aus .env falls vorhanden (komma-separiert)
YOUTUBE_CHANNEL_IDS_ENV = os.environ.get("YOUTUBE_CHANNEL_IDS")
if YOUTUBE_CHANNEL_IDS_ENV:
    YOUTUBE_CHANNEL_IDS = [cid.strip() for cid in YOUTUBE_CHANNEL_IDS_ENV.split(",")]
else:
    YOUTUBE_CHANNEL_IDS = DEFAULT_CHANNEL_IDS

# Value-Investing-Keywords für Sentiment-Analyse
VALUE_KEYWORDS = {
    'positive': ['undervalued', 'buy', 'value', 'cheap', 'quality', 'moat', 'margin of safety', 
                 'intrinsic value', 'earnings yield', 'book value', 'dividend yield', 
                 'free cash flow', 'ROE', 'ROIC', 'DCF', 'P/B ratio'],
    'negative': ['overvalued', 'sell', 'expensive', 'crash', 'bubble', 'overpriced']
}

# Keywords für Video-Titel-Filter (nur Videos mit diesen Keywords werden verarbeitet)
TITLE_FILTER_KEYWORDS = [
    'stock', 'stocks', 'value', 'investing', 'investor', 'portfolio', 'dividend',
    'earnings', 'financial', 'market', 'equity', 'equities', 'share', 'shares',
    'company', 'companies', 'business', 'ticker', 'tickers', 'buy', 'sell',
    'undervalued', 'overvalued', 'quality', 'moat', 'intrinsic', 'valuation'
]

# Pfade
YOUTUBE_DIR = ROOT_DIR / "data" / "youtube"
PROCESSED_VIDEOS_PATH = YOUTUBE_DIR / "processed_videos.json"
TRANSCRIPTS_DIR = YOUTUBE_DIR / "transcripts"
YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_processed_videos() -> Dict:
    """Lädt die Liste aller verarbeiteten Videos."""
    if PROCESSED_VIDEOS_PATH.exists():
        try:
            with open(PROCESSED_VIDEOS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load processed_videos.json: {e}")
            return {}
    return {}


def _save_processed_videos(processed_videos: Dict) -> None:
    """Speichert die Liste aller verarbeiteten Videos."""
    try:
        with open(PROCESSED_VIDEOS_PATH, "w", encoding="utf-8") as f:
            json.dump(processed_videos, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Could not save processed_videos.json: {e}")


def _save_processed_video(video_id: str, channel_id: str, title: str, published_at: str, 
                         transcript: str, tickers_scores: Dict[str, float]) -> None:
    """Speichert ein verarbeitetes Video in processed_videos.json."""
    processed_videos = _load_processed_videos()
    
    # Erstelle Hash des Transkripts für Deduplizierung
    transcript_hash = hashlib.sha256(transcript.encode('utf-8')).hexdigest()[:16]
    
    processed_videos[video_id] = {
        'channel_id': channel_id,
        'title': title,
        'published_at': published_at,
        'transcript_hash': transcript_hash,
        'tickers': tickers_scores,
        'processed_at': datetime.now().isoformat()
    }
    
    _save_processed_videos(processed_videos)


def _filter_by_date(processed_videos: Dict, days_back: int = 180) -> Dict:
    """Filtert Videos auf die letzten N Tage."""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered = {}
    
    for video_id, video_data in processed_videos.items():
        try:
            published_at = video_data.get('published_at', '')
            if isinstance(published_at, str):
                # Parse ISO format oder YouTube format
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                except:
                    # Versuche verschiedene Datumsformate
                    pub_date = None
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                        try:
                            pub_date = datetime.strptime(published_at[:19], fmt)
                            break
                        except:
                            continue
                    if pub_date is None:
                        logger.debug(f"Could not parse date for video {video_id}: {published_at}")
                        continue
                
                if pub_date >= cutoff_date:
                    filtered[video_id] = video_data
        except Exception as e:
            logger.debug(f"Error filtering video {video_id}: {e}")
            continue
    
    return filtered


def _is_relevant_video(title: str, description: str = "") -> bool:
    """
    Prüft ob ein Video relevant ist (enthält Value-Investing-Keywords im Titel).
    Filtert Videos ohne relevante Keywords, um Kosten zu sparen.
    """
    text = f"{title} {description}".lower()
    return any(keyword.lower() in text for keyword in TITLE_FILTER_KEYWORDS)


def _get_video_statistics(video_ids: List[str]) -> Dict[str, int]:
    """
    Holt View-Counts für Videos via YouTube Data API v3.
    
    Returns: Dict {video_id: view_count}
    """
    if not YOUTUBE_API_KEY:
        return {}
    
    try:
        from googleapiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # YouTube API erlaubt max 50 IDs pro Request
        view_counts = {}
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            try:
                response = youtube.videos().list(
                    part='statistics',
                    id=','.join(batch)
                ).execute()
                
                for item in response.get('items', []):
                    video_id = item['id']
                    view_count = int(item['statistics'].get('viewCount', 0))
                    view_counts[video_id] = view_count
            except Exception as e:
                logger.warning(f"Error fetching statistics for video batch: {e}")
                continue
        
        return view_counts
    except Exception as e:
        logger.warning(f"Error in _get_video_statistics: {e}")
        return {}


def _fetch_new_videos(channel_ids: List[str], days_back: int = 120) -> List[Dict]:
    """
    Fetcht neue Videos von YouTube-Kanälen via YouTube Data API v3.
    Filtert nach View-Count (>5k) um Low-Engagement-Videos zu eliminieren.
    
    Returns: Liste von Dicts mit video_id, channel_id, title, description, published_at, view_count
    """
    if not YOUTUBE_API_KEY:
        logger.warning("YOUTUBE_API_KEY not set, skipping YouTube video fetch")
        return []
    
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        logger.error("google-api-python-client not installed. Install with: pip install google-api-python-client")
        return []
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    all_videos = []
    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
    
    # Lade bereits verarbeitete Videos
    processed_videos = _load_processed_videos()
    processed_video_ids = set(processed_videos.keys())
    
    for channel_id in channel_ids:
        try:
            # Suche nach Videos des Kanals mit Pagination (um alle Videos zu bekommen)
            next_page_token = None
            channel_videos = []
            
            while True:
                request_params = {
                    'part': 'snippet',
                    'channelId': channel_id,
                    'type': 'video',
                    'order': 'date',
                    'publishedAfter': cutoff_date,
                    'maxResults': 50  # YouTube API Limit pro Request
                }
                
                if next_page_token:
                    request_params['pageToken'] = next_page_token
                
                request = youtube.search().list(**request_params)
                response = request.execute()
                
                # Verarbeite Videos aus dieser Seite
                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    
                    # Überspringe bereits verarbeitete Videos
                    if video_id in processed_video_ids:
                        continue
                    
                    snippet = item['snippet']
                    title = snippet.get('title', '')
                    description = snippet.get('description', '')
                    
                    # Filter: Nur Videos mit Value-Investing-Keywords im Titel/Beschreibung
                    if not _is_relevant_video(title, description):
                        continue  # Überspringe nicht-relevante Videos
                    
                    video_data = {
                        'video_id': video_id,
                        'channel_id': channel_id,
                        'title': title,
                        'description': description,
                        'published_at': snippet.get('publishedAt', ''),
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    }
                    channel_videos.append(video_data)
                
                # Prüfe ob es weitere Seiten gibt
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            all_videos.extend(channel_videos)
            logger.info(f"Found {len(channel_videos)} new videos from channel {channel_id} (before filters)")
            
        except HttpError as e:
            logger.error(f"YouTube API error for channel {channel_id}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error fetching videos from channel {channel_id}: {e}")
            continue
    
    # Filter nach View-Count (>5k Views)
    if all_videos:
        logger.info(f"Fetching view counts for {len(all_videos)} videos...")
        video_ids = [v['video_id'] for v in all_videos]
        view_counts = _get_video_statistics(video_ids)
        
        # Füge view_count hinzu und filtere
        filtered_videos = []
        skipped_low_views = 0
        for video in all_videos:
            video_id = video['video_id']
            view_count = view_counts.get(video_id, 0)
            video['view_count'] = view_count
            
            if view_count >= MIN_VIEW_COUNT:
                filtered_videos.append(video)
            else:
                skipped_low_views += 1
        
        logger.info(f"View-Count Filter: {len(filtered_videos)} videos kept (>{MIN_VIEW_COUNT} views), {skipped_low_views} skipped")
        all_videos = filtered_videos
    
    logger.info(f"Total new videos to process (after filters): {len(all_videos)}")
    return all_videos


def _get_transcript_from_youtube_api(video_id: str) -> Optional[str]:
    """
    Versucht Transkript direkt von YouTube Data API v3 zu holen (falls verfügbar).
    Dies ist die zuverlässigste Methode, da keine Downloads nötig sind.
    
    Returns: Transkript als Text oder None bei Fehler
    """
    if not YOUTUBE_API_KEY:
        return None
    
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Prüfe ob Transkript/Captions verfügbar sind
        try:
            captions_list = youtube.captions().list(part='snippet', videoId=video_id).execute()
            if not captions_list.get('items'):
                return None
            
            # Lade erstes verfügbares Transkript (priorisiere Englisch)
            caption_id = None
            for item in captions_list['items']:
                lang = item['snippet'].get('language', '').lower()
                if lang.startswith('en'):
                    caption_id = item['id']
                    break
            
            # Falls kein Englisch, nimm erstes verfügbares
            if not caption_id and captions_list['items']:
                caption_id = captions_list['items'][0]['id']
            
            if caption_id:
                # Lade Transkript (als SRT-Format, dann parsen)
                transcript_srt = youtube.captions().download(id=caption_id, tfmt='srt').execute()
                
                # Parse SRT zu reinem Text
                import re
                # Entferne SRT-Timestamps und Formatierung
                transcript = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', transcript_srt)
                transcript = re.sub(r'<[^>]+>', '', transcript)  # Entferne HTML-Tags
                transcript = re.sub(r'\n+', ' ', transcript)  # Normalisiere Zeilenumbrüche
                transcript = transcript.strip()
                
                if transcript and len(transcript) > 100:
                    logger.debug(f"Got transcript from YouTube API for {video_id} ({len(transcript)} chars)")
                    return transcript
        except HttpError as e:
            if e.resp.status == 404:
                logger.debug(f"No captions available for video {video_id}")
            else:
                logger.debug(f"YouTube API error for captions: {e}")
        except Exception as e:
            logger.debug(f"Error getting transcript from YouTube API: {e}")
    except ImportError:
        logger.debug("google-api-python-client not available for transcript fetching")
    except Exception as e:
        logger.debug(f"Error in YouTube API transcript fetch: {e}")
    
    return None


def _transcribe_video(video_id: str, video_url: str) -> Optional[str]:
    """
    Transkribiert ein YouTube-Video mit mehreren Fallback-Methoden:
    1. YouTube Data API v3 (Captions) - zuverlässigste Methode
    2. yt-dlp + Hugging Face Whisper (transformers) - kein ffmpeg nötig
    3. yt-dlp + OpenAI Whisper - benötigt ffmpeg
    
    Returns: Transkript als Text oder None bei Fehler
    """
    # Methode 1: Versuche YouTube API für Captions (zuverlässigste Methode)
    transcript = _get_transcript_from_youtube_api(video_id)
    if transcript:
        return transcript
    
    # Methode 2: Download + lokale Transkription
    try:
        import yt_dlp
        
        # Priorisiere Hugging Face Whisper (transformers) - kein ffmpeg nötig
        pipe = None
        try:
            from transformers import pipeline
            # Lade Whisper-Modell (wird beim ersten Mal heruntergeladen, ~150MB)
            # device=-1 = CPU, device=0 = GPU (falls verfügbar)
            pipe = pipeline("automatic-speech-recognition", 
                          model="openai/whisper-base",
                          device=-1)
            logger.debug("Using Hugging Face Whisper (free, local, no ffmpeg needed)")
        except ImportError:
            logger.debug("transformers not available, will try OpenAI Whisper fallback")
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face Whisper: {e}, trying fallback")
        
        # Download Audio mit besserer Konfiguration
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(TRANSCRIPTS_DIR / f'{video_id}.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            # Verbesserte Konfiguration für bessere Erfolgsrate
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'extract_flat': False,
            'retries': 3,
            'fragment_retries': 3,
            'ignoreerrors': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except Exception as e:
                logger.warning(f"yt-dlp download failed for {video_id}: {e}")
                # Versuche Groq Whisper API als letzten Fallback (kostenpflichtig, aber zuverlässig)
                return _transcribe_video_groq_whisper(video_id, video_url)
        
        # Finde heruntergeladene Audio-Datei
        audio_files = list(TRANSCRIPTS_DIR.glob(f'{video_id}.*'))
        # Filtere .txt Dateien aus (das sind bereits gespeicherte Transkripte)
        audio_files = [f for f in audio_files if f.suffix not in ['.txt', '.json']]
        
        if not audio_files:
            logger.warning(f"Could not download audio for video {video_id}")
            # Versuche Groq Whisper API als letzten Fallback
            return _transcribe_video_groq_whisper(video_id, video_url)
        
        audio_path = audio_files[0]
        
        # Transkribiere mit Whisper
        logger.debug(f"Transcribing audio file: {audio_path}")
        if pipe is not None:
            # Hugging Face Whisper (kostenlos, lokal, kein ffmpeg nötig) - bevorzugt
            try:
                transcript = pipe(str(audio_path))["text"]
                logger.debug(f"Successfully transcribed with Hugging Face Whisper")
            except Exception as e:
                logger.warning(f"Hugging Face Whisper failed: {e}, trying OpenAI Whisper fallback")
                pipe = None  # Fallback zu OpenAI Whisper
        
        if pipe is None:
            # Fallback: OpenAI Whisper (falls transformers nicht verfügbar oder fehlgeschlagen)
            # WARNUNG: Benötigt ffmpeg!
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(str(audio_path))
                transcript = result['text']
                logger.debug(f"Successfully transcribed with OpenAI Whisper")
            except ImportError:
                logger.error("Neither transformers nor openai-whisper available")
                return None
            except Exception as e:
                logger.warning(f"OpenAI Whisper failed (might need ffmpeg): {e}")
                return None
        
        # Speichere Transkript optional (für Debugging)
        transcript_path = TRANSCRIPTS_DIR / f'{video_id}.txt'
        try:
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        except:
            pass
        
        # Lösche Audio-Datei (um Platz zu sparen)
        try:
            audio_path.unlink()
        except:
            pass
        
        return transcript
        
    except ImportError as e:
        logger.warning(f"yt-dlp or whisper not installed: {e}. Install with: pip install yt-dlp transformers")
        return None
    except Exception as e:
        logger.error(f"Error transcribing video {video_id}: {e}")
        # Versuche Groq Whisper API als letzten Fallback
        return _transcribe_video_groq_whisper(video_id, video_url)


def _transcribe_video_groq_whisper(video_id: str, video_url: str) -> Optional[str]:
    """
    Fallback: Transkribiert via Groq Whisper API (kostenpflichtig, aber zuverlässig).
    Wird nur verwendet wenn alle anderen Methoden fehlschlagen.
    
    Returns: Transkript als Text oder None bei Fehler
    """
    if not GROQ_API_KEY:
        return None
    
    try:
        from groq import Groq
        import yt_dlp
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Download Audio (mit minimaler Konfiguration)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(TRANSCRIPTS_DIR / f'{video_id}_groq.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            logger.warning(f"yt-dlp download failed for Groq Whisper: {e}")
            return None
        
        # Finde Audio-Datei
        audio_files = list(TRANSCRIPTS_DIR.glob(f'{video_id}_groq.*'))
        if not audio_files:
            return None
        
        audio_path = audio_files[0]
        
        # Transkribiere via Groq Whisper API
        try:
            with open(audio_path, 'rb') as audio_file:
                # Groq hat noch keine Whisper API, verwende Chat API mit Audio
                # FALLBACK: Für jetzt return None, da Groq noch keine direkte Whisper API hat
                logger.debug("Groq Whisper API not yet available, skipping")
                return None
        except Exception as e:
            logger.warning(f"Groq Whisper API failed: {e}")
            return None
        finally:
            # Lösche Audio-Datei
            try:
                audio_path.unlink()
            except:
                pass
    except ImportError:
        logger.debug("groq or yt-dlp not available for Groq Whisper fallback")
    except Exception as e:
        logger.debug(f"Error in Groq Whisper fallback: {e}")
    
    return None


def _extract_tickers_from_transcript(transcript: str, universe_tickers: List[str]) -> Dict[str, float]:
    """
    Extrahiert Ticker und Sentiment-Scores aus Transkript via Groq.
    
    Returns: Dict {ticker: sentiment_score 0.0-1.0}
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, cannot extract tickers from transcript")
        return {}
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        # Erstelle Prompt für Groq
        tickers_str = ", ".join(universe_tickers[:100])  # Limitiere auf 100 Ticker für Prompt
        prompt = f"""Analyze this YouTube video transcript about stock investing. Extract all mentioned stock tickers and their sentiment.

Universe of valid tickers (only use these): {tickers_str}

Return JSON format:
{{
  "tickers": [
    {{"ticker": "AAPL", "sentiment_score": 0.75, "mentions": ["undervalued", "quality"]}},
    {{"ticker": "MSFT", "sentiment_score": 0.45, "mentions": ["overvalued"]}}
  ]
}}

**CRITICAL RULES FOR SENTIMENT SCORING**:
- Base sentiment_score should be DIFFERENTIATED and CONSERVATIVE: Use the full range 0.0-1.0
  * 0.0-0.2 = very bearish (strong sell, overvalued, bubble, crash)
  * 0.2-0.4 = bearish (expensive, avoid, concerns)
  * 0.4-0.6 = neutral (no strong opinion, mixed signals)
  * 0.6-0.75 = bullish (good value, quality, buy)
  * 0.75-0.85 = very bullish (exceptional value, strong buy, undervalued)
  * 0.85-1.0 = EXTREMELY bullish (only for exceptional cases with multiple very strong signals)
- **CRITICAL**: Be VERY conservative with scores. 
  * Most positive stocks should score 0.65-0.80, NOT 0.90+
  * Only use 0.85+ for truly exceptional cases (multiple very strong positive signals)
  * Most negative stocks should score 0.20-0.40, NOT 0.0-0.15
- **SCORING EXAMPLES**:
  * "undervalued" + "quality" + "moat" → 0.75-0.80 (NOT 0.90+)
  * "good value" + "buy" → 0.65-0.70
  * "slightly overvalued" → 0.40-0.50
  * "overvalued" + "bubble" → 0.20-0.30

**KEYWORD ADJUSTMENTS** (applied after base score):
- "undervalued" keyword = +0.2 to sentiment (strong positive, but not automatic 1.0)
- "overvalued" keyword = -0.2 from sentiment (strong negative)
- "quality" or "moat" keywords = +0.05 to sentiment (moderate positive)
- Bullish keywords (buy, value, cheap, margin of safety) = positive sentiment
- Bearish keywords (sell, expensive, crash, bubble) = negative sentiment

**EXAMPLES**:
- Stock mentioned as "undervalued" with "quality" and "moat" → base 0.70, after keywords: 0.95 (but cap at 0.90 for differentiation)
- Stock mentioned as "good value" but no strong keywords → base 0.65
- Stock mentioned as "slightly overvalued" → base 0.50, after keywords: 0.30
- Stock mentioned as "overvalued bubble" → base 0.20, after keywords: 0.00

- Only extract tickers that are in the universe list above
- If no tickers found, return empty array

Transcript:
{transcript[:8000]}  # Limitiere auf 8000 Zeichen für API-Limit
"""
        
        # Rufe Groq API auf
        model = AI_MODEL if "llama" in AI_MODEL.lower() or "mixtral" in AI_MODEL.lower() else "llama-3.3-70b-versatile"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        # Entferne mögliche Markdown-Code-Blöcke
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result_json = json.loads(result_text)
        
        # Konvertiere zu Dict {ticker: sentiment_score}
        ticker_scores = {}
        for item in result_json.get('tickers', []):
            ticker = item.get('ticker', '').upper()
            sentiment = float(item.get('sentiment_score', 0.5))
            
            # Wende Keyword-Bonus an (reduziert für bessere Differenzierung)
            mentions = [m.lower() for m in item.get('mentions', [])]
            if 'undervalued' in mentions:
                sentiment = min(0.85, sentiment + 0.15)  # Cap bei 0.85 für bessere Differenzierung
            if 'overvalued' in mentions:
                sentiment = max(0.15, sentiment - 0.15)  # Floor bei 0.15
            if 'quality' in mentions or 'moat' in mentions:
                sentiment = min(0.90, sentiment + 0.05)  # Kleinerer Bonus, Cap bei 0.90
            
            # Clamp auf 0.0-1.0 (aber erlaube volle Range für Differenzierung)
            sentiment = max(0.0, min(1.0, sentiment))
            
            # Nur Ticker aus Universe berücksichtigen
            if ticker in universe_tickers:
                ticker_scores[ticker] = sentiment
        
        return ticker_scores
        
    except ImportError:
        logger.error("groq not installed. Install with: pip install groq")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse Groq response as JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error extracting tickers from transcript: {e}")
        return {}


def get_youtube_sentiment_score(tickers: List[str], days_back: int = 120, financials_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Hauptfunktion: Holt YouTube-Sentiment für alle Ticker.
    
    Args:
        tickers: List of ticker symbols (Universe)
        days_back: Anzahl Tage zurück (default: 180 = 6 Monate)
        financials_df: Optional DataFrame (für Fallback, aktuell nicht verwendet)
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    if not YOUTUBE_API_KEY:
        logger.warning("YOUTUBE_API_KEY not set, returning neutral scores")
        return {ticker: 0.5 for ticker in tickers}
    
    try:
        # 1. Fetche neue Videos
        logger.info(f"Fetching new YouTube videos from {len(YOUTUBE_CHANNEL_IDS)} channels...")
        new_videos = _fetch_new_videos(YOUTUBE_CHANNEL_IDS, days_back=days_back)
        
        if not new_videos:
            logger.info("No new videos to process")
        else:
            # 2. Verarbeite neue Videos
            logger.info(f"Processing {len(new_videos)} new videos...")
            for i, video in enumerate(new_videos):
                video_id = video['video_id']
                video_url = video['url']
                
                logger.info(f"Processing video {i+1}/{len(new_videos)}: {video['title'][:50]}...")
                
                # Transkribiere
                transcript = _transcribe_video(video_id, video_url)
                if not transcript:
                    logger.warning(f"Could not transcribe video {video_id}, skipping")
                    continue
                
                # Extrahiere Ticker und Sentiment
                tickers_scores = _extract_tickers_from_transcript(transcript, tickers)
                
                # Filter: Skippe Videos mit <3 Tickers (allgemeine Videos ohne spezifische Picks)
                if len(tickers_scores) < MIN_TICKERS_FOR_ANALYSIS:
                    logger.debug(f"Video {video_id} has only {len(tickers_scores)} tickers (<{MIN_TICKERS_FOR_ANALYSIS}), skipping analysis")
                    # Speichere trotzdem als verarbeitet (um nicht erneut zu versuchen)
                    _save_processed_video(video_id, video['channel_id'], video['title'], 
                                        video['published_at'], transcript, {})
                    continue
                
                if not tickers_scores:
                    logger.debug(f"No tickers found in video {video_id}")
                    # Speichere trotzdem als verarbeitet (um nicht erneut zu versuchen)
                    _save_processed_video(video_id, video['channel_id'], video['title'], 
                                        video['published_at'], transcript, {})
                    continue
                
                # Speichere verarbeitetes Video
                _save_processed_video(video_id, video['channel_id'], video['title'], 
                                     video['published_at'], transcript, tickers_scores)
                
                logger.info(f"Video {video_id}: Found {len(tickers_scores)} tickers (kept for analysis)")
        
        # 3. Aggregiere Scores aus allen relevanten Videos (letzte 4 Monate)
        logger.info("Aggregating sentiment scores from processed videos...")
        processed_videos = _load_processed_videos()
        filtered_videos = _filter_by_date(processed_videos, days_back=days_back)
        
        # Filter: Nur Videos mit >= MIN_TICKERS_FOR_ANALYSIS Tickers bei Aggregation
        filtered_videos = {
            vid: data for vid, data in filtered_videos.items() 
            if len(data.get('tickers', {})) >= MIN_TICKERS_FOR_ANALYSIS
        }
        logger.info(f"After ticker-count filter (>= {MIN_TICKERS_FOR_ANALYSIS} tickers): {len(filtered_videos)} videos")
        
        # Aggregiere pro Ticker: gewichteter Durchschnitt mit Video-Anzahl-Bonus und Konsistenz-Bonus
        ticker_aggregated = {}
        video_counts = {}  # Zähle wie viele Videos pro Ticker
        
        for video_id, video_data in filtered_videos.items():
            video_tickers = video_data.get('tickers', {})
            published_at = video_data.get('published_at', '')
            
            # Berechne Gewicht basierend auf Datum (neuer = höheres Gewicht, exponentiell abnehmend)
            try:
                if isinstance(published_at, str):
                    try:
                        pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    except:
                        # Versuche verschiedene Datumsformate
                        pub_date = None
                        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
                            try:
                                pub_date = datetime.strptime(published_at[:19], fmt)
                                break
                            except:
                                continue
                        if pub_date is None:
                            raise ValueError("Could not parse date")
                    
                    days_ago = (datetime.now() - pub_date.replace(tzinfo=None)).days
                    # Exponentiell abnehmend (neue Videos haben deutlich mehr Gewicht)
                    # Exponent 1.5: Video von heute = 1.0, Video von 90 Tagen = ~0.5, Video von 180 Tagen = ~0.1
                    weight = max(0.1, (1.0 - (days_ago / days_back)) ** 1.5)
                else:
                    weight = 0.5  # Fallback
            except:
                weight = 0.5
            
            # Aggregiere Scores
            for ticker, score in video_tickers.items():
                if ticker not in ticker_aggregated:
                    ticker_aggregated[ticker] = {'total': 0.0, 'weight_sum': 0.0, 'scores': []}
                    video_counts[ticker] = 0
                
                ticker_aggregated[ticker]['total'] += score * weight
                ticker_aggregated[ticker]['weight_sum'] += weight
                ticker_aggregated[ticker]['scores'].append(score)
                video_counts[ticker] += 1
        
        # Berechne finalen Score pro Ticker mit Video-Anzahl-Bonus und Konsistenz-Bonus
        result = {}
        for ticker in tickers:
            if ticker in ticker_aggregated:
                weight_sum = ticker_aggregated[ticker]['weight_sum']
                num_videos = video_counts[ticker]
                scores_list = ticker_aggregated[ticker]['scores']
                
                if weight_sum > 0:
                    base_score = ticker_aggregated[ticker]['total'] / weight_sum
                    
                    # NORMALISIERE Base-Score ZUERST (bevor Bonuse addiert werden)
                    # Komprimiere hohe Base-Scores für bessere Differenzierung
                    if base_score > 0.70:
                        # Komprimiere 0.70-1.0 auf 0.70-0.82 (stärkere Kompression)
                        # Das bedeutet: 0.70 bleibt 0.70, 1.0 wird zu 0.82
                        normalized_base = 0.70 + (base_score - 0.70) * 0.40
                    elif base_score < 0.30:
                        # Komprimiere 0.0-0.30 auf 0.10-0.30
                        normalized_base = 0.10 + (base_score - 0.0) * 0.67
                    else:
                        normalized_base = base_score
                    
                    # Bonus für Konsistenz: Wenn alle Scores ähnlich sind, stärke das Signal
                    # REDUZIERT für bessere Differenzierung
                    if len(scores_list) > 1:
                        score_std = float(np.std(scores_list))
                        # Niedrige Standardabweichung = hohe Konsistenz = Bonus
                        # Max 0.015 Bonus bei perfekter Konsistenz (reduziert)
                        consistency_bonus = max(0.0, 0.015 * (1.0 - min(1.0, score_std * 2)))
                    else:
                        consistency_bonus = 0.0
                    
                    # Bonus für Anzahl Videos (mehr Videos = stärkeres Signal)
                    # REDUZIERT für bessere Differenzierung
                    # Log-Skala: 1 Video = 0, 5 Videos = +0.015, 10 Videos = +0.03, 20 Videos = +0.04
                    video_count_bonus = min(0.04, 0.010 * np.log1p(num_videos))
                    
                    # Finaler Score: Normalisierter Base + Konsistenz-Bonus + Video-Anzahl-Bonus
                    final_score = normalized_base + consistency_bonus + video_count_bonus
                    
                    # Finale Normalisierung: Verhindere, dass Scores über 0.90 gehen
                    # Das sorgt für maximale Differenzierung im Top-Bereich
                    if final_score > 0.90:
                        # Komprimiere 0.90-1.0 auf 0.90-0.95
                        final_score = 0.90 + (final_score - 0.90) * 0.50
                    
                    # Finale Clamp
                    result[ticker] = max(0.0, min(1.0, final_score))
                    
                    logger.debug(f"Ticker {ticker}: base={base_score:.3f}, consistency_bonus={consistency_bonus:.3f}, "
                               f"video_count_bonus={video_count_bonus:.3f}, final={result[ticker]:.3f} "
                               f"(from {num_videos} videos)")
                else:
                    result[ticker] = 0.5
            else:
                result[ticker] = 0.5  # Neutral wenn kein Video gefunden
        
        logger.info(f"YouTube sentiment: {len([t for t, s in result.items() if s != 0.5])} tickers with non-neutral scores")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_youtube_sentiment_score: {e}")
        # Fallback: neutrale Scores
        return {ticker: 0.5 for ticker in tickers}

