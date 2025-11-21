# src/dataroma.py
import requests
from bs4 import BeautifulSoup
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

# Root-Verzeichnis bestimmen (funktioniert sowohl von Root als auch von scripts/)
try:
    ROOT_DIR = Path(__file__).parent.parent
except:
    ROOT_DIR = Path.cwd()

CACHE_FILE = ROOT_DIR / "data/community_data/dataroma_cache.json"


def get_superinvestor_data(universe: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Holt echte Anzahl Superinvestoren + neue Käufe von dataroma.com
    
    Args:
        universe: Optional list of tickers to filter results. If None, returns all.
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    # Check cache first
    if CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < 24*3600:  # Cache 24h gültig
            cached_data = json.load(open(CACHE_FILE))
            if universe:
                # Filter to universe tickers only
                return {t: cached_data.get(t, 0.5) for t in universe}
            return cached_data

    url = "https://www.dataroma.com/m/holdings.php?m=hold"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        session = requests.Session()
        session.headers.update(headers)
        # Erster Request triggert Cloudflare
        r = session.get("https://www.dataroma.com", timeout=15)
        time.sleep(3)
        r = session.get(url, timeout=15)
        
        soup = BeautifulSoup(r.text, 'lxml')
        data = {}
        for row in soup.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) < 6:
                continue
            ticker_cell = cols[0].get_text(strip=True)
            ticker = ticker_cell.split()[0] if ticker_cell else ""
            if not ticker or ticker == "Stock":
                continue
            
            holders = cols[3].get_text(strip=True)
            new_buys = cols[4].get_text(strip=True)
            
            try:
                holders = int(holders) if holders.isdigit() else 0
                new_buys = int(new_buys) if new_buys.isdigit() else 0
            except:
                holders = new_buys = 0
                
            if holders > 0:
                score = min(holders / 25, 1.0) * 0.7 + min(new_buys / 6, 1.0) * 0.3
                data[ticker] = round(score, 3)
        
        # Cache speichern
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(CACHE_FILE, "w"))
        
        # Filter to universe if provided
        if universe:
            # Normalize ticker formats (BRK-B vs BRK.B)
            universe_normalized = {t.upper(): t for t in universe}
            universe_normalized.update({t.replace('-', '.').upper(): t for t in universe})
            universe_normalized.update({t.replace('.', '-').upper(): t for t in universe})
            
            filtered_data = {}
            for ticker, score in data.items():
                ticker_upper = ticker.upper()
                if ticker_upper in universe_normalized:
                    # Use original ticker format from universe
                    original_ticker = universe_normalized[ticker_upper]
                    filtered_data[original_ticker] = score
            
            # Add missing tickers with neutral score
            for ticker in universe:
                if ticker not in filtered_data:
                    filtered_data[ticker] = 0.5
            
            return filtered_data
        
        return data
        
    except Exception as e:
        print(f"Dataroma scraping failed: {e}, using fallback")
        # Fallback data
        fallback = {
            'BRK-B': 1.00, 'JPM': 0.95, 'GOOGL': 0.88, 'META': 0.85,
            'UNH': 0.82, 'KO': 0.78, 'PG': 0.75, 'V': 0.70, 'MA': 0.68
        }
        
        if universe:
            # Return fallback filtered to universe
            return {t: fallback.get(t, 0.5) for t in universe}
        
        return fallback
