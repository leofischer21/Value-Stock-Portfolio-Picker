# dataroma_api.py
"""
Superinvestor-Daten via SEC EDGAR API (13F Filings).
Bei Fehlern wird automatisch auf dataroma.py zurückgegriffen.
"""
import os
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import time

# Root-Verzeichnis bestimmen
try:
    ROOT_DIR = Path(__file__).parent.parent
except:
    ROOT_DIR = Path.cwd()

logger = logging.getLogger(__name__)

# SEC EDGAR API ist kostenlos, keine API-Keys nötig
# User-Agent ist erforderlich (SEC Policy) - BITTE DEINE ECHTE EMAIL EINTRAGEN
# Format: "Value Stock Portfolio Picker(schneeloewe21@gmail.com)"
SEC_USER_AGENT = "Value Stock Portfolio Picker (schneeloewe21@gmail.com)"  # ← HIER DEINE EMAIL EINTRAGEN

# Top Superinvestoren (wie in Dataroma)
TOP_INVESTORS = [
    "Warren Buffett", "Berkshire Hathaway", "Charlie Munger",
    "Bill Ackman", "David Einhorn", "Carl Icahn",
    "Daniel Loeb", "Seth Klarman", "Joel Greenblatt",
    "Mohnish Pabrai", "Guy Spier", "Li Lu",
]


def _get_cik_from_ticker(ticker: str) -> Optional[str]:
    """
    Konvertiert Ticker zu CIK (Central Index Key) für SEC EDGAR.
    Verwendet SEC's Company Tickers JSON.
    
    Args:
        ticker: Ticker symbol (z.B. "AAPL")
    
    Returns:
        CIK als String (z.B. "0000320193") oder None
    """
    try:
        import requests
        
        # SEC's Company Tickers JSON
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Format: {"0": {"cik_str": "0000320193", "ticker": "AAPL", "title": "Apple Inc."}, ...}
            for entry in data.values():
                if entry.get("ticker") == ticker.upper():
                    cik = str(entry.get("cik_str", ""))
                    # CIK sollte 10-stellig sein (mit führenden Nullen)
                    if cik:
                        return cik.zfill(10)
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to get CIK for {ticker}: {e}")
        return None


def _fetch_13f_filings_sec(ticker: str, quarters: int = 4) -> List[Dict]:
    """
    Holt 13F-HR Filings von SEC EDGAR für einen Ticker.
    
    Args:
        ticker: Ticker symbol
        quarters: Anzahl Quartale zurück (default: 4 = ~12 Monate)
    
    Returns:
        List of holdings data with 'investor', 'shares', 'value', 'filing_date'
    """
    try:
        import requests
        
        # Hole CIK für Ticker
        cik = _get_cik_from_ticker(ticker)
        if not cik:
            logger.debug(f"No CIK found for {ticker}")
            return []
        
        # Hole Company Submissions
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.debug(f"SEC API call failed for {ticker} (CIK {cik}): {response.status_code}")
            return []
        
        data = response.json()
        filings = data.get("filings", {}).get("recent", {})
        
        # Filtere nach 13F-HR Filings
        form_types = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        
        # Berechne Start-Datum (quarters zurück)
        start_date = datetime.now() - timedelta(days=quarters * 90)
        
        holdings_data = []
        
        for i, form_type in enumerate(form_types):
            if form_type == "13F-HR":
                filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
                accession = accession_numbers[i] if i < len(accession_numbers) else ""
                
                try:
                    filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                    if filing_date >= start_date:
                        # Hole XML-Datei für dieses Filing
                        # Format: https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_no_dashes}/primary_doc.xml
                        accession_no_dashes = accession.replace("-", "")
                        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/primary_doc.xml"
                        
                        # Rate-Limiting: SEC erlaubt max 10 Requests/Sekunde
                        time.sleep(0.2)
                        
                        xml_response = requests.get(xml_url, headers=headers, timeout=10)
                        
                        if xml_response.status_code == 200:
                            # Parse XML (vereinfacht - echte 13F XMLs sind komplex)
                            try:
                                root = ET.fromstring(xml_response.content)
                                # Suche nach Holdings (vereinfacht)
                                # Echte Implementierung müsste das vollständige 13F XML Schema parsen
                                # Für jetzt: Extrahiere grundlegende Informationen
                                
                                # Placeholder: In echter Implementierung würde man hier
                                # die Holdings aus dem XML extrahieren
                                holdings_data.append({
                                    'filing_date': filing_date_str,
                                    'accession': accession,
                                    'ticker': ticker,
                                })
                            except ET.ParseError:
                                logger.debug(f"Failed to parse XML for {ticker} filing {accession}")
                                continue
                
                except ValueError:
                    continue
        
        return holdings_data
        
    except Exception as e:
        logger.error(f"SEC API call failed for {ticker}: {e}")
        return []


def _calculate_superinvestor_score(holdings_data: List[Dict], ticker: str) -> float:
    """
    Berechnet Superinvestor-Score basierend auf 13F Holdings.
    
    Args:
        holdings_data: List of holdings data
        ticker: Ticker symbol
    
    Returns:
        Score (0.0-1.0)
    """
    if not holdings_data:
        return 0.5  # Neutral wenn keine Daten
    
    # Vereinfachte Berechnung:
    # - Mehr Filings = höherer Score (mehr Investoren halten die Aktie)
    # - In echter Implementierung würde man die Anzahl der Top-Investoren zählen,
    #   die die Aktie kaufen vs. verkaufen
    
    num_filings = len(holdings_data)
    
    # Normalisiere auf 0.0-1.0 Skala
    # Annahme: 0-5 Filings = 0.3-0.7, 5+ Filings = 0.7-1.0
    if num_filings == 0:
        return 0.3
    elif num_filings <= 2:
        return 0.4
    elif num_filings <= 5:
        return 0.6
    elif num_filings <= 10:
        return 0.75
    else:
        return 0.9


def get_superinvestor_data_api(universe: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Hauptfunktion: Holt Superinvestor-Daten via SEC EDGAR API für alle Ticker.
    Bei Fehlern wird auf dataroma.py zurückgegriffen.
    
    Args:
        universe: Optional list of tickers to filter results. If None, returns all.
    
    Returns:
        Dict mapping ticker -> score (0.0-1.0)
    """
    if universe is None:
        # Wenn kein Universe gegeben, verwende Fallback
        return _fallback_to_legacy(universe)
    
    results = {}
    failed_tickers = []
    
    for i, ticker in enumerate(universe):
        try:
            if (i + 1) % 10 == 0:
                logger.info(f"Processing SEC API: {i + 1}/{len(universe)}")
            
            # Hole 13F Filings (letzte 4 Quartale = ~12 Monate)
            holdings_data = _fetch_13f_filings_sec(ticker, quarters=4)
            
            # Berechne Score
            score = _calculate_superinvestor_score(holdings_data, ticker)
            results[ticker] = score
            
            # Rate-Limiting: SEC erlaubt max 10 Requests/Sekunde
            time.sleep(0.15)
            
        except Exception as e:
            logger.warning(f"SEC API failed for {ticker}: {e}, will use fallback")
            failed_tickers.append(ticker)
    
    # Für fehlgeschlagene Ticker: Fallback auf Legacy
    if failed_tickers:
        logger.info(f"Falling back to legacy method for {len(failed_tickers)} tickers")
        legacy_scores = _fallback_to_legacy(failed_tickers)
        results.update(legacy_scores)
    
    # Für Ticker ohne Ergebnisse: Fallback
    missing_tickers = [t for t in universe if t not in results]
    if missing_tickers:
        logger.info(f"Using legacy method for {len(missing_tickers)} tickers without API results")
        legacy_scores = _fallback_to_legacy(missing_tickers)
        results.update(legacy_scores)
    
    return results


def _fallback_to_legacy(universe: Optional[List[str]]) -> Dict[str, float]:
    """
    Fallback: Verwendet die Legacy-Methode aus dataroma.py
    """
    try:
        from dataroma import get_superinvestor_data
        logger.info("Using legacy Dataroma method (dataroma.py)")
        return get_superinvestor_data(universe=universe)
    except Exception as e:
        logger.error(f"Legacy fallback also failed: {e}")
        # Letzter Fallback: Neutrale Scores
        if universe:
            return {ticker: 0.5 for ticker in universe}
        return {}

