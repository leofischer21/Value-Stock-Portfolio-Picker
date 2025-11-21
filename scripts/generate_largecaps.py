# save as generate_largecaps.py
import requests
from bs4 import BeautifulSoup
import time
import yfinance as yf
import re
import csv
from tqdm import tqdm

BASE = "https://stockanalysis.com/list/biggest-companies/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ValueStockBot/1.0; +https://github.com/)"}

def parse_shorthand(mcap_str):
    """Convert shorthand like '4.53T', '123.4B', '987M' to float in USD."""
    if mcap_str is None:
        return None
    s = mcap_str.strip().replace(',', '').upper()
    # Some entries may be like '$123.4B' or '123.4 B'
    s = s.replace('$','').replace(' ','')
    m = re.match(r'^([0-9]*\.?[0-9]+)([TMKB]?)$', s)
    if not m:
        try:
            return float(s)
        except:
            return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == 'T':
        return val * 1e12
    if suffix == 'B':
        return val * 1e9
    if suffix == 'M':
        return val * 1e6
    if suffix == 'K':
        return val * 1e3
    return val

def get_total_pages():
    # Try to detect how many pages (StockAnalysis paginates). We'll check last page number on first page.
    r = requests.get(BASE, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    # find page links
    pagelinks = soup.select("a[aria-label='Page']")
    nums = []
    for a in pagelinks:
        try:
            nums.append(int(a.get_text().strip()))
        except:
            pass
    if nums:
        return max(nums)
    # fallback: guess 12 pages (seen on the site)
    return 12

def scrape_stockanalysis_pages(max_pages=None):
    tickers = {}
    pages = max_pages or get_total_pages()
    for p in range(1, pages+1):
        url = BASE if p == 1 else f"{BASE}?page={p}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print("Warning: page", p, "status", r.status_code)
            continue
        # Try lxml first, fallback to html.parser
        try:
            soup = BeautifulSoup(r.text, "lxml")
        except Exception:
            soup = BeautifulSoup(r.text, "html.parser")
        # rows in table
        rows = soup.select("table tbody tr")
        for tr in rows:
            tds = tr.find_all("td")
            if not tds or len(tds) < 3:
                continue
            # typically: Symbol is link in first td
            a = tds[1].find("a")
            if not a:
                continue
            symbol = a.get_text().strip()
            # Market Cap usually in column 3 or 4 - inspect text
            # On the site it shows: No. | Symbol | Company Name | Market Cap | Stock Price ...
            # So market cap might be tds[3]
            mcap_text = None
            # heuristics: find any td that contains 'B' or 'M' or 'T'
            for td in tds:
                txt = td.get_text().strip()
                if re.search(r'\d+(\.\d+)?\s*[TMBK]$', txt.replace(',','').upper()) or re.search(r'^\$?\d', txt):
                    # but this is noisy, we try column 3/4 first
                    pass
            try:
                # common layout: tds[3] contains market cap
                mcap_text = tds[3].get_text().strip()
            except Exception:
                mcap_text = None
            if not mcap_text:
                # fallback: try to find token with B/M/T
                for td in tds:
                    txt = td.get_text().strip()
                    if re.search(r'[0-9]+\.[0-9]+\s*[TMBK]|[0-9]+\s*[TMBK]$', txt.upper()):
                        mcap_text = txt
                        break
            mc = parse_shorthand(mcap_text)
            if mc is None:
                continue
            tickers[symbol] = mc
        time.sleep(0.3)
    return tickers

def is_us_nyse_nasdaq(symbol):
    # use yfinance to check exchange and marketCap
    try:
        t = yf.Ticker(symbol)
        info = t.info
        # exchange: examples: 'NMS' 'NYS' 'NYSE' 'NasdaqGS'
        exch = info.get('exchange') or info.get('fullExchangeName') or info.get('market') or ""
        exch_str = str(exch).upper()
        # some tickers map to foreign companies (ASML etc.) but listed on US exchanges too; we accept 'NASDAQ'/'NMS'/'NYSE'/'NYQ'/'NASDAQGS'
        if any(x in exch_str for x in ['NASDAQ', 'NMS', 'NASDAQGS', 'NYSE', 'NYQ', 'NYSEAMERICAN', 'AMERICAN']):
            # also ensure ticker is permissible (simple check: alnum and - and .)
            if re.match(r'^[A-Z0-9\.-]{1,7}$', symbol.upper()):
                marketcap = info.get('marketCap') or 0
                return True, exch_str, marketcap
        return False, exch_str, info.get('marketCap')
    except Exception as e:
        # if yfinance fails, return unknown
        return False, "", None

def main(output_txt="tickers_over_50B.txt", output_csv=None):
    from pathlib import Path
    from datetime import datetime
    # Root-Verzeichnis bestimmen (für Scripts in scripts/)
    ROOT_DIR = Path(__file__).parent.parent
    
    print("Scraping list pages...")
    scraped = scrape_stockanalysis_pages(max_pages=12)
    print(f"Found {len(scraped)} symbols on scraped pages. Filtering by >50B (coarse)...")
    candidates = [s for s,m in scraped.items() if (m is not None and m >= 50e9)]
    print("Coarse candidates:", len(candidates))
    final = []
    rows = []
    print("Validating with yfinance (this may take a while)...")
    for sym in tqdm(candidates):
        okay, exch, mcap = is_us_nyse_nasdaq(sym)
        # if yfinance returns a marketcap smaller than threshold, reject
        if okay and (mcap is not None and mcap >= 50e9):
            final.append(sym)
            rows.append((sym, mcap, exch))
        else:
            # try extra: sometimes StockAnalysis uses different ticker (BRK.B vs BRK-B)
            alt = sym.replace('.', '-')
            if alt != sym:
                okay2, exch2, mcap2 = is_us_nyse_nasdaq(alt)
                if okay2 and (mcap2 is not None and mcap2 >= 50e9):
                    final.append(alt)
                    rows.append((alt, mcap2, exch2))
        time.sleep(0.8)  # polite delay
    print("Final count:", len(final))
    
    # Determine output paths
    month = datetime.now().strftime("%Y-%m")
    universe_dir = ROOT_DIR / "data/tickers"
    universe_dir.mkdir(parents=True, exist_ok=True)
    
    # Write monthly CSV to data/tickers/
    if output_csv is None:
        output_csv = f"{month}.csv"
    csv_path = universe_dir / output_csv
    
    with open(csv_path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["ticker","marketCap","exchange"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])
    
    # Also write latest.csv
    latest_path = universe_dir / "latest.csv"
    with open(latest_path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["ticker","marketCap","exchange"])
        for r in rows:
            w.writerow([r[0], r[1], r[2]])
    
    # Optional: write txt file to root
    if output_txt:
        with open(ROOT_DIR / output_txt, "w") as f:
            for s in sorted(final):
                f.write(s + "\n")
    
    print(f"Wrote: {csv_path} and {latest_path}")
    return final

if __name__ == "__main__":
    main()
