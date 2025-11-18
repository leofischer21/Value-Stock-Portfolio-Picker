# Value-Stock-Portfolio-Picker

**Goal:** automatisches Screening / Ranking großer, qualitativ hochwertiger und unterbewerteter Aktien (MarketCap ≥ 30bn) und Vorschlag eines diversifizierten Portfolios (20–25 Titel). Starter-Repo mit Data ingestion (yfinance, dataroma), Scoring, Portfolio-Konstruktion und Rebalancing-Regeln.

## Quickstart
1. Clone repo
2. Create conda env: `conda env create -f environment.yml`
3. Activate: `conda activate value_portfolio_picker`
4. Run demo: `python src/run_picker.py`

## Struktur
- `environment.yml`, `requirements.txt` — dependencies
- `src/` — core modules
  - `run_picker.py` — main starter script (MVP)
  - `fetch_dataroma.py` — scraper to fetch recent superinvestor buys from dataroma
  - `portfolio.py` — portfolio construction + rebalancing logic
- `notebooks/demo_colab.ipynb` — interactive demo (cells in repo)
- `examples/selected_portfolio.csv` — example output

## Design
- Universe: S&P500 + Nasdaq100 (user can extend)
- Filters: marketCap ≥ 30_000_000_000
- Scoring: Weighted combination of Valuation (PE/PFCF), Quality (ROE, margins), Moat proxies (consistent ROIC/margins), Superinvestor interest, Risk (beta)
- Rebalancing: monthly cadence + threshold-based replacements to avoid churn

## Data sources & legal
- Uses Yahoo Finance (`yfinance`) for fundamentals; Dataroma scraping for superinvestor buys. Respect Terms of Service and rate limits.

## Next steps / Improvements
- Add Morningstar fair value ingestion (requires subscription / API)
- Backtesting engine (zipline/backtrader)
- Chart-image parsing pipeline using Tesseract + PlotDigitizer
