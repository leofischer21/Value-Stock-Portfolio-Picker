# Project Structure

This document describes the organization of the Value Stock Portfolio Picker project.

## Directory Structure

```
Value-Stock-Portfolio-Picker/
├── app/                    # Streamlit dashboard application
│   ├── app.py
│   ├── dashboard.py
│   └── dashboard_final.py
│
├── data/                   # All data files (generated and cached)
│   ├── ai_scores/         # AI-generated scores (moat, quality, performance)
│   ├── cache/             # API response cache
│   ├── community_data/    # Community signals and superinvestor data
│   ├── extended_scores/   # Extended scoring metrics
│   ├── financials/        # Financial data (CSV files by month)
│   ├── scores/            # Sentiment scores (Reddit, X, YouTube, Superinvestor)
│   ├── tickers/           # Ticker lists and universe files
│   └── youtube/            # YouTube video data and transcripts
│
├── docs/                  # Documentation files
│   ├── README.md          # Documentation index
│   ├── API_SETUP.md       # API key setup instructions
│   ├── AI_SCORES_README.md # AI scoring system documentation
│   └── SETUP_AI_SCORES.md  # AI scores setup guide
│
├── examples/              # Example portfolio outputs (CSV and HTML)
│
├── scripts/               # All Python scripts
│   ├── api_utils.py       # Shared API utilities (retry logic, etc.)
│   ├── monthly_update.py  # Main monthly update script (legacy)
│   ├── monthly_update_api.py # Main monthly update script (API version)
│   ├── reddit_api.py      # Reddit sentiment via Apify API
│   ├── twitter_api.py     # X/Twitter sentiment via Apify API
│   ├── youtube_sentiment.py # YouTube video sentiment analysis
│   ├── dataroma_api.py    # Superinvestor data via SEC EDGAR API
│   ├── ai_scores.py       # AI-based scoring (moat, quality, performance)
│   ├── portfolio_calculator.py # Portfolio calculation logic
│   └── ...                # Other utility scripts
│
├── tests/                 # All test files
│   ├── test_*.py          # Unit and integration tests
│   └── ...
│
├── config.yaml            # Configuration file
├── environment.yml         # Conda environment definition
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md              # Main project documentation
```

## Key Files

### Configuration
- `config.yaml`: Main configuration file
- `.env`: Environment variables (API keys) - **NOT committed to Git**

### Main Scripts
- `scripts/monthly_update_api.py`: Main orchestrator for monthly data updates
- `scripts/monthly_update.py`: Legacy version (scraping-based)
- `app/dashboard_final.py`: Streamlit dashboard application

### Data Files
- `data/scores/{YYYY-MM}.json`: Monthly sentiment scores
- `data/financials/{YYYY-MM}.csv`: Monthly financial data
- `data/tickers/{YYYY-MM}.csv`: Monthly ticker universe

## Running the Project

1. **Setup Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate valuepicker
   ```

2. **Configure API Keys**:
   - Copy `.env.example` to `.env` (if exists)
   - Add your API keys (see `docs/API_SETUP.md`)

3. **Run Monthly Update**:
   ```bash
   python scripts/monthly_update_api.py
   ```

4. **Start Dashboard**:
   ```bash
   streamlit run app/dashboard_final.py
   ```

## Notes

- All test files are in `tests/` directory
- All documentation is in `docs/` directory
- All scripts are in `scripts/` directory
- All data is in `data/` directory (organized by type and month)
- The `examples/` directory contains sample portfolio outputs
