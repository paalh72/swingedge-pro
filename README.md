# SwingEdge Pro v1.0

Evidence-based swing trading scanner built with Streamlit.

## Features

- **Confluence-based scoring (0-10)** — requires multiple independent signals to confirm
- **Market regime filter** — SPY + VIX based regime detection (Bull/Bear/Fear/Neutral)
- **Built-in backtester** — test strategy against historical data with equity curve
- **Position sizing** — fixed-risk model (never risk more than X% per trade)
- **Trailing stop + R:R targets** — complete exit strategy
- **10 independent signals**: Trend, Momentum, Volume, Price Structure, Volatility Squeeze, Ichimoku, Stochastic RSI, Bollinger Bounce, DeMark Sequential, Volume Profile POC

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
[general]
password = "your_password"

[connections.gsheets]
spreadsheet = "your_google_sheet_url"
type = "service_account"
project_id = ""
private_key_id = ""
private_key = ""
client_email = ""
client_id = ""
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

## Run

```bash
streamlit run app.py
```

## Architecture

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit UI and orchestration |
| `indicators.py` | All technical indicator calculations |
| `scoring.py` | Confluence scoring and entry quality |
| `market_regime.py` | SPY/VIX market regime detection |
| `backtester.py` | Walk-forward backtesting engine |
| `risk_manager.py` | Position sizing and trailing stops |
| `chart_builder.py` | Plotly chart construction |
| `data_sources.py` | Ticker lists for each market |
| `data_fetcher.py` | yfinance data fetching |
| `watchlist.py` | Google Sheets watchlist persistence |
