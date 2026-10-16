# Techno-Fundamental Stock Screener

A dark-themed Streamlit app for screening US and Indian equities using both technical and fundamental criteria.

## Features

- **Universes** — US S&P 500, NSE Nifty 500, or custom ticker list
- **Technical signals** — SMA 20/50/150/200, 52-week High/Low, RS Rating (percentile 1–99), Minervini Trend Template
- **Fundamental signals** — Quarterly EPS YoY growth, Revenue YoY growth, EPS Acceleration
- **Filters** — Sidebar sliders for min EPS/Revenue growth and max distance from 52-week high
- **Export** — Download screened results as CSV or JSON
- **Scan history** — Last 5 runs stored in SQLite (`screener_history.db`), reloadable anytime

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.9+
- See `requirements.txt` (streamlit, yfinance, pandas, numpy, lxml)

## Data Source

All market data is fetched via [yFinance](https://github.com/ranaroussi/yfinance). Scan history is persisted locally in a SQLite database — no external database required.

> **Not financial advice.** For educational and research purposes only.
