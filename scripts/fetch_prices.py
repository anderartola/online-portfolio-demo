"""Download daily adjusted close prices and write them to data/prices.json.

Run by the GitHub Action in .github/workflows/update-data.yml after each US
trading day, so the page always loads up-to-date data. Prices are adjusted for
splits and dividends (total return), which is what a buy-and-hold investor
actually earns.

    pip install yfinance pandas
    python scripts/fetch_prices.py
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

START = "2010-01-01"

ASSETS = [
    {"ticker": "SPY", "name": "S&P 500", "kind": "Equity index ETF"},
    {"ticker": "GLD", "name": "Gold", "kind": "Commodity ETF"},
    {"ticker": "TLT", "name": "Long-term Treasuries", "kind": "Bond ETF (20+ yr)"},
    {"ticker": "IEF", "name": "Mid-term Treasuries", "kind": "Bond ETF (7–10 yr)"},
    {"ticker": "AAPL", "name": "Apple", "kind": "Stock"},
    {"ticker": "MSFT", "name": "Microsoft", "kind": "Stock"},
    {"ticker": "NVDA", "name": "NVIDIA", "kind": "Stock"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "kind": "Stock"},
    {"ticker": "XOM", "name": "ExxonMobil", "kind": "Stock"},
]

OUT = Path(__file__).resolve().parent.parent / "data" / "prices.json"


def round_sig(x, sig=6):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def main():
    tickers = [a["ticker"] for a in ASSETS]
    df = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    # Keep only days on which every asset traded, so returns line up.
    df = df[tickers].dropna(how="any")
    if len(df) < 500:
        raise SystemExit(f"Only {len(df)} rows downloaded; refusing to overwrite data.")

    payload = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "source": "Yahoo Finance (adjusted close, via yfinance)",
        "assets": ASSETS,
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "prices": {t: [round_sig(float(v)) for v in df[t].values] for t in tickers},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {len(df)} days ({payload['dates'][0]} → {payload['dates'][-1]}) to {OUT}")


if __name__ == "__main__":
    main()
