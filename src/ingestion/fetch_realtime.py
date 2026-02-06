import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.common.config import RAW_DATA_DIR, DEFAULT_TICKER
from src.common.utils import logger

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

CACHE_FILE = RAW_DATA_DIR / "realtime_prices.csv"
CACHE_TTL_SECONDS = 60  # 1 minute cooldown


def is_cache_fresh(path: Path, ttl: int) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def fetch_realtime_data(
    ticker: str = DEFAULT_TICKER,
    interval: str = "15min",
    force_refresh: bool = False,   # ✅ NEW
) -> pd.DataFrame:
    """
    Fetch near real-time intraday data with caching, rate-limit handling,
    and safe fallback to historical data.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Use cache if fresh (unless forced refresh)
    if not force_refresh and is_cache_fresh(CACHE_FILE, CACHE_TTL_SECONDS):
        logger.info("Using cached intraday data")
        return pd.read_csv(CACHE_FILE)

    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")

    logger.info(f"Fetching intraday data for {ticker} ({interval})")

    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}"
        f"&interval={interval}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
        "&outputsize=compact"
    )

    data = requests.get(url, timeout=30).json()

    # 2️⃣ Handle rate limit / info messages
    if "Note" in data or "Information" in data:
        logger.warning("Rate-limited by Alpha Vantage")

        if CACHE_FILE.exists():
            logger.info("Using cached intraday data")
            return pd.read_csv(CACHE_FILE)

        logger.warning("No cache found. Falling back to historical slice")
        df = pd.read_csv("data/raw/historical_prices.csv").tail(50)
        df.to_csv(CACHE_FILE, index=False)
        logger.info("Saved fallback data as realtime_prices.csv")
        return df

    # 3️⃣ Handle valid intraday data
    key = f"Time Series ({interval})"
    if key in data:
        df = pd.DataFrame.from_dict(data[key], orient="index").astype(float)

        df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            },
            inplace=True,
        )

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)

        df.to_csv(CACHE_FILE, index=False)
        logger.success(f"Saved {len(df)} rows → {CACHE_FILE}")
        return df

    # 4️⃣ Final fallback (market closed / unexpected response)
    logger.warning("Intraday data unavailable. Falling back to historical slice")
    df = pd.read_csv("data/raw/historical_prices.csv").tail(50)
    df.to_csv(CACHE_FILE, index=False)
    logger.info("Saved fallback data as realtime_prices.csv")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--interval", default="15min")

    args = parser.parse_args()
    fetch_realtime_data(args.ticker, args.interval)
