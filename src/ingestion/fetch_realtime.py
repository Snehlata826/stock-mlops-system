import os
import time
from pathlib import Path

from matplotlib import ticker
import pandas as pd
import requests
from dotenv import load_dotenv

from src.common.config import RAW_DATA_DIR
from src.common.utils import logger, validate_asset

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

CACHE_TTL_SECONDS = 60  # 1 minute cooldown


def _cache_file_for_ticker(ticker: str) -> Path:
    """Create per-asset cache file"""
    return RAW_DATA_DIR / f"realtime_{ticker}.csv"


def is_cache_fresh(path: Path, ttl: int) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def fetch_realtime_data(
    ticker: str,
    interval: str = "15min",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch near real-time intraday data with caching, rate-limit handling,
    and safe fallback to historical data.
    """

    # ✅ Validate asset (Top 3 + Gold only)
    validate_asset(ticker)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file_for_ticker(ticker)

    # 1️⃣ Use cache if fresh (unless forced refresh)
    if not force_refresh and is_cache_fresh(cache_file, CACHE_TTL_SECONDS):
        logger.info(f"Using cached intraday data for {ticker}")
        return pd.read_csv(cache_file)

    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")

    logger.info(f"Fetching intraday data for {ticker} ({interval})")

    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}"
        f"&interval={interval}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
        "&outputsize=full"
    )

    data = requests.get(url, timeout=30).json()

    # 2️⃣ Handle rate limits
    if "Note" in data or "Information" in data:
        logger.warning("Rate-limited by Alpha Vantage")

        if cache_file.exists():
            logger.info(f"Using cached data for {ticker}")
            return pd.read_csv(cache_file)

        logger.warning("No cache found. Falling back to historical slice")
        historical_path = RAW_DATA_DIR / f"historical_{ticker}.csv"

    if historical_path.exists():
        df = pd.read_csv(historical_path).tail(50)
    else:
        raise FileNotFoundError(f"No historical data found for {ticker}")
        df.to_csv(cache_file, index=False)
        return df

    # 3️⃣ Valid intraday data
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

        df.to_csv(cache_file, index=False)
        logger.success(f"Saved {len(df)} rows → {cache_file}")
        return df

    # 4️⃣ Final fallback
    logger.warning("Intraday data unavailable. Falling back to historical slice")
    df = pd.read_csv("data/raw/historical_prices.csv").tail(50)
    df.to_csv(cache_file, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--interval", default="15min")

    args = parser.parse_args()
    fetch_realtime_data(args.ticker, args.interval)
