import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf # type: ignore
from dotenv import load_dotenv

from src.common.config import RAW_DATA_DIR, DEFAULT_ASSET, HISTORICAL_PERIOD
from src.common.utils import logger


load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_from_alpha_vantage(ticker: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set")

    logger.info("Using Alpha Vantage (free tier)")

    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
        "&outputsize=full"
    )

    data = requests.get(url, timeout=30).json()

    if "Time Series (Daily)" not in data:
        raise ValueError("Alpha Vantage returned no data")

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    ).astype(float)

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

    return df


def fetch_historical_data(
    ticker: str = DEFAULT_ASSET,
    period: str = HISTORICAL_PERIOD,
) -> pd.DataFrame:
    logger.info(f"Fetching historical data for {ticker} (training data)")


    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)

        if df.empty:
            raise ValueError

        df.reset_index(inplace=True)
        logger.info("Data fetched from Yahoo Finance")

    except Exception:
        logger.warning("Yahoo Finance unavailable")
        df = fetch_from_alpha_vantage(ticker)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / f"historical_{ticker}.csv"
    df.to_csv(output_path, index=False)

    logger.success(f"Saved {len(df)} rows → {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=DEFAULT_ASSET)
    parser.add_argument("--period", default=HISTORICAL_PERIOD)

    args = parser.parse_args()
    fetch_historical_data(args.ticker, args.period)
