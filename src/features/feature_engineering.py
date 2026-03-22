import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    ROLLING_WINDOWS, VOLATILITY_WINDOW, TARGET_COLUMN,
    PROCESSED_DATA_DIR, RAW_DATA_DIR, DEFAULT_ASSET,
)
from src.common.utils import logger, validate_dataframe


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in ROLLING_WINDOWS:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_upper"] = sma_20 + 2 * std_20
    df["BB_lower"] = sma_20 - 2 * std_20
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma_20
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_14) / (high_14 - low_14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # ATR
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    df["ATR_14"] = df["TR"].rolling(window=14).mean()
    df.drop(columns=["TR"], inplace=True)

    # Volatility & returns
    df["volatility"] = df["Close"].rolling(window=VOLATILITY_WINDOW).std()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Momentum
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)

    # Volume features
    df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + 1e-9)
    df["volume_change"] = df["Volume"].pct_change()

    # Price position features
    df["hl_spread"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    # Price vs moving averages
    for w in ROLLING_WINDOWS:
        df[f"price_vs_SMA_{w}"] = (df["Close"] - df[f"SMA_{w}"]) / (df[f"SMA_{w}"] + 1e-9)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_close"] = df["Close"].shift(-1)
    df[TARGET_COLUMN] = (df["next_close"] > df["Close"]).astype(int)
    df.drop(columns=["next_close"], inplace=True)
    return df


def engineer_features(input_path: Path, output_path: Path, is_training: bool = True) -> pd.DataFrame:
    logger.info(f"Engineering features from {input_path}")
    df = pd.read_csv(input_path)
    validate_dataframe(df, required_columns=["Open", "High", "Low", "Close", "Volume"])
    logger.info(f"Loaded {len(df)} rows")

    df = calculate_technical_indicators(df)

    if is_training:
        df = create_target(df)
        logger.info(f"Target column created: {TARGET_COLUMN}")

    before = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN values")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--ticker", type=str, default=DEFAULT_ASSET)
    args = parser.parse_args()
    ticker = args.ticker.upper()

    if args.mode == "train":
        input_path = RAW_DATA_DIR / f"historical_{ticker}.csv"
        output_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
        engineer_features(input_path, output_path, is_training=True)
    else:
        input_path = RAW_DATA_DIR / f"realtime_{ticker}.csv"
        output_path = PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv"
        engineer_features(input_path, output_path, is_training=False)
