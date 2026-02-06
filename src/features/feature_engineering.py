import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import (
    ROLLING_WINDOWS, VOLATILITY_WINDOW, TARGET_COLUMN,
    PROCESSED_DATA_DIR, RAW_DATA_DIR
)
from src.common.utils import logger, validate_dataframe

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators
    
    Features:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD
    - Bollinger Bands
    - Volatility
    - Volume features
    """
    df = df.copy()
    
    # Moving averages
    for window in ROLLING_WINDOWS:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma_20 + (2 * std_20)
    df['BB_lower'] = sma_20 - (2 * std_20)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / sma_20
    
    # Volatility
    df['volatility'] = df['Close'].rolling(window=VOLATILITY_WINDOW).std()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price momentum
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Volume features
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    
    # High-Low spread
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 if next day's close > today's close, else 0
    """
    df = df.copy()
    df['next_close'] = df['Close'].shift(-1)
    df[TARGET_COLUMN] = (df['next_close'] > df['Close']).astype(int)
    df.drop('next_close', axis=1, inplace=True)
    return df

def engineer_features(input_path: Path, output_path: Path, is_training: bool = True):
    """
    Main feature engineering pipeline
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed features
        is_training: Whether this is for training (creates target) or inference
    """
    logger.info(f"Engineering features from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    validate_dataframe(df, required_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    logger.info(f"  Loaded {len(df)} records")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create target (only for training)
    if is_training:
        df = create_target(df)
        logger.info(f"  Created target column: {TARGET_COLUMN}")
    
    # Drop rows with NaN (from rolling calculations)
    initial_rows = len(df)
    df.dropna(inplace=True)
    logger.info(f"  Dropped {initial_rows - len(df)} rows with NaN values")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(df)} processed records to {output_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], 
                       default="train", help="Mode: train or inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        input_path = RAW_DATA_DIR / "historical_prices.csv"
        output_path = PROCESSED_DATA_DIR / "features_train.csv"
        engineer_features(input_path, output_path, is_training=True)
    else:
        input_path = RAW_DATA_DIR / "realtime_prices.csv"
        output_path = PROCESSED_DATA_DIR / "features_inference.csv"
        engineer_features(input_path, output_path, is_training=False)
