import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import PROCESSED_DATA_DIR, TARGET_COLUMN, MODEL_NAME
from src.common.utils import logger, validate_dataframe, validate_asset
from src.inference.model_loader import ModelLoader


def predict(ticker: str):
    """
    Make predictions for a specific asset

    Args:
        ticker: Asset ticker (from companies.py)

    Returns:
        DataFrame with predictions and probabilities
    """
    # ✅ Validate asset
    validate_asset(ticker)

    data_path = PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Features file not found: {data_path}")

    logger.info(f"Making predictions for {ticker} from {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    validate_dataframe(df)
    logger.info(f"  Loaded {len(df)} records")

    # Prepare features
    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    # Load model (ticker-aware)
    loader = ModelLoader()
    model = loader.get_model(model_name=f"{MODEL_NAME}_{ticker}")

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Create results DataFrame
    results = pd.DataFrame({
        "Date": df["Date"],
        "prediction": predictions,
        "prob_down": probabilities[:, 0],
        "prob_up": probabilities[:, 1],
        "confidence": np.max(probabilities, axis=1),
    })

    results["direction"] = results["prediction"].map({0: "DOWN", 1: "UP"})

    logger.info("✓ Predictions complete")
    logger.info(f"  UP predictions: {(predictions == 1).sum()}")
    logger.info(f"  DOWN predictions: {(predictions == 0).sum()}")
    logger.info(f"  Average confidence: {results['confidence'].mean():.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    preds = predict(args.ticker)
    print(preds.tail(10))
