"""
Statistical baseline models: ARIMA and SARIMA.
Used to benchmark against XGBoost — if XGBoost can't beat these, it's not adding value.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_COLUMN
from src.common.utils import logger, validate_asset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def _direction_accuracy(y_true, y_pred_direction):
    return accuracy_score(y_true, y_pred_direction)


def run_arima_baseline(ticker: str, order=(2, 1, 2)) -> dict:
    """
    Fit ARIMA on Close prices, predict next-period direction.
    Uses a rolling one-step-ahead forecast to avoid leakage.
    """
    validate_asset(ticker)
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        logger.error("statsmodels not installed. Run: pip install statsmodels")
        return {}

    raw_path = RAW_DATA_DIR / f"historical_{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Historical data not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"].values
    n = len(close)
    min_train = 100
    test_size = max(50, int(n * 0.2))
    train_end = n - test_size

    logger.info(f"ARIMA baseline for {ticker}: train={train_end}, test={test_size}")

    predictions = []
    actuals = []

    # Rolling one-step-ahead forecast
    for i in range(train_end, n - 1):
        train_slice = close[:i]
        try:
            model = ARIMA(train_slice, order=order)
            fit = model.fit()
            forecast = fit.forecast(steps=1)[0]
            direction = 1 if forecast > train_slice[-1] else 0
            actual = 1 if close[i + 1] > close[i] else 0
            predictions.append(direction)
            actuals.append(actual)
        except Exception:
            continue

        if (i - train_end) % 20 == 0:
            logger.info(f"  ARIMA progress: {i - train_end}/{test_size}")

    if not predictions:
        return {"model": "ARIMA", "error": "No predictions generated"}

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = {
        "model": f"ARIMA{order}",
        "ticker": ticker,
        "n_test": len(actuals),
        "accuracy": accuracy_score(actuals, predictions),
        "f1": f1_score(actuals, predictions, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(actuals, predictions)
    except Exception:
        metrics["roc_auc"] = float("nan")

    logger.info(f"ARIMA Results — acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
    return metrics


def run_naive_baseline(ticker: str) -> dict:
    """
    Naive baseline: predict tomorrow = today's direction (momentum).
    Also compute random-guess (50%) as lower bound.
    """
    validate_asset(ticker)
    raw_path = RAW_DATA_DIR / f"historical_{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Historical data not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"].values
    n = len(close)
    test_start = int(n * 0.8)

    actuals = []
    naive_preds = []

    for i in range(test_start, n - 1):
        actual = 1 if close[i + 1] > close[i] else 0
        naive = 1 if close[i] > close[i - 1] else 0
        actuals.append(actual)
        naive_preds.append(naive)

    actuals = np.array(actuals)
    naive_preds = np.array(naive_preds)

    metrics = {
        "model": "Naive (momentum)",
        "ticker": ticker,
        "n_test": len(actuals),
        "accuracy": accuracy_score(actuals, naive_preds),
        "f1": f1_score(actuals, naive_preds, zero_division=0),
        "random_guess_accuracy": 0.5,
    }
    logger.info(f"Naive baseline — acc={metrics['accuracy']:.4f}")
    return metrics


def run_all_baselines(ticker: str) -> dict:
    """Run all baselines and return comparison dict."""
    logger.info(f"\nRunning all baselines for {ticker}...")

    results = {}
    results["naive"] = run_naive_baseline(ticker)

    try:
        results["arima"] = run_arima_baseline(ticker, order=(2, 1, 2))
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")
        results["arima"] = {"model": "ARIMA", "error": str(e)}

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    results = run_all_baselines(args.ticker)
    for name, r in results.items():
        print(f"\n{name}: {r}")
