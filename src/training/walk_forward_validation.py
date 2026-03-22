"""
Walk-forward validation for time-series models.
Avoids data leakage by training only on past data and testing on future data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    PROCESSED_DATA_DIR, TARGET_COLUMN, XGBOOST_PARAMS,
    RANDOM_SEED, WF_TRAIN_PERIODS, WF_TEST_PERIODS
)
from src.common.utils import logger, validate_asset
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)


def mean_absolute_error(y_true, y_prob):
    return np.mean(np.abs(y_true - y_prob))


def rmse(y_true, y_prob):
    return np.sqrt(np.mean((y_true - y_prob) ** 2))


def mape(y_true, y_prob, eps=1e-9):
    return float("nan")


def create_time_splits(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple]:
    """
    Create walk-forward splits.
    Each fold: train on first k folds, test on fold k+1.
    """
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"],utc=True).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    fold_size = n // (n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = min(train_end + fold_size, n)
        train_idx = list(range(0, train_end))
        test_idx = list(range(train_end, test_end))
        if len(train_idx) > 30 and len(test_idx) > 5:
            splits.append((train_idx, test_idx))

    logger.info(f"Created {len(splits)} walk-forward splits")
    return splits


def run_walk_forward_validation(ticker: str, n_splits: int = 5) -> Dict:
    """
    Run walk-forward validation for a ticker.
    Returns fold-level and aggregate metrics.
    """
    validate_asset(ticker)

    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"],utc=True).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)

    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    splits = create_time_splits(df, n_splits=n_splits)

    fold_results = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        params = {k: v for k, v in XGBOOST_PARAMS.items() if k != "use_label_encoder"}
        model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fold_metrics = {
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "mae": mean_absolute_error(y_test, y_prob),
            "rmse": rmse(y_test, y_prob),
            "mape": mape(y_test, y_prob),
        }
        fold_results.append(fold_metrics)
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        logger.info(
            f"  Fold {fold_idx+1}: acc={fold_metrics['accuracy']:.3f} "
            f"f1={fold_metrics['f1']:.3f} auc={fold_metrics['roc_auc']:.3f} "
            f"mae={fold_metrics['mae']:.3f}"
        )

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = (all_y_prob >= 0.5).astype(int)

    aggregate = {
        "ticker": ticker,
        "n_folds": len(fold_results),
        "accuracy_mean": np.mean([f["accuracy"] for f in fold_results]),
        "accuracy_std": np.std([f["accuracy"] for f in fold_results]),
        "f1_mean": np.mean([f["f1"] for f in fold_results]),
        "f1_std": np.std([f["f1"] for f in fold_results]),
        "roc_auc_mean": np.mean([f["roc_auc"] for f in fold_results]),
        "roc_auc_std": np.std([f["roc_auc"] for f in fold_results]),
        "mae_mean": np.mean([f["mae"] for f in fold_results]),
        "rmse_mean": np.mean([f["rmse"] for f in fold_results]),
        "mape_mean": np.mean([f["mape"] for f in fold_results]),
        "overall_accuracy": accuracy_score(all_y_true, all_y_pred),
        "overall_roc_auc": roc_auc_score(all_y_true, all_y_prob),
        "overall_mae": mean_absolute_error(all_y_true, all_y_prob),
        "overall_rmse": rmse(all_y_true, all_y_prob),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Walk-Forward Results for {ticker}")
    logger.info(f"  Accuracy:  {aggregate['accuracy_mean']:.4f} ± {aggregate['accuracy_std']:.4f}")
    logger.info(f"  F1 Score:  {aggregate['f1_mean']:.4f} ± {aggregate['f1_std']:.4f}")
    logger.info(f"  ROC-AUC:   {aggregate['roc_auc_mean']:.4f} ± {aggregate['roc_auc_std']:.4f}")
    logger.info(f"  MAE:       {aggregate['mae_mean']:.4f}")
    logger.info(f"  RMSE:      {aggregate['rmse_mean']:.4f}")
    logger.info(f"  MAPE:      {aggregate['mape_mean']:.2f}%")

    return {"folds": fold_results, "aggregate": aggregate}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    results = run_walk_forward_validation(args.ticker, args.n_splits)
