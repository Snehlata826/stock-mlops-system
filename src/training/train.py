import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    PROCESSED_DATA_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_SEED,
    XGBOOST_PARAMS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME, DRIFT_DATA_DIR, get_params_for_ticker
)
from src.common.utils import logger, validate_dataframe, validate_asset


def prepare_data(df: pd.DataFrame):
    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y, feature_cols


def train_model(ticker: str):
    validate_asset(ticker)

    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    logger.info(f"Starting training for {ticker}")
    df = pd.read_csv(data_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    validate_dataframe(df, required_columns=[TARGET_COLUMN])
    X, y, feature_cols = prepare_data(df)

    # Time-aware split (no shuffle!) — last TEST_SIZE% is test
    split_idx = int(len(df) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Save reference features for drift monitoring
    reference_df = X_train.copy()
    reference_df[TARGET_COLUMN] = y_train.values
    DRIFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = DRIFT_DATA_DIR / f"reference_features_{ticker}.csv"
    reference_df.to_csv(reference_path, index=False)
    logger.info(f"Saved reference features → {reference_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_name = f"{MODEL_NAME}_{ticker}"

    from src.common.config import get_params_for_ticker
    params = {k: v for k, v in get_params_for_ticker(ticker).items() if k != "use_label_encoder"}

    with mlflow.start_run(run_name=f"xgboost_training_{ticker}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_params(params)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("split_strategy", "time_ordered_no_shuffle")

        model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "mae": float(np.mean(np.abs(y_test.values - y_pred_proba))),
            "rmse": float(np.sqrt(np.mean((y_test.values - y_pred_proba) ** 2))),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            logger.info(f"  {k}: {v:.4f}")

        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=model_name)
        mlflow.log_dict({"features": feature_cols}, "features.json")

        logger.success(f"Training complete for {ticker} | Model: {model_name}")
        return run.info.run_id, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    train_model(args.ticker)
