import pandas as pd
import numpy as np
from pathlib import Path
import sys

import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---- Add project root to path ----
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    XGBOOST_PARAMS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    DRIFT_DATA_DIR
)

from src.common.utils import logger, validate_dataframe, validate_asset
from src.backtesting.walk_forward import walk_forward_validation
from src.backtesting.strategy import generate_signals, backtest_strategy
from src.backtesting.metrics import sharpe_ratio, max_drawdown, win_rate


# ---------------------------------------------------
# Data Preparation
# ---------------------------------------------------

def prepare_data(df: pd.DataFrame):
    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    return X, y, feature_cols


# ---------------------------------------------------
# Training Pipeline
# ---------------------------------------------------

def train_model(ticker: str):

    validate_asset(ticker)

    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    logger.info(f"Starting training for {ticker}")

    df = pd.read_csv(data_path)

    # 🔥 Ensure time order
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    validate_dataframe(df, required_columns=[TARGET_COLUMN])

    X, y, feature_cols = prepare_data(df)

    logger.info(f"Dataset size: {len(df)}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # ---------------------------------------------------
    # Time-Based Split
    # ---------------------------------------------------

    split_index = int(len(X) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Save reference features for drift
    reference_df = X_train.copy()
    reference_df[TARGET_COLUMN] = y_train

    reference_path = DRIFT_DATA_DIR / f"reference_features_{ticker}.csv"
    reference_df.to_csv(reference_path, index=False)

    # ---------------------------------------------------
    # MLflow Setup
    # ---------------------------------------------------

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model_name = f"{MODEL_NAME}_{ticker}"

    with mlflow.start_run(run_name=f"xgboost_training_{ticker}") as run:

        logger.info(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("dataset_size", len(df))

        # ---------------------------------------------------
        # Train Model
        # ---------------------------------------------------

        model = xgb.XGBClassifier(**XGBOOST_PARAMS)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # ---------------------------------------------------
        # Classification Metrics
        # ---------------------------------------------------

        metrics = {}

        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_test, y_pred, zero_division=0)

        if len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        else:
            metrics["roc_auc"] = np.nan

        for k, v in metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(k, v)
            logger.info(f"{k}: {v:.4f}")

        # ---------------------------------------------------
        # Walk-Forward Validation
        # ---------------------------------------------------

        logger.info("Running walk-forward validation...")

        wf_results = walk_forward_validation(
            model=xgb.XGBClassifier(**XGBOOST_PARAMS),
            df=pd.concat([X, y], axis=1),
            feature_cols=feature_cols,
            target_col=TARGET_COLUMN,
            train_window=250,
            test_window=25
        )

        if not wf_results.empty:

            wf_mean_acc = wf_results["accuracy"].mean()
            wf_mean_auc = wf_results["roc_auc"].mean(skipna=True)

            mlflow.log_metric("wf_mean_accuracy", wf_mean_acc)

            if not np.isnan(wf_mean_auc):
                mlflow.log_metric("wf_mean_roc_auc", wf_mean_auc)

            logger.info(f"Walk-forward Mean Accuracy: {wf_mean_acc:.4f}")
            logger.info(f"Walk-forward Mean ROC-AUC: {wf_mean_auc:.4f}")

            wf_path = DRIFT_DATA_DIR / f"walk_forward_{ticker}.csv"
            wf_results.to_csv(wf_path, index=False)

        else:
            logger.warning("Walk-forward returned no windows.")

        # ---------------------------------------------------
        # Strategy Backtesting
        # ---------------------------------------------------

        logger.info("Running strategy backtest...")

        bt_df = X_test.copy()
        bt_df["Close"] = df.iloc[split_index:]["Close"].values
        bt_df["bullish_prob"] = y_pred_proba

        bt_df = generate_signals(bt_df, prob_col="bullish_prob", threshold=0.55)
        bt_df = backtest_strategy(bt_df, price_col="Close")

        strategy_return = bt_df["strategy_return"].dropna()

        if len(strategy_return) > 5:

            strategy_cum = bt_df["strategy_cum"]

            sr = sharpe_ratio(strategy_return)
            mdd = max_drawdown(strategy_cum)
            wr = win_rate(strategy_return)

            if not np.isnan(sr):
                mlflow.log_metric("strategy_sharpe", sr)

            mlflow.log_metric("strategy_max_drawdown", mdd)
            mlflow.log_metric("strategy_win_rate", wr)

            logger.info(f"Strategy Sharpe: {sr:.4f}")
            logger.info(f"Max Drawdown: {mdd:.4f}")
            logger.info(f"Win Rate: {wr:.4f}")

        else:
            logger.warning("Not enough data for strategy metrics.")

        bt_path = DRIFT_DATA_DIR / f"backtest_results_{ticker}.csv"
        bt_df.to_csv(bt_path, index=False)

        # ---------------------------------------------------
        # Register Model
        # ---------------------------------------------------

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )

        mlflow.log_dict({"features": feature_cols}, "features.json")

        logger.info(f"✓ Training complete for {ticker}")
        logger.info(f"✓ Registered model: {model_name}")

        return run.info.run_id, metrics


# ---------------------------------------------------
# CLI Entry
# ---------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    train_model(args.ticker)