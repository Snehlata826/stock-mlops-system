import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, mean_absolute_error
)
import mlflow

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    PROCESSED_DATA_DIR, TARGET_COLUMN, BASE_DIR,
    MLFLOW_TRACKING_URI, MODEL_NAME
)
from src.common.utils import logger, validate_asset


def evaluate_model(ticker: str, run_id: str = None):
    validate_asset(ticker)
    logger.info(f"Evaluating model for {ticker}...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    X = test_df[feature_cols]
    y = test_df[TARGET_COLUMN]

    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        model_uri = f"models:/{MODEL_NAME}_{ticker}/latest"

    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Classification metrics
    logger.info("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Down", "Up"]))

    # Regression-style metrics on probabilities
    mae = mean_absolute_error(y, y_pred_proba)
    rmse = float(np.sqrt(np.mean((y.values - y_pred_proba) ** 2)))
    mape = float(np.mean(np.abs((y.values - y_pred_proba) / (y.values + 1e-9))) * 100)

    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")

    eval_dir = BASE_DIR / "evaluation" / ticker
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix ({ticker})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"]); ax.set_yticklabels(["Down", "Up"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="#2196F3")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({ticker})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curve.png", dpi=120)
    plt.close()

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#4CAF50")
    ax.set_xlabel("Importance"); ax.set_title(f"Top 15 Feature Importances ({ticker})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(eval_dir / "feature_importance.png", dpi=120)
    plt.close()

    logger.success(f"Evaluation complete → {eval_dir}")
    return {"mae": mae, "rmse": rmse, "mape": mape, "roc_auc": roc_auc}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    evaluate_model(ticker=args.ticker, run_id=args.run_id)
