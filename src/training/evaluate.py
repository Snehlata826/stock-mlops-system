import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
import mlflow

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import (
    PROCESSED_DATA_DIR, TARGET_COLUMN, BASE_DIR,
    MLFLOW_TRACKING_URI, MODEL_NAME
)
from src.common.utils import logger, validate_asset


def evaluate_model(ticker: str, run_id: str = None):
    """
    Evaluate model performance for a specific asset

    Args:
        ticker: Asset ticker (AAPL, MSFT, TSLA, GLD)
        run_id: Optional MLflow run ID
    """
    logger.info(f"Evaluating model performance for {ticker}...")

    # ✅ Validate asset
    validate_asset(ticker)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load training data for asset
    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)

    # Prepare data
    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    # Load model
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    else:
        model_uri = f"models:/{MODEL_NAME}_{ticker}/latest"

    logger.info(f"  Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Classification report
    logger.info("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Down", "Up"]))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    # Output directory
    eval_dir = BASE_DIR / "evaluation" / ticker
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Confusion Matrix Plot ----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix ({ticker})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrix.png")
    plt.close()
    logger.info(f"  Saved confusion matrix to {eval_dir / 'confusion_matrix.png'}")

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({ticker})")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curve.png")
    plt.close()
    logger.info(f"  Saved ROC curve to {eval_dir / 'roc_curve.png'}")

    # ---------------- Feature Importance ----------------
    feature_importance = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance["feature"], feature_importance["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 15 Feature Importances ({ticker})")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(eval_dir / "feature_importance.png")
    plt.close()
    logger.info(f"  Saved feature importance to {eval_dir / 'feature_importance.png'}")

    logger.info("✓ Evaluation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--ticker", required=True, help="Asset ticker")
    parser.add_argument("--run-id", type=str, default=None)

    args = parser.parse_args()

    evaluate_model(ticker=args.ticker, run_id=args.run_id)
