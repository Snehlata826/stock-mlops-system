import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import (
    PROCESSED_DATA_DIR, TARGET_COLUMN, TEST_SIZE, RANDOM_SEED,
    XGBOOST_PARAMS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME, BASE_DIR, DRIFT_DATA_DIR
)
from src.common.utils import logger, validate_dataframe

def prepare_data(df: pd.DataFrame):
    """Prepare features and target"""
    # Exclude non-feature columns
    exclude_cols = ['Date', TARGET_COLUMN, 'Dividends', 'Stock Splits']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    return X, y, feature_cols

def train_model(data_path: Path = PROCESSED_DATA_DIR / "features_train.csv"):
    """
    Train XGBoost model with MLflow tracking
    """
    logger.info(f"Starting model training from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    validate_dataframe(df, required_columns=[TARGET_COLUMN])
    logger.info(f"  Loaded {len(df)} records")
    
    # Prepare data
    X, y, feature_cols = prepare_data(df)
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Save reference data for drift monitoring
    reference_data = X_train.copy()
    reference_data[TARGET_COLUMN] = y_train
    reference_path = DRIFT_DATA_DIR / "reference_data.csv"
    reference_data.to_csv(reference_path, index=False)
    logger.info(f"  Saved reference data to {reference_path}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Start MLflow run
    with mlflow.start_run(run_name="xgboost_training") as run:
        logger.info(f"  MLflow run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", len(feature_cols))
        
        # Train model
        logger.info("  Training XGBoost model...")
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        
        # Save feature names
        mlflow.log_dict({"features": feature_cols}, "features.json")
        
        logger.info(f"✓ Model training complete")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return run.info.run_id, metrics

if __name__ == "__main__":
    train_model()
