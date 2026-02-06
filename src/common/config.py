import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Stock configuration
DEFAULT_TICKER = "AAPL"
HISTORICAL_PERIOD = "2y"  # 2 years of data
REALTIME_INTERVAL = "5m"  # 5-minute intervals
REALTIME_PERIOD = "5d"    # Last 5 days

# Feature engineering
ROLLING_WINDOWS = [5, 10, 20]
VOLATILITY_WINDOW = 20

# Model configuration
MODEL_NAME = "stock_direction_predictor"
MLFLOW_TRACKING_URI = "file:" + str(BASE_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "stock_prediction"

# Training
TEST_SIZE = 0.2
RANDOM_SEED = 42
TARGET_COLUMN = "target"

# XGBoost parameters
XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss"
}

# Drift monitoring
DRIFT_THRESHOLD = 0.1
