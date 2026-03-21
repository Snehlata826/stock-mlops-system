import os
from pathlib import Path
from src.common.companies import COMPANIES


# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ======================
# Asset Configuration
# ======================
# ======================
# Asset Configuration
# ======================

# Single source of truth comes from companies.py
SUPPORTED_ASSETS = list(COMPANIES.values())

# Default asset = first entry (safe fallback)
DEFAULT_ASSET = SUPPORTED_ASSETS[0]


HISTORICAL_PERIOD = "10y"
REALTIME_INTERVAL = "5m"
REALTIME_PERIOD = "5d"

# ======================
# Feature Engineering
# ======================
ROLLING_WINDOWS = [5, 10, 20]
VOLATILITY_WINDOW = 20

# ======================
# Model Configuration
# ======================
MODEL_NAME = "stock_direction_predictor"
MLFLOW_TRACKING_URI = "file:" + str(BASE_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "stock_prediction"

# ======================
# Training
# ======================
TEST_SIZE = 0.2
RANDOM_SEED = 42
TARGET_COLUMN = "target"

# ======================
# XGBoost Parameters
# ======================
XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss"
}

# ======================
# Drift Monitoring
# ======================
DRIFT_THRESHOLD = 0.1

# ======================
# Asset Types (Optional)
# ======================
ASSET_TYPES = {
    ticker: "commodity" if ticker in ["GLD"] else "equity"
    for ticker in SUPPORTED_ASSETS
}
