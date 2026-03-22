import os
from pathlib import Path
from src.common.companies import COMPANIES

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_DATA_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ======================
# Asset Configuration
# ======================
SUPPORTED_ASSETS = list(COMPANIES.values())
DEFAULT_ASSET = SUPPORTED_ASSETS[0]

HISTORICAL_PERIOD = "2y"
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

# Walk-forward validation
WF_TRAIN_PERIODS = 12
WF_TEST_PERIODS = 1

# ======================
# XGBoost Base Parameters
# Used for all tickers unless overridden in ASSET_PARAMS
# ======================
XGBOOST_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# ======================
# Per-Asset Parameter Overrides
# Only specify what changes from base XGBOOST_PARAMS
# AAPL, MSFT -> use base params (no override needed)
# ======================
ASSET_PARAMS = {
    "TSLA": {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 2},
    "NVDA": {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 2},
    "IBIT": {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 2},
    "GLD":  {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 4},
    "SPY":  {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 4},
    "QQQ":  {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 4},
}


def get_params_for_ticker(ticker: str) -> dict:
    """
    Returns merged XGBoost params for a specific ticker.
    Base params apply to all tickers.
    ASSET_PARAMS overrides only what is different per asset.

    Examples:
        AAPL -> base params
        TSLA -> base + momentum overrides (deeper tree)
        GLD  -> base + mean-reversion overrides (shallower, slower)
    """
    params = XGBOOST_PARAMS.copy()
    if ticker in ASSET_PARAMS:
        params.update(ASSET_PARAMS[ticker])
    return params


# ======================
# Drift Monitoring
# ======================
DRIFT_THRESHOLD = 0.1

# ======================
# Asset Types
# ======================
ASSET_TYPES = {
    ticker: (
        "crypto"    if ticker in ["IBIT"] else
        "commodity" if ticker in ["GLD"] else
        "index_etf" if ticker in ["SPY", "QQQ"] else
        "equity"
    )
    for ticker in SUPPORTED_ASSETS
}