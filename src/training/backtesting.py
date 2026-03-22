"""
Backtesting module: simulate trading strategy based on model predictions.
Computes cumulative return, Sharpe ratio, max drawdown, win rate.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import PROCESSED_DATA_DIR, TARGET_COLUMN, MODEL_NAME
from src.common.utils import logger, validate_asset
from src.inference.model_loader import ModelLoader


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def compute_max_drawdown(cumulative_returns: np.ndarray) -> float:
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1e-9)
    return float(drawdown.min())


def run_backtest(ticker: str, strategy: str = "long_only") -> dict:
    """
    Simulate a trading strategy using model predictions on held-out data.

    Strategies:
    - long_only: buy when model predicts UP, hold cash otherwise
    - long_short: buy when UP, short when DOWN
    - buy_and_hold: baseline — always long

    Returns metrics dict + trade log DataFrame.
    """
    validate_asset(ticker)

    data_path = PROCESSED_DATA_DIR / f"features_train_{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # Use last 20% as backtest period (never seen during training)
    n = len(df)
    test_start = int(n * 0.8)
    backtest_df = df.iloc[test_start:].copy().reset_index(drop=True)

    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    feature_cols = [c for c in backtest_df.columns if c not in exclude_cols]
    X = backtest_df[feature_cols]

    loader = ModelLoader()
    try:
        model = loader.get_model(f"{MODEL_NAME}_{ticker}")
    except Exception as e:
        logger.error(f"Could not load model for {ticker}: {e}")
        return {"error": str(e)}

    probs = model.predict_proba(X)[:, 1]
    predictions = (probs >= 0.5).astype(int)

    # Compute actual returns from Close prices
    close = backtest_df["Close"].values
    actual_returns = np.diff(close) / close[:-1]  # daily return

    # Align predictions with returns (predict at t, return at t+1)
    preds = predictions[:-1]
    probs_aligned = probs[:-1]

    # Strategy signals
    if strategy == "long_only":
        signals = np.where(preds == 1, 1.0, 0.0)
        strategy_label = "Long only (cash when bearish)"
    elif strategy == "long_short":
        signals = np.where(preds == 1, 1.0, -1.0)
        strategy_label = "Long/Short"
    else:
        signals = np.ones(len(actual_returns))
        strategy_label = "Buy and Hold"

    strategy_returns = signals * actual_returns
    bah_returns = actual_returns  # buy-and-hold baseline

    # Cumulative returns
    cum_strategy = np.cumprod(1 + strategy_returns)
    cum_bah = np.cumprod(1 + bah_returns)

    # Metrics
    n_trades = int(np.sum(np.abs(np.diff(np.concatenate([[0], signals])))))
    win_trades = int(np.sum((strategy_returns > 0) & (signals != 0)))
    total_trades = int(np.sum(signals != 0))
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0

    metrics = {
        "ticker": ticker,
        "strategy": strategy_label,
        "n_days": len(actual_returns),
        "total_return_strategy": float(cum_strategy[-1] - 1),
        "total_return_bah": float(cum_bah[-1] - 1),
        "sharpe_strategy": compute_sharpe_ratio(strategy_returns),
        "sharpe_bah": compute_sharpe_ratio(bah_returns),
        "max_drawdown_strategy": compute_max_drawdown(cum_strategy),
        "max_drawdown_bah": compute_max_drawdown(cum_bah),
        "win_rate": win_rate,
        "n_trades": total_trades,
        "alpha": float(cum_strategy[-1] - cum_bah[-1]),
    }

    # Trade log
    dates = backtest_df["Date"].values[:-1] if "Date" in backtest_df.columns else np.arange(len(actual_returns))
    trade_log = pd.DataFrame({
        "date": dates,
        "signal": signals,
        "prob_up": probs_aligned,
        "actual_return": actual_returns,
        "strategy_return": strategy_returns,
        "cum_strategy": cum_strategy,
        "cum_bah": cum_bah,
    })

    logger.info(f"\nBacktest Results [{ticker} | {strategy_label}]")
    logger.info(f"  Strategy return:  {metrics['total_return_strategy']:.2%}")
    logger.info(f"  Buy-and-hold:     {metrics['total_return_bah']:.2%}")
    logger.info(f"  Alpha:            {metrics['alpha']:.2%}")
    logger.info(f"  Sharpe (strat):   {metrics['sharpe_strategy']:.3f}")
    logger.info(f"  Max drawdown:     {metrics['max_drawdown_strategy']:.2%}")
    logger.info(f"  Win rate:         {metrics['win_rate']:.2%}")
    logger.info(f"  N trades:         {metrics['n_trades']}")

    return {"metrics": metrics, "trade_log": trade_log}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--strategy", default="long_only", choices=["long_only", "long_short", "buy_and_hold"])
    args = parser.parse_args()
    result = run_backtest(args.ticker, args.strategy)
    print(result["metrics"])
