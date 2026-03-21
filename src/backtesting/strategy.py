import numpy as np
import pandas as pd


def generate_signals(df, prob_col="bullish_prob", threshold=0.55):
    """
    Generate trading signals from probability predictions.
    """
    df = df.copy()

    df["signal"] = 0
    df.loc[df[prob_col] > threshold, "signal"] = 1
    df.loc[df[prob_col] < (1 - threshold), "signal"] = -1

    return df


def backtest_strategy(df,
                      price_col="Close",
                      signal_col="signal",
                      transaction_cost=0.001):
    """
    Simulate trading strategy performance.
    """

    df = df.copy()

    df["return"] = df[price_col].pct_change()

    # shift to prevent look-ahead bias
    df["strategy_return"] = df[signal_col].shift(1) * df["return"]

    # transaction cost
    df["trade"] = df[signal_col].diff().abs()
    df["strategy_return"] -= df["trade"] * transaction_cost

    df["strategy_cum"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_cum"] = (1 + df["return"]).cumprod()

    return df