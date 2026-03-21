import numpy as np


def sharpe_ratio(returns, periods_per_year=252):
    return np.sqrt(periods_per_year) * (
        returns.mean() / returns.std()
    )


def max_drawdown(cumulative_returns):
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    return drawdown.min()


def win_rate(returns):
    return (returns > 0).mean()