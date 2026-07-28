"""Pure functions for backtest performance metrics (numpy-based)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def calculate_total_return(equity_curve: NDArray[np.float64]) -> float:
    """Total return as a fraction (e.g., 0.15 = 15%).

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Total return fraction. 0.0 for degenerate inputs.
    """
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return float((equity_curve[-1] / equity_curve[0]) - 1.0)


def calculate_cagr(
    equity_curve: NDArray[np.float64], periods_per_year: int = 252
) -> float:
    """Compound Annual Growth Rate.

    Args:
        equity_curve: Array of portfolio values over time.
        periods_per_year: Trading periods per year (default 252).

    Returns:
        CAGR as a fraction. 0.0 for degenerate inputs.
    """
    n = len(equity_curve)
    if n < 2 or equity_curve[0] <= 0 or equity_curve[-1] <= 0:
        return 0.0

    years = (n - 1) / periods_per_year
    if years <= 0:
        return 0.0

    return float((equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0)


def calculate_sharpe(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe Ratio.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate (default 4%).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Sharpe ratio. 0.0 if std is zero or input is degenerate.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    std = np.std(excess, ddof=1)

    if std < 1e-12 or np.isnan(std):
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def calculate_sortino(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino Ratio (downside deviation only).

    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate (default 4%).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Sortino ratio. 0.0 if downside deviation is zero or input is degenerate.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    downside = excess[excess < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def calculate_max_drawdown(
    equity_curve: NDArray[np.float64],
) -> tuple[float, int]:
    """Maximum drawdown and its duration in periods.

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Tuple of (max_drawdown_fraction, duration_in_periods).
        Drawdown is positive (e.g., 0.20 = 20% loss from peak).
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    running_max = np.maximum.accumulate(equity_curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(
            running_max > 0,
            (running_max - equity_curve) / running_max,
            0.0,
        )

    max_dd = float(np.max(drawdowns))

    # Calculate duration of the worst drawdown
    peak_idx = 0
    max_duration = 0
    in_drawdown = False
    for i in range(1, len(equity_curve)):
        if equity_curve[i] < running_max[i]:
            # In drawdown
            if not in_drawdown:
                in_drawdown = True
            duration = i - peak_idx
            if duration > max_duration:
                max_duration = duration
        else:
            # New high or recovered
            in_drawdown = False
            peak_idx = i

    return max_dd, max_duration


def calculate_win_rate(trade_returns: NDArray[np.float64]) -> float:
    """Fraction of trades that were profitable.

    Args:
        trade_returns: Array of per-trade returns.

    Returns:
        Win rate as a fraction (0.0 to 1.0). 0.0 for empty input.
    """
    if len(trade_returns) == 0:
        return 0.0
    wins = np.sum(trade_returns > 0)
    return float(wins / len(trade_returns))


def calculate_profit_factor(trade_returns: NDArray[np.float64]) -> float:
    """Ratio of gross profit to gross loss.

    Args:
        trade_returns: Array of per-trade returns.

    Returns:
        Profit factor. float('inf') if no losses, 0.0 if no profits or empty.
    """
    if len(trade_returns) == 0:
        return 0.0

    gross_profit = float(np.sum(trade_returns[trade_returns > 0]))
    gross_loss = float(np.abs(np.sum(trade_returns[trade_returns < 0])))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


# ── Benchmark-Relative Metrics ──────────────────────────────────────


def calculate_beta(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
) -> float:
    """Market beta: sensitivity of portfolio to benchmark moves.

    Args:
        portfolio_returns: Array of portfolio period returns.
        benchmark_returns: Array of benchmark period returns (same length).

    Returns:
        Beta coefficient. 0.0 for degenerate inputs.
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    rp, rb = portfolio_returns[:n], benchmark_returns[:n]
    var_b = np.var(rb, ddof=1)
    if var_b < 1e-15:
        return 0.0
    cov_pb = np.cov(rp, rb, ddof=1)[0, 1]
    return float(cov_pb / var_b)


def calculate_alpha(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """Jensen's alpha: annualized excess return above CAPM prediction.

    Args:
        portfolio_returns: Array of portfolio period returns.
        benchmark_returns: Array of benchmark period returns.
        risk_free_rate: Annual risk-free rate (default 4%).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized alpha. 0.0 for degenerate inputs.
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    rp, rb = portfolio_returns[:n], benchmark_returns[:n]
    daily_rf = risk_free_rate / periods_per_year
    beta = calculate_beta(rp, rb)
    # Jensen's alpha (annualized): (mean(rp) - rf_daily - beta*(mean(rb) - rf_daily)) * periods
    alpha_daily = np.mean(rp) - daily_rf - beta * (np.mean(rb) - daily_rf)
    return float(alpha_daily * periods_per_year)


def calculate_tracking_error(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualized tracking error: volatility of active returns.

    Args:
        portfolio_returns: Array of portfolio period returns.
        benchmark_returns: Array of benchmark period returns.
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized tracking error. 0.0 for degenerate inputs.
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    active = portfolio_returns[:n] - benchmark_returns[:n]
    std = np.std(active, ddof=1)
    if np.isnan(std):
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def calculate_information_ratio(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Information ratio: active return per unit of tracking error.

    Args:
        portfolio_returns: Array of portfolio period returns.
        benchmark_returns: Array of benchmark period returns.
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Annualized information ratio. 0.0 for degenerate inputs.
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    active = portfolio_returns[:n] - benchmark_returns[:n]
    std = np.std(active, ddof=1)
    if std < 1e-12 or np.isnan(std):
        return 0.0
    return float(np.mean(active) / std * np.sqrt(periods_per_year))


def calculate_treynor(
    portfolio_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> float:
    """Treynor ratio: excess return per unit of systematic risk (beta).

    Args:
        portfolio_returns: Array of portfolio period returns.
        benchmark_returns: Array of benchmark period returns.
        risk_free_rate: Annual risk-free rate (default 4%).
        periods_per_year: Trading periods per year (default 252).

    Returns:
        Treynor ratio. 0.0 for degenerate inputs or zero beta.
    """
    if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    rp, rb = portfolio_returns[:n], benchmark_returns[:n]
    beta = calculate_beta(rp, rb)
    if abs(beta) < 1e-12:
        return 0.0
    ann_return = float(np.mean(rp) * periods_per_year)
    return float((ann_return - risk_free_rate) / beta)
