"""Portfolio analysis functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from quant_tools.data.market_data import fetch_prices, fetch_ticker_info
from quant_tools.errors import ComputeError, DataError
from quant_tools.models import (
    BenchmarkDefinition,
    BenchmarkMetrics,
    PortfolioAnalysis,
    PortfolioHolding,
    PortfolioOptimization,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger("quant_tools")


def calculate_correlation_matrix(
    returns_dict: dict[str, NDArray],
) -> dict[str, dict[str, float]]:
    """Calculate pairwise correlation matrix from a dict of return arrays.

    Args:
        returns_dict: Mapping of symbol to daily return arrays (same length).

    Returns:
        Nested dict: ``{symbol_a: {symbol_b: correlation}}``.
    """
    symbols = list(returns_dict.keys())
    n = len(symbols)
    matrix: dict[str, dict[str, float]] = {}

    for i in range(n):
        row: dict[str, float] = {}
        for j in range(n):
            a = returns_dict[symbols[i]]
            b = returns_dict[symbols[j]]
            min_len = min(len(a), len(b))
            if min_len < 2:
                row[symbols[j]] = 0.0
            else:
                corr = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
                row[symbols[j]] = corr if np.isfinite(corr) else 0.0
        matrix[symbols[i]] = row

    return matrix


def calculate_sector_exposure(
    holdings: dict[str, float],
) -> dict[str, float]:
    """Calculate portfolio sector exposure by weight.

    Fetches ticker info for each holding to determine sector.

    Args:
        holdings: Mapping of symbol to portfolio weight.

    Returns:
        Mapping of sector name to aggregate weight.
    """
    sector_weights: dict[str, float] = {}

    for symbol, weight in holdings.items():
        try:
            info = fetch_ticker_info(symbol)
            sector = info.get("sector", "Unknown") or "Unknown"
        except (DataError, Exception):
            log.debug("Could not fetch sector for %s, using Unknown", symbol)
            sector = "Unknown"

        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    return sector_weights


def calculate_portfolio_metrics(
    weights: NDArray,
    returns_matrix: NDArray,
    risk_free_rate: float,
) -> dict:
    """Calculate portfolio-level metrics from weights and a returns matrix.

    Args:
        weights: 1-D array of portfolio weights (sums to 1).
        returns_matrix: 2-D array, shape (n_days, n_assets).
        risk_free_rate: Annualized risk-free rate (e.g. 0.04).

    Returns:
        Dict with keys: total_return, annualized_return, volatility,
        sharpe_ratio, max_drawdown.
    """
    # Weighted daily portfolio returns
    portfolio_returns = returns_matrix @ weights
    n_days = len(portfolio_returns)

    if n_days < 2:
        raise ComputeError("Not enough data to compute portfolio metrics")

    # Cumulative return
    cumulative = np.cumprod(1 + portfolio_returns)
    total_return = float(cumulative[-1] - 1)

    # Annualized return (252 trading days)
    annualized_return = float((1 + total_return) ** (252 / n_days) - 1)

    # Annualized volatility
    daily_vol = float(np.std(portfolio_returns, ddof=1))
    volatility = daily_vol * np.sqrt(252)

    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = portfolio_returns - daily_rf
    if daily_vol > 0:
        sharpe_ratio = float(np.mean(excess_returns)) / daily_vol * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "portfolio_returns": portfolio_returns,
        "cumulative": cumulative,
    }


def analyze_portfolio(
    holdings: dict[str, float],
    period: str = "1y",
    risk_free_rate: float = 0.04,
) -> PortfolioAnalysis:
    """Full portfolio analysis.

    Fetches prices for each holding, computes returns, correlation matrix,
    sector exposure, and portfolio-level metrics.

    Args:
        holdings: Mapping of symbol to target weight (should sum to 1.0).
        period: Look-back period for price data (e.g. "1y", "6mo").
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.

    Returns:
        PortfolioAnalysis dataclass with all results.

    Raises:
        DataError: If prices cannot be fetched for any holding.
        ComputeError: If there is insufficient data.
    """
    symbols = list(holdings.keys())
    weights_list = [holdings[s] for s in symbols]
    weights_arr = np.array(weights_list)

    # Fetch prices and compute daily returns for each holding
    price_data: dict[str, np.ndarray] = {}
    returns_dict: dict[str, np.ndarray] = {}

    for symbol in symbols:
        df = fetch_prices(symbol, period=period)
        close = df["Close"].values.astype(float)
        if len(close) < 2:
            raise DataError(f"Not enough price data for {symbol}")
        daily_ret = np.diff(close) / close[:-1]
        price_data[symbol] = close
        returns_dict[symbol] = daily_ret

    # Align return arrays to shortest length
    min_len = min(len(r) for r in returns_dict.values())
    for symbol in symbols:
        returns_dict[symbol] = returns_dict[symbol][:min_len]

    # Build returns matrix (n_days x n_assets)
    returns_matrix = np.column_stack([returns_dict[s] for s in symbols])

    # Portfolio-level metrics
    metrics = calculate_portfolio_metrics(weights_arr, returns_matrix, risk_free_rate)

    # Per-holding metrics
    holding_list: list[PortfolioHolding] = []
    for i, symbol in enumerate(symbols):
        asset_returns = returns_dict[symbol]
        asset_cum = np.cumprod(1 + asset_returns)
        asset_total = float(asset_cum[-1] - 1)
        asset_ann = float((1 + asset_total) ** (252 / len(asset_returns)) - 1)
        asset_vol = float(np.std(asset_returns, ddof=1) * np.sqrt(252))

        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        asset_excess = asset_returns - daily_rf
        daily_vol = float(np.std(asset_returns, ddof=1))
        asset_sharpe = (
            float(np.mean(asset_excess)) / daily_vol * np.sqrt(252)
            if daily_vol > 0
            else 0.0
        )

        holding_list.append(
            PortfolioHolding(
                symbol=symbol,
                weight=weights_list[i],
                current_price=float(price_data[symbol][-1]),
                total_return=asset_total,
                annualized_return=asset_ann,
                volatility=asset_vol,
                sharpe=asset_sharpe,
            )
        )

    # Correlation matrix
    corr_matrix = calculate_correlation_matrix(returns_dict)

    # Sector exposure
    sector_exposure = calculate_sector_exposure(holdings)

    return PortfolioAnalysis(
        holdings=holding_list,
        total_return=metrics["total_return"],
        annualized_return=metrics["annualized_return"],
        volatility=metrics["volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        correlation_matrix=corr_matrix,
        sector_exposure=sector_exposure,
    )


def optimize_portfolio(
    holdings: dict[str, float],
    period: str = "1y",
    method: str = "markowitz",
    risk_free_rate: float = 0.04,
) -> PortfolioOptimization:
    """Convenience function to optimize a portfolio.

    Fetches historical prices for each symbol, builds a returns DataFrame,
    and runs the specified optimization method.

    Args:
        holdings: Dict of symbol -> weight (used to determine which assets to include).
        period: Historical data period for price fetching.
        method: Optimization method (markowitz, risk_parity, min_variance).
        risk_free_rate: Annual risk-free rate.

    Returns:
        PortfolioOptimization with optimal weights.

    Raises:
        DataError: If prices cannot be fetched for any symbol.
    """
    import pandas as pd

    from quant_tools.portfolio.optimizer import PortfolioOptimizer

    symbols = list(holdings.keys())
    returns_dict: dict[str, np.ndarray] = {}

    for symbol in symbols:
        df = fetch_prices(symbol, period=period)
        close = df["Close"].values.astype(float)
        if len(close) < 2:
            raise DataError(f"Not enough price data for {symbol}")
        returns_dict[symbol] = np.diff(close) / close[:-1]

    # Align to shortest length and build DataFrame
    min_len = min(len(r) for r in returns_dict.values())
    returns_df = pd.DataFrame({s: r[:min_len] for s, r in returns_dict.items()})

    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
    return optimizer.optimize(returns_df, method=method)


def analyze_portfolio_with_benchmark(
    holdings: dict[str, float],
    benchmark: BenchmarkDefinition | str,
    period: str = "1y",
    risk_free_rate: float = 0.04,
) -> tuple[PortfolioAnalysis, BenchmarkMetrics]:
    """Analyze portfolio and compute benchmark-relative metrics.

    Args:
        holdings: Mapping of symbol to target weight (should sum to 1.0).
        benchmark: A BenchmarkDefinition, or a string key into STANDARD_BENCHMARKS.
        period: Look-back period for price data (e.g. "1y", "6mo").
        risk_free_rate: Annualized risk-free rate for metric calculations.

    Returns:
        Tuple of (PortfolioAnalysis, BenchmarkMetrics).

    Raises:
        DataError: If prices cannot be fetched.
        ComputeError: If there is insufficient data.
        ValueError: If string benchmark key is not found.
    """
    from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS, BenchmarkAnalyzer

    # Resolve string to BenchmarkDefinition
    if isinstance(benchmark, str):
        if benchmark not in STANDARD_BENCHMARKS:
            msg = f"Unknown benchmark '{benchmark}'. Available: {list(STANDARD_BENCHMARKS)}"
            raise ValueError(msg)
        bench_def = STANDARD_BENCHMARKS[benchmark]
    else:
        bench_def = benchmark

    # Run standard portfolio analysis
    analysis = analyze_portfolio(holdings, period=period, risk_free_rate=risk_free_rate)

    # Compute portfolio daily returns for benchmark comparison
    symbols = list(holdings.keys())
    returns_dict: dict[str, np.ndarray] = {}
    for symbol in symbols:
        df = fetch_prices(symbol, period=period)
        close = df["Close"].values.astype(float)
        returns_dict[symbol] = np.diff(close) / close[:-1]

    min_len = min(len(r) for r in returns_dict.values())
    weights_arr = np.array([holdings[s] for s in symbols])
    returns_matrix = np.column_stack([returns_dict[s][:min_len] for s in symbols])
    portfolio_returns = returns_matrix @ weights_arr

    # Fetch benchmark returns and compute metrics
    analyzer = BenchmarkAnalyzer()
    bench_returns = analyzer.fetch_benchmark_returns(bench_def, period=period)
    bench_metrics = analyzer.calculate_benchmark_metrics(
        portfolio_returns,
        bench_returns,
        benchmark_name=bench_def.name,
        risk_free_rate=risk_free_rate,
    )

    return analysis, bench_metrics
