"""Benchmark framework for relative performance measurement."""

from __future__ import annotations

import numpy as np
import yfinance as yf
from numpy.typing import NDArray

from quant_tools.backtest.metrics import (
    calculate_alpha,
    calculate_beta,
    calculate_information_ratio,
    calculate_tracking_error,
    calculate_treynor,
)
from quant_tools.models import BenchmarkDefinition, BenchmarkMetrics

STANDARD_BENCHMARKS: dict[str, BenchmarkDefinition] = {
    "SPY": BenchmarkDefinition(name="S&P 500", symbol="SPY"),
    "AGG": BenchmarkDefinition(name="US Aggregate Bond", symbol="AGG"),
    "60/40": BenchmarkDefinition(name="60/40 Portfolio", weights={"SPY": 0.6, "AGG": 0.4}),
    "QQQ": BenchmarkDefinition(name="Nasdaq 100", symbol="QQQ"),
    "IWM": BenchmarkDefinition(name="Russell 2000", symbol="IWM"),
    "XLK": BenchmarkDefinition(name="Technology Select", symbol="XLK"),
    "XLF": BenchmarkDefinition(name="Financial Select", symbol="XLF"),
    "XLE": BenchmarkDefinition(name="Energy Select", symbol="XLE"),
    "XLV": BenchmarkDefinition(name="Health Care Select", symbol="XLV"),
    "GLD": BenchmarkDefinition(name="Gold", symbol="GLD"),
}


class BenchmarkAnalyzer:
    """Analyze portfolio performance relative to benchmarks."""

    def fetch_benchmark_returns(
        self,
        benchmark: BenchmarkDefinition,
        period: str = "1y",
    ) -> NDArray[np.float64]:
        """Fetch benchmark returns for the given period.

        For single-symbol benchmarks, fetch from yfinance.
        For composite benchmarks (weights dict), fetch each component and combine.
        For custom benchmarks (returns array), return directly.

        Args:
            benchmark: Benchmark definition (symbol, weights, or custom returns).
            period: Look-back period for price data (e.g. "1y", "6mo").

        Returns:
            Array of daily returns for the benchmark.
        """
        # Custom returns array — return directly
        if benchmark.returns is not None:
            return np.asarray(benchmark.returns, dtype=np.float64)

        # Composite benchmark (weighted combination of symbols)
        if benchmark.weights:
            component_returns: dict[str, NDArray[np.float64]] = {}
            for symbol in benchmark.weights:
                component_returns[symbol] = self._fetch_symbol_returns(symbol, period)

            # Align to shortest length
            min_len = min(len(r) for r in component_returns.values())
            combined = np.zeros(min_len, dtype=np.float64)
            for symbol, weight in benchmark.weights.items():
                combined += weight * component_returns[symbol][:min_len]
            return combined

        # Single-symbol benchmark
        if benchmark.symbol:
            return self._fetch_symbol_returns(benchmark.symbol, period)

        return np.array([], dtype=np.float64)

    def _fetch_symbol_returns(
        self, symbol: str, period: str,
    ) -> NDArray[np.float64]:
        """Fetch daily returns for a single symbol from yfinance."""
        df = yf.download(symbol, period=period, progress=False)
        close = df["Close"].values.astype(np.float64).flatten()
        if len(close) < 2:
            return np.array([], dtype=np.float64)
        return np.diff(close) / close[:-1]

    def calculate_benchmark_metrics(
        self,
        portfolio_returns: NDArray[np.float64],
        benchmark_returns: NDArray[np.float64],
        benchmark_name: str = "",
        risk_free_rate: float = 0.04,
    ) -> BenchmarkMetrics:
        """Calculate all benchmark-relative metrics.

        Args:
            portfolio_returns: Array of portfolio daily returns.
            benchmark_returns: Array of benchmark daily returns.
            benchmark_name: Human-readable benchmark name.
            risk_free_rate: Annual risk-free rate (default 4%).

        Returns:
            BenchmarkMetrics with all relative performance measures.
        """
        n = min(len(portfolio_returns), len(benchmark_returns))
        rp = portfolio_returns[:n]
        rb = benchmark_returns[:n]

        beta = calculate_beta(rp, rb)
        alpha = calculate_alpha(rp, rb, risk_free_rate=risk_free_rate)
        te = calculate_tracking_error(rp, rb)
        ir = calculate_information_ratio(rp, rb)
        treynor = calculate_treynor(rp, rb, risk_free_rate=risk_free_rate)

        # Correlation
        if n >= 2:
            corr = float(np.corrcoef(rp, rb)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Active return (annualized)
        if n >= 1:
            active_return = float((np.mean(rp) - np.mean(rb)) * 252)
        else:
            active_return = 0.0

        return BenchmarkMetrics(
            benchmark_name=benchmark_name,
            beta=beta,
            alpha=alpha,
            tracking_error=te,
            information_ratio=ir,
            treynor_ratio=treynor,
            correlation=corr,
            active_return=active_return,
        )
