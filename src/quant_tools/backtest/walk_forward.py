"""Walk-forward optimization engine.

Provides:
- WalkForwardResult: Dataclass holding IS/OOS metrics for one window.
- WalkForwardOptimizer: Runs walk-forward parameter optimization.
- aggregate_oos_metrics: Combines OOS results across all windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np

from quant_tools.backtest.metrics import calculate_sharpe


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window.

    Attributes:
        is_metrics: In-sample performance metrics.
        oos_metrics: Out-of-sample performance metrics.
        is_window: Number of in-sample observations.
        oos_window: Number of out-of-sample observations.
        optimal_params: Best parameters found during IS optimization.
        equity_curve_oos: OOS equity curve values.
    """

    is_metrics: dict[str, float] = field(default_factory=dict)
    oos_metrics: dict[str, float] = field(default_factory=dict)
    is_window: int = 0
    oos_window: int = 0
    optimal_params: dict = field(default_factory=dict)
    equity_curve_oos: list[float] = field(default_factory=list)


class WalkForwardOptimizer:
    """Walk-forward optimization engine.

    For each rolling IS/OOS window pair:
      1. Optimize strategy parameters on the IS window (maximize Sharpe).
      2. Apply the optimal parameters to the OOS window.
      3. Record both IS and OOS metrics for comparison.

    The gap between IS and OOS performance indicates overfitting risk.
    """

    def run(
        self,
        close: np.ndarray,
        strategy_fn: Callable[..., np.ndarray],
        param_grid: dict[str, list],
        is_window: int = 756,
        oos_window: int = 252,
    ) -> list[WalkForwardResult]:
        """Run walk-forward optimization.

        Args:
            close: Full array of closing prices.
            strategy_fn: Strategy function: (close, **params) -> positions array.
            param_grid: Dict of param_name -> list of values to search.
            is_window: In-sample window size (default 756 = 3 years).
            oos_window: Out-of-sample window size (default 252 = 1 year).

        Returns:
            List of WalkForwardResult, one per window.
        """
        n = len(close)
        results: list[WalkForwardResult] = []

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        if not param_names:
            return results

        param_combos = [
            dict(zip(param_names, combo)) for combo in product(*param_values)
        ]

        if not param_combos:
            return results

        start = 0
        while start + is_window + oos_window <= n:
            is_slice = close[start : start + is_window]
            oos_slice = close[start + is_window : start + is_window + oos_window]

            # Step 1: Optimize on IS window
            best_sharpe = -np.inf
            best_params: dict = param_combos[0]

            for params in param_combos:
                positions = strategy_fn(is_slice, **params)
                returns = _positions_to_returns(is_slice, positions)
                sharpe = calculate_sharpe(returns)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params

            # Step 2: Apply optimal params to IS and OOS
            is_positions = strategy_fn(is_slice, **best_params)
            is_returns = _positions_to_returns(is_slice, is_positions)
            is_sharpe = calculate_sharpe(is_returns)

            oos_positions = strategy_fn(oos_slice, **best_params)
            oos_returns = _positions_to_returns(oos_slice, oos_positions)
            oos_sharpe = calculate_sharpe(oos_returns)

            # Build OOS equity curve
            equity = [1.0]
            for r in oos_returns:
                equity.append(equity[-1] * (1.0 + r))

            # Step 3: Compute additional metrics
            is_total_return = float(np.prod(1 + is_returns) - 1) if len(is_returns) > 0 else 0.0
            oos_total_return = float(np.prod(1 + oos_returns) - 1) if len(oos_returns) > 0 else 0.0

            result = WalkForwardResult(
                is_metrics={"sharpe": is_sharpe, "total_return": is_total_return},
                oos_metrics={"sharpe": oos_sharpe, "total_return": oos_total_return},
                is_window=is_window,
                oos_window=oos_window,
                optimal_params=best_params,
                equity_curve_oos=equity,
            )
            results.append(result)

            start += oos_window

        return results


def aggregate_oos_metrics(results: list[WalkForwardResult]) -> dict:
    """Combine all OOS results into aggregate performance metrics.

    Args:
        results: List of WalkForwardResult from a walk-forward run.

    Returns:
        Dictionary with aggregated OOS metrics:
            mean_sharpe, median_sharpe, std_sharpe,
            mean_return, total_return, n_windows,
            is_oos_sharpe_gap (avg IS Sharpe - avg OOS Sharpe).
    """
    if not results:
        return {
            "mean_sharpe": 0.0,
            "median_sharpe": 0.0,
            "std_sharpe": 0.0,
            "mean_return": 0.0,
            "total_return": 0.0,
            "n_windows": 0,
            "is_oos_sharpe_gap": 0.0,
        }

    oos_sharpes = [r.oos_metrics.get("sharpe", 0.0) for r in results]
    is_sharpes = [r.is_metrics.get("sharpe", 0.0) for r in results]
    oos_returns = [r.oos_metrics.get("total_return", 0.0) for r in results]

    # Compound total return across all OOS windows
    total_return = float(np.prod([1 + r for r in oos_returns]) - 1)

    return {
        "mean_sharpe": float(np.mean(oos_sharpes)),
        "median_sharpe": float(np.median(oos_sharpes)),
        "std_sharpe": float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0,
        "mean_return": float(np.mean(oos_returns)),
        "total_return": total_return,
        "n_windows": len(results),
        "is_oos_sharpe_gap": float(np.mean(is_sharpes) - np.mean(oos_sharpes)),
    }


def _positions_to_returns(
    close: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Convert close prices and positions to strategy returns.

    Args:
        close: Array of closing prices.
        positions: Array of positions (-1, 0, 1).

    Returns:
        Array of strategy returns (length n-1).
    """
    if len(close) < 2:
        return np.array([], dtype=np.float64)
    daily_returns = np.diff(close) / close[:-1]
    return positions[:-1] * daily_returns
