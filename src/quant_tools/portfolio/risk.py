"""Portfolio risk analysis: VaR, CVaR, Kelly criterion."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_tools.models import RiskMetrics

log = logging.getLogger("quant_tools")


class RiskAnalyzer:
    """Calculate portfolio risk metrics."""

    def calculate_var(self, returns: np.ndarray | pd.Series, confidence: float = 0.95) -> float:
        """Historical Value at Risk.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            VaR as a negative number (loss).
        """
        if len(returns) < 10:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    def calculate_cvar(self, returns: np.ndarray | pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall).

        Average loss beyond VaR threshold.
        """
        if len(returns) < 10:
            return 0.0
        var = self.calculate_var(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return float(np.mean(tail))

    def calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion for optimal position sizing.

        Args:
            win_rate: Probability of winning (0-1).
            avg_win: Average winning return (positive).
            avg_loss: Average losing return (positive, will be used as loss).

        Returns:
            Kelly fraction (0-1). Capped at 0.25 for safety.
        """
        if avg_loss <= 0 or avg_win <= 0 or not (0 < win_rate < 1):
            return 0.0
        b = avg_win / avg_loss  # Win/loss ratio
        kelly = (b * win_rate - (1 - win_rate)) / b
        return float(max(0.0, min(kelly, 0.25)))

    def calculate_max_position(
        self, portfolio_value: float, var_95: float, max_loss_pct: float = 0.02
    ) -> float:
        """Maximum position size based on VaR.

        Args:
            portfolio_value: Total portfolio value.
            var_95: 95% VaR (negative number).
            max_loss_pct: Maximum acceptable loss as fraction of portfolio.

        Returns:
            Maximum position size in currency.
        """
        if var_95 >= 0 or portfolio_value <= 0:
            return portfolio_value
        max_loss = portfolio_value * max_loss_pct
        return float(min(portfolio_value, max_loss / abs(var_95)))

    def full_risk_report(self, returns: np.ndarray | pd.Series) -> RiskMetrics:
        """Comprehensive risk analysis.

        Args:
            returns: Array of portfolio returns.

        Returns:
            RiskMetrics with VaR, CVaR, Kelly, max position size.
        """
        returns_arr = np.asarray(returns)
        var_95 = self.calculate_var(returns_arr)
        cvar_95 = self.calculate_cvar(returns_arr)

        # Calculate Kelly from win/loss stats
        wins = returns_arr[returns_arr > 0]
        losses = returns_arr[returns_arr < 0]
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(returns_arr)
            avg_win = float(np.mean(wins))
            avg_loss = float(np.mean(np.abs(losses)))
            kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        else:
            kelly = 0.0

        max_pos = self.calculate_max_position(100_000, var_95)

        return RiskMetrics(
            var_95=round(var_95, 6),
            cvar_95=round(cvar_95, 6),
            kelly_fraction=round(kelly, 4),
            max_position_size=round(max_pos, 2),
        )
