"""Regression tests for how the backtest engine splits a position series into trades.

``_extract_trade_returns`` documented a trade as "a contiguous block of non-zero,
same-sign positions" but only tested ``pos != 0``. A strategy that reverses
straight from long to short without a flat day therefore had every reversal
merged into one enormous trade. On a five year AAPL crossover backtest that
reported 1 trade, a 100% win rate and an infinite profit factor.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.engine import _extract_trade_returns, run_backtest


def _returns(n: int, value: float = 0.01) -> np.ndarray:
    return np.full(n, value, dtype=np.float64)


class TestTradeSegmentation:
    """One trade per contiguous same-sign run."""

    def test_continuous_long_is_one_trade(self) -> None:
        positions = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(4))) == 1

    def test_long_then_flat_then_long_is_two_trades(self) -> None:
        positions = np.array([1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(5))) == 2

    def test_reversal_without_a_flat_day_is_two_trades(self) -> None:
        """The bug. Long straight to short is a close and an open."""
        positions = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(4))) == 2

    def test_repeated_reversals_each_count(self) -> None:
        positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(5))) == 5

    def test_all_flat_is_no_trades(self) -> None:
        positions = np.zeros(5, dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(5))) == 0

    def test_open_trade_at_the_end_is_still_counted(self) -> None:
        positions = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        assert len(_extract_trade_returns(positions, _returns(4))) == 1


class TestTradeReturns:
    """Each segment accumulates its own signed return."""

    def test_short_segment_profits_when_price_falls(self) -> None:
        positions = np.array([1.0, -1.0], dtype=np.float64)
        daily = np.array([0.02, -0.05], dtype=np.float64)
        long_leg, short_leg = _extract_trade_returns(positions, daily)
        assert long_leg == pytest.approx(0.02)
        assert short_leg == pytest.approx(0.05)

    def test_a_reversal_does_not_pool_opposite_pnl(self) -> None:
        """Merged, these cancel to zero and hide both a winner and a loser."""
        positions = np.array([1.0, -1.0], dtype=np.float64)
        daily = np.array([0.03, 0.03], dtype=np.float64)
        trades = _extract_trade_returns(positions, daily)
        assert len(trades) == 2
        assert trades[0] == pytest.approx(0.03)
        assert trades[1] == pytest.approx(-0.03)
        assert sum(trades) == pytest.approx(0.0)


class TestDerivedMetrics:
    """Win rate and profit factor are computed from the trade list."""

    def test_alternating_strategy_reports_every_trade(self) -> None:
        """Six bars give five returns, so five reversals are five trades.

        Before the fix this was 1 regardless of how often the signal flipped.
        """
        close = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 106.0])
        positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        result = run_backtest(close, positions, ticker="TEST")
        assert result.total_trades == 5

    def test_reversals_produce_a_finite_profit_factor(self) -> None:
        """Pooling every reversal left no losing trade, so this was inf.

        The series is chosen so the signal is right three times and wrong twice.
        """
        close = np.array([100.0, 105.0, 110.0, 105.0, 100.0, 108.0])
        positions = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        result = run_backtest(close, positions, ticker="TEST")
        assert result.total_trades == 5
        assert np.isfinite(result.profit_factor)
        assert 0.0 < result.win_rate < 1.0
