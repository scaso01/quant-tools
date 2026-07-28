"""Tests for transaction cost models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.backtest.costs import (
    MarketImpactCosts,
    ProportionalCosts,
    SpreadCosts,
    TransactionCostModel,
    apply_costs,
)


# =============================================================================
# ProportionalCosts Tests
# =============================================================================


class TestProportionalCosts:
    """Test fixed basis point cost model."""

    def test_no_trades_no_cost(self):
        """Constant position should incur zero cost after initial entry."""
        positions = np.ones(100, dtype=np.float64)
        prices = np.linspace(100, 110, 100)
        model = ProportionalCosts(bps=10.0)
        costs = model.calculate(positions, prices)
        # Only the first bar has a position change (0 -> 1)
        assert costs[0] == pytest.approx(10.0 / 10_000)
        assert np.sum(costs[1:]) == pytest.approx(0.0)

    def test_every_trade_incurs_cost(self):
        """Alternating positions should incur cost every bar."""
        positions = np.array([1, -1, 1, -1, 1], dtype=np.float64)
        prices = np.ones(5)
        model = ProportionalCosts(bps=10.0)
        costs = model.calculate(positions, prices)
        # Position changes: |1-0|=1, |-1-1|=2, |1-(-1)|=2, |-1-1|=2, |1-(-1)|=2
        assert costs[0] == pytest.approx(1 * 10.0 / 10_000)
        assert costs[1] == pytest.approx(2 * 10.0 / 10_000)

    def test_zero_bps(self):
        """Zero bps should produce zero costs."""
        positions = np.array([0, 1, -1, 0], dtype=np.float64)
        prices = np.ones(4)
        model = ProportionalCosts(bps=0.0)
        costs = model.calculate(positions, prices)
        assert np.all(costs == 0.0)

    def test_output_length_matches_positions(self):
        """Output should be same length as positions."""
        positions = np.array([0, 1, 1, 0, -1], dtype=np.float64)
        prices = np.ones(5)
        model = ProportionalCosts(bps=5.0)
        costs = model.calculate(positions, prices)
        assert len(costs) == len(positions)


# =============================================================================
# MarketImpactCosts Tests
# =============================================================================


class TestMarketImpactCosts:
    """Test square-root market impact cost model."""

    def test_no_volume_fallback(self):
        """Without volume, should fall back to sigma as proportional cost."""
        positions = np.array([0, 1, 1, 0], dtype=np.float64)
        prices = np.ones(4)
        model = MarketImpactCosts(sigma=0.01)
        costs = model.calculate(positions, prices, volumes=None)
        # Trades at index 1 (0->1) and index 3 (1->0)
        assert costs[1] == pytest.approx(0.01)
        assert costs[3] == pytest.approx(0.01)

    def test_no_trades_no_cost(self):
        """Constant position with volume should produce zero cost after entry."""
        positions = np.ones(50, dtype=np.float64)
        prices = np.ones(50)
        volumes = np.full(50, 1_000_000.0)
        model = MarketImpactCosts(sigma=0.01)
        costs = model.calculate(positions, prices, volumes)
        # Only first bar has a trade (0->1)
        assert np.sum(costs[1:]) == pytest.approx(0.0, abs=1e-15)

    def test_output_length(self):
        """Output should be same length as positions."""
        positions = np.array([0, 1, 0], dtype=np.float64)
        prices = np.ones(3)
        volumes = np.full(3, 1_000_000.0)
        model = MarketImpactCosts(sigma=0.01)
        costs = model.calculate(positions, prices, volumes)
        assert len(costs) == 3


# =============================================================================
# SpreadCosts Tests
# =============================================================================


class TestSpreadCosts:
    """Test bid-ask spread cost model."""

    def test_spread_cost_on_trade(self):
        """Trade should incur spread cost."""
        positions = np.array([0, 1, 1, 0], dtype=np.float64)
        prices = np.ones(4)
        model = SpreadCosts(spread_bps=5.0)
        costs = model.calculate(positions, prices)
        # Trade at index 1 (0->1): cost = 1 * 5/10000
        assert costs[1] == pytest.approx(5.0 / 10_000)
        # No trade at index 2: cost = 0
        assert costs[2] == pytest.approx(0.0)
        # Trade at index 3 (1->0): cost = 1 * 5/10000
        assert costs[3] == pytest.approx(5.0 / 10_000)

    def test_zero_spread(self):
        """Zero spread should produce zero costs."""
        positions = np.array([0, 1, -1], dtype=np.float64)
        prices = np.ones(3)
        model = SpreadCosts(spread_bps=0.0)
        costs = model.calculate(positions, prices)
        assert np.all(costs == 0.0)

    def test_output_length(self):
        """Output should match positions length."""
        positions = np.zeros(10, dtype=np.float64)
        prices = np.ones(10)
        model = SpreadCosts(spread_bps=5.0)
        costs = model.calculate(positions, prices)
        assert len(costs) == 10


# =============================================================================
# apply_costs Tests
# =============================================================================


class TestApplyCosts:
    """Test cost application to returns."""

    def test_net_returns_less_than_gross(self):
        """Net returns should be <= gross returns when costs are positive."""
        positions = np.array([0, 1, 1, 0, -1], dtype=np.float64)
        returns = np.array([0.01, 0.02, -0.01, 0.005], dtype=np.float64)
        model = ProportionalCosts(bps=10.0)
        prices = np.ones(5)
        net = apply_costs(returns, positions, model, prices)
        assert np.all(net <= returns + 1e-15)

    def test_zero_cost_preserves_returns(self):
        """Zero cost model should preserve original returns."""
        positions = np.array([0, 1, -1, 0], dtype=np.float64)
        returns = np.array([0.01, -0.02, 0.005], dtype=np.float64)
        model = ProportionalCosts(bps=0.0)
        net = apply_costs(returns, positions, model)
        np.testing.assert_array_almost_equal(net, returns)

    def test_output_length_matches_returns(self):
        """Output should be same length as returns."""
        positions = np.array([0, 1, 1, 0], dtype=np.float64)
        returns = np.array([0.01, 0.02, -0.01], dtype=np.float64)
        model = ProportionalCosts(bps=5.0)
        net = apply_costs(returns, positions, model)
        assert len(net) == len(returns)

    def test_abc_cannot_instantiate(self):
        """TransactionCostModel ABC should not be directly instantiable."""
        with pytest.raises(TypeError):
            TransactionCostModel()
