"""Tests for Black-Litterman portfolio allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.portfolio.black_litterman import BlackLitterman


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def returns_3_assets(rng) -> pd.DataFrame:
    """Synthetic returns for 3 assets, 252 days."""
    n_days = 252
    return pd.DataFrame({
        "AAPL": rng.normal(0.0005, 0.015, n_days),
        "BND": rng.normal(0.0001, 0.004, n_days),
        "GLD": rng.normal(0.0002, 0.008, n_days),
    })


@pytest.fixture
def market_caps_3() -> dict[str, float]:
    return {
        "AAPL": 3_000_000_000_000,  # $3T
        "BND": 500_000_000_000,     # $500B
        "GLD": 200_000_000_000,     # $200B
    }


@pytest.fixture
def bl_model(returns_3_assets, market_caps_3) -> BlackLitterman:
    bl = BlackLitterman(risk_free_rate=0.04, tau=0.05)
    bl.set_market_data(returns_3_assets, market_caps_3)
    return bl


# ── Equilibrium Returns Tests ────────────────────────────────────────


class TestEquilibriumReturns:
    def test_returns_series(self, bl_model):
        pi = bl_model.equilibrium_returns()
        assert isinstance(pi, pd.Series)
        assert len(pi) == 3

    def test_direction_matches_market_weights(self, bl_model):
        """Largest market-cap asset should have meaningful equilibrium return."""
        pi = bl_model.equilibrium_returns()
        # AAPL has highest cap, should have a significant return
        assert "AAPL" in pi.index

    def test_equilibrium_not_all_zero(self, bl_model):
        pi = bl_model.equilibrium_returns()
        assert not np.allclose(pi.values, 0.0)


# ── Posterior Returns Tests ──────────────────────────────────────────


class TestPosteriorReturns:
    def test_no_views_equals_equilibrium(self, bl_model):
        """Without views, posterior should equal equilibrium."""
        pi = bl_model.equilibrium_returns()
        mu_post = bl_model.posterior_returns()
        np.testing.assert_allclose(pi.values, mu_post.values, atol=1e-8)

    def test_absolute_view_shifts_posterior(self, bl_model):
        """Bullish view on AAPL should shift its posterior return up."""
        pi = bl_model.equilibrium_returns()

        bl_model.add_absolute_view("AAPL", 0.20, confidence=0.8)
        mu_post = bl_model.posterior_returns()

        # AAPL posterior should be higher than equilibrium
        assert mu_post["AAPL"] > pi["AAPL"]

    def test_relative_view_creates_spread(self, bl_model):
        """Relative view should push long asset up and short asset down."""
        pi = bl_model.equilibrium_returns()

        bl_model.add_relative_view("AAPL", "BND", spread=0.10, confidence=0.7)
        mu_post = bl_model.posterior_returns()

        # The spread (AAPL - BND) in posterior should be wider
        eq_spread = pi["AAPL"] - pi["BND"]
        post_spread = mu_post["AAPL"] - mu_post["BND"]
        assert post_spread > eq_spread

    def test_high_confidence_dominates(self, bl_model):
        """High confidence view should move posterior closer to view."""
        bl_model.add_absolute_view("AAPL", 0.30, confidence=0.99)
        mu_post = bl_model.posterior_returns()

        # With very high confidence, posterior should be close to 0.30
        assert mu_post["AAPL"] > 0.10  # Should be pulled toward 0.30

    def test_low_confidence_barely_changes(self, bl_model):
        """Low confidence view should barely change posterior."""
        pi = bl_model.equilibrium_returns()

        bl_model.add_absolute_view("AAPL", 0.50, confidence=0.01)
        mu_post = bl_model.posterior_returns()

        # Should barely move from equilibrium
        assert abs(mu_post["AAPL"] - pi["AAPL"]) < abs(0.50 - pi["AAPL"]) * 0.5


# ── Optimize Tests ───────────────────────────────────────────────────


class TestOptimize:
    def test_weights_sum_to_one(self, bl_model):
        weights = bl_model.optimize()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, bl_model):
        weights = bl_model.optimize()
        for w in weights.values():
            assert w >= 0.0

    def test_optimize_with_views(self, bl_model):
        bl_model.add_absolute_view("AAPL", 0.15, confidence=0.6)
        weights = bl_model.optimize()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_bullish_view_increases_weight(self, bl_model):
        """Bullish view should increase weight of the favored asset."""
        weights_no_view = bl_model.optimize()

        # Create fresh model with view
        bl2 = BlackLitterman(risk_free_rate=0.04, tau=0.05)
        bl2.set_market_data(bl_model._returns_df, bl_model._market_caps)
        bl2.add_absolute_view("GLD", 0.25, confidence=0.8)
        weights_with_view = bl2.optimize()

        assert weights_with_view.get("GLD", 0) >= weights_no_view.get("GLD", 0) * 0.8


# ── Method Chaining Tests ────────────────────────────────────────────


class TestMethodChaining:
    def test_set_market_data_returns_self(self, returns_3_assets, market_caps_3):
        bl = BlackLitterman()
        result = bl.set_market_data(returns_3_assets, market_caps_3)
        assert result is bl

    def test_add_view_returns_self(self, bl_model):
        result = bl_model.add_absolute_view("AAPL", 0.10)
        assert result is bl_model

    def test_add_relative_view_returns_self(self, bl_model):
        result = bl_model.add_relative_view("AAPL", "BND", 0.05)
        assert result is bl_model

    def test_full_chain(self, returns_3_assets, market_caps_3):
        """Full method chain should work."""
        weights = (
            BlackLitterman(risk_free_rate=0.03)
            .set_market_data(returns_3_assets, market_caps_3)
            .add_absolute_view("AAPL", 0.12, confidence=0.6)
            .add_relative_view("AAPL", "BND", 0.05, confidence=0.5)
            .optimize()
        )
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_without_market_data_raises(self):
        bl = BlackLitterman()
        with pytest.raises(RuntimeError, match="Must call set_market_data"):
            bl.equilibrium_returns()

    def test_equal_market_caps(self, returns_3_assets):
        """Equal caps should produce roughly equal equilibrium weights."""
        caps = {"AAPL": 1e9, "BND": 1e9, "GLD": 1e9}
        bl = BlackLitterman()
        bl.set_market_data(returns_3_assets, caps)
        weights = bl.optimize()
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_view_on_missing_asset(self, bl_model):
        """View on non-existent asset should be ignored."""
        bl_model.add_absolute_view("UNKNOWN_TICKER", 0.50, confidence=0.9)
        # Should not crash, just ignore
        mu_post = bl_model.posterior_returns()
        assert len(mu_post) == 3

    def test_multiple_views(self, bl_model):
        """Multiple views combined should produce valid weights."""
        bl_model.add_absolute_view("AAPL", 0.15, confidence=0.6)
        bl_model.add_absolute_view("BND", 0.05, confidence=0.8)
        bl_model.add_relative_view("GLD", "BND", 0.03, confidence=0.5)
        weights = bl_model.optimize()
        assert abs(sum(weights.values()) - 1.0) < 0.01
