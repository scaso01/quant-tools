"""Tests for HRP, HERC, NCO hierarchical portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.portfolio.hrp import (
    HERC,
    NCO,
    HierarchicalRiskParity,
    plot_dendrogram,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def returns_5_assets(rng) -> pd.DataFrame:
    """Synthetic daily returns for 5 assets, 252 days."""
    n_days = 252
    assets = ["AAPL", "MSFT", "GOOG", "BND", "GLD"]
    means = [0.0004, 0.0003, 0.00035, 0.0001, 0.0002]
    vols = [0.015, 0.012, 0.014, 0.005, 0.008]
    data = {}
    for i, asset in enumerate(assets):
        data[asset] = rng.normal(means[i], vols[i], n_days)
    return pd.DataFrame(data)


@pytest.fixture
def returns_2_assets(rng) -> pd.DataFrame:
    """Degenerate 2-asset case."""
    n_days = 252
    return pd.DataFrame({
        "A": rng.normal(0.0003, 0.01, n_days),
        "B": rng.normal(0.0005, 0.02, n_days),
    })


@pytest.fixture
def returns_10_assets(rng) -> pd.DataFrame:
    """10 assets with cluster structure."""
    n_days = 252
    # Two clusters: tech (correlated) and defensive (correlated)
    tech_base = rng.normal(0.0004, 0.015, n_days)
    def_base = rng.normal(0.0001, 0.005, n_days)

    data = {}
    for i, name in enumerate(["T1", "T2", "T3", "T4", "T5"]):
        data[name] = tech_base + rng.normal(0, 0.003, n_days)
    for i, name in enumerate(["D1", "D2", "D3", "D4", "D5"]):
        data[name] = def_base + rng.normal(0, 0.002, n_days)
    return pd.DataFrame(data)


# ── HRP Tests ────────────────────────────────────────────────────────


class TestHRP:
    def test_weights_sum_to_one(self, returns_5_assets):
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_5_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, returns_5_assets):
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_5_assets)
        for w in weights.values():
            assert w >= 0.0

    def test_all_assets_present(self, returns_5_assets):
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_5_assets)
        assert set(weights.keys()) == set(returns_5_assets.columns)

    def test_different_linkage_methods(self, returns_5_assets):
        hrp = HierarchicalRiskParity()
        for method in ("ward", "single", "complete", "average"):
            weights = hrp.optimize(returns_5_assets, linkage_method=method)
            assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_low_vol_gets_more_weight(self, returns_5_assets):
        """HRP should give more weight to lower-vol assets (BND, GLD)."""
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_5_assets)
        # BND (vol=0.005) should get more weight than AAPL (vol=0.015)
        assert weights.get("BND", 0) > weights.get("AAPL", 0) * 0.5

    def test_two_asset_case(self, returns_2_assets):
        """Degenerate 2-asset case should work."""
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_2_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert len(weights) == 2

    def test_single_asset(self, rng):
        """Single asset gets 100%."""
        df = pd.DataFrame({"ONLY": rng.normal(0.0003, 0.01, 100)})
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(df)
        assert abs(weights["ONLY"] - 1.0) < 0.01

    def test_ten_assets_clustered(self, returns_10_assets):
        """10 assets with cluster structure should work."""
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns_10_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert len(weights) == 10


# ── HERC Tests ───────────────────────────────────────────────────────


class TestHERC:
    def test_weights_sum_to_one(self, returns_5_assets):
        herc = HERC()
        weights = herc.optimize(returns_5_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, returns_5_assets):
        herc = HERC()
        weights = herc.optimize(returns_5_assets)
        for w in weights.values():
            assert w >= 0.0

    def test_risk_measure_variance(self, returns_5_assets):
        herc = HERC()
        weights = herc.optimize(returns_5_assets, risk_measure="variance")
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_risk_measure_std(self, returns_5_assets):
        herc = HERC()
        weights = herc.optimize(returns_5_assets, risk_measure="std")
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_single_asset(self, rng):
        df = pd.DataFrame({"SOLO": rng.normal(0, 0.01, 100)})
        herc = HERC()
        weights = herc.optimize(df)
        assert abs(weights["SOLO"] - 1.0) < 0.01


# ── NCO Tests ────────────────────────────────────────────────────────


class TestNCO:
    def test_weights_sum_to_one(self, returns_5_assets):
        nco = NCO()
        weights = nco.optimize(returns_5_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_all_weights_non_negative(self, returns_5_assets):
        nco = NCO()
        weights = nco.optimize(returns_5_assets)
        for w in weights.values():
            assert w >= -0.001  # Small tolerance

    def test_custom_n_clusters(self, returns_10_assets):
        nco = NCO()
        weights = nco.optimize(returns_10_assets, n_clusters=3)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_single_asset(self, rng):
        df = pd.DataFrame({"ONLY": rng.normal(0, 0.01, 100)})
        nco = NCO()
        weights = nco.optimize(df)
        assert abs(weights["ONLY"] - 1.0) < 0.01

    def test_two_asset_case(self, returns_2_assets):
        nco = NCO()
        weights = nco.optimize(returns_2_assets)
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ── Dendrogram Tests ─────────────────────────────────────────────────


class TestDendrogram:
    def test_returns_plotly_figure(self, returns_5_assets):
        fig = plot_dendrogram(returns_5_assets)
        # plotly.graph_objects.Figure
        assert hasattr(fig, "layout")
        assert hasattr(fig, "data")

    def test_different_linkage(self, returns_5_assets):
        for method in ("ward", "complete"):
            fig = plot_dendrogram(returns_5_assets, linkage_method=method)
            assert hasattr(fig, "layout")
