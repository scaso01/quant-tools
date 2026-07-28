"""Tests for factors/scoring.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.errors import DataError
from quant_tools.factors.scoring import (
    _compute_momentum_scores,
    _compute_quality_scores,
    _compute_value_scores,
    _safe_zscore,
    calculate_factor_scores,
)
from quant_tools.models import FactorScore


# =============================================================================
# _safe_zscore
# =============================================================================


class TestSafeZscore:
    """Test z-score computation with NaN handling."""

    def test_normal_values(self):
        """Z-scores of normal values should have mean ~0 and std ~1."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        z = _safe_zscore(values)
        assert len(z) == 5
        assert abs(np.mean(z)) < 1e-10
        assert abs(np.std(z) - 1.0) < 0.01

    def test_with_none_values(self):
        """None values should get z-score of 0."""
        values = [10.0, None, 30.0, None, 50.0]
        z = _safe_zscore(values)
        assert len(z) == 5
        assert z[1] == 0.0
        assert z[3] == 0.0

    def test_all_none(self):
        """All-None input should return all zeros."""
        values = [None, None, None]
        z = _safe_zscore(values)
        assert np.all(z == 0.0)

    def test_single_value(self):
        """Single value should return zeros (can't compute z-score)."""
        values = [42.0]
        z = _safe_zscore(values)
        assert np.all(z == 0.0)

    def test_identical_values(self):
        """Identical values (std=0) should return all zeros."""
        values = [5.0, 5.0, 5.0, 5.0]
        z = _safe_zscore(values)
        assert np.all(z == 0.0)

    def test_empty_list(self):
        """Empty input should return empty array."""
        z = _safe_zscore([])
        assert len(z) == 0


# =============================================================================
# _compute_value_scores
# =============================================================================


class TestValueScores:
    """Test value factor z-score computation."""

    def test_lower_pe_higher_value(self):
        """Stocks with lower PE should have higher value z-scores."""
        infos = [
            {"pe_ratio": 10.0, "pb_ratio": 1.0, "dividend_yield": 0.04},
            {"pe_ratio": 30.0, "pb_ratio": 3.0, "dividend_yield": 0.01},
            {"pe_ratio": 50.0, "pb_ratio": 5.0, "dividend_yield": 0.005},
        ]
        scores = _compute_value_scores(infos)
        assert len(scores) == 3
        # Lower PE/PB + higher div yield = highest value
        assert scores[0] > scores[1] > scores[2]

    def test_missing_metrics(self):
        """Missing metrics should not crash."""
        infos = [
            {"pe_ratio": 10.0, "pb_ratio": None, "dividend_yield": None},
            {"pe_ratio": None, "pb_ratio": 2.0, "dividend_yield": 0.02},
        ]
        scores = _compute_value_scores(infos)
        assert len(scores) == 2
        assert not np.any(np.isnan(scores))


# =============================================================================
# _compute_quality_scores
# =============================================================================


class TestQualityScores:
    """Test quality factor z-score computation."""

    def test_higher_roe_higher_quality(self):
        """Stocks with higher ROE should score higher on quality."""
        infos = [
            {"roe": 0.30, "profit_margin": 0.25, "debt_to_equity": 0.5},
            {"roe": 0.15, "profit_margin": 0.15, "debt_to_equity": 1.0},
            {"roe": 0.05, "profit_margin": 0.05, "debt_to_equity": 2.0},
        ]
        scores = _compute_quality_scores(infos)
        assert scores[0] > scores[1] > scores[2]

    def test_lower_debt_higher_quality(self):
        """Stocks with lower debt/equity should contribute positively."""
        infos = [
            {"roe": 0.15, "profit_margin": 0.15, "debt_to_equity": 0.1},
            {"roe": 0.15, "profit_margin": 0.15, "debt_to_equity": 5.0},
        ]
        scores = _compute_quality_scores(infos)
        assert scores[0] > scores[1]


# =============================================================================
# _compute_momentum_scores
# =============================================================================


class TestMomentumScores:
    """Test momentum factor z-score computation."""

    def test_momentum_with_mock_prices(self):
        """Momentum scores should reflect return differences."""
        np.random.seed(42)
        n = 252

        def _make_prices(start: float, drift: float) -> pd.DataFrame:
            dates = pd.bdate_range(start="2023-01-02", periods=n)
            returns = np.random.normal(drift, 0.01, n)
            close = start * np.cumprod(1 + returns)
            return pd.DataFrame(
                {"Close": close, "Open": close * 0.999, "High": close * 1.005,
                 "Low": close * 0.995, "Volume": 1_000_000, "Adj Close": close},
                index=dates,
            )

        prices_map = {
            "WINNER": _make_prices(100, 0.002),   # Strong uptrend
            "FLAT": _make_prices(100, 0.0),        # Flat
            "LOSER": _make_prices(100, -0.002),    # Downtrend
        }

        with patch("quant_tools.factors.scoring.fetch_prices") as mock_fp:
            mock_fp.side_effect = lambda sym, **kw: prices_map[sym]
            scores = _compute_momentum_scores(["WINNER", "FLAT", "LOSER"])

        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_momentum_handles_fetch_failure(self):
        """Failed price fetches should result in None -> 0 z-score."""
        with patch("quant_tools.factors.scoring.fetch_prices") as mock_fp:
            mock_fp.side_effect = Exception("API error")
            scores = _compute_momentum_scores(["FAIL1", "FAIL2"])

        assert len(scores) == 2
        assert np.all(scores == 0.0)


# =============================================================================
# calculate_factor_scores
# =============================================================================


class TestCalculateFactorScores:
    """Test composite factor score calculation."""

    def test_calculates_and_ranks(self):
        """Should produce ranked FactorScore objects."""
        mock_infos = {
            "AAPL": {
                "symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology",
                "pe_ratio": 28.0, "pb_ratio": 40.0, "dividend_yield": 0.005,
                "roe": 0.15, "profit_margin": 0.25, "debt_to_equity": 1.5,
            },
            "JPM": {
                "symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financials",
                "pe_ratio": 12.0, "pb_ratio": 1.8, "dividend_yield": 0.025,
                "roe": 0.13, "profit_margin": 0.35, "debt_to_equity": 2.5,
            },
            "JNJ": {
                "symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare",
                "pe_ratio": 15.0, "pb_ratio": 5.0, "dividend_yield": 0.03,
                "roe": 0.20, "profit_margin": 0.20, "debt_to_equity": 0.5,
            },
        }

        np.random.seed(42)
        n = 252
        dates = pd.bdate_range(start="2023-01-02", periods=n)

        def _mock_prices(symbol, **kwargs):
            close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
            return pd.DataFrame(
                {"Close": close, "Open": close, "High": close, "Low": close,
                 "Volume": 1_000_000, "Adj Close": close},
                index=dates,
            )

        with patch("quant_tools.factors.scoring.fetch_ticker_info") as mock_info, \
             patch("quant_tools.factors.scoring.fetch_prices") as mock_prices:
            mock_info.side_effect = lambda sym: mock_infos[sym]
            mock_prices.side_effect = _mock_prices

            results = calculate_factor_scores(["AAPL", "JPM", "JNJ"])

        assert len(results) == 3
        assert all(isinstance(r, FactorScore) for r in results)
        # Ranks should be 1, 2, 3
        ranks = [r.rank for r in results]
        assert sorted(ranks) == [1, 2, 3]
        # Sorted by composite descending
        assert results[0].composite_z >= results[1].composite_z >= results[2].composite_z

    def test_empty_symbols_raises(self):
        """Empty symbol list should raise DataError."""
        with pytest.raises(DataError, match="No symbols provided"):
            calculate_factor_scores([])

    def test_all_symbols_fail_raises(self):
        """All symbols failing should raise DataError."""
        with patch("quant_tools.factors.scoring.fetch_ticker_info") as mock_info:
            mock_info.side_effect = Exception("API down")
            with pytest.raises(DataError, match="No valid symbols"):
                calculate_factor_scores(["FAIL1", "FAIL2"])

    def test_custom_weights(self):
        """Custom weights should change composite scores."""
        mock_infos = {
            "A": {
                "symbol": "A", "name": "Stock A", "sector": "Tech",
                "pe_ratio": 10.0, "pb_ratio": 1.0, "dividend_yield": 0.05,
                "roe": 0.05, "profit_margin": 0.05, "debt_to_equity": 3.0,
            },
            "B": {
                "symbol": "B", "name": "Stock B", "sector": "Tech",
                "pe_ratio": 50.0, "pb_ratio": 10.0, "dividend_yield": 0.0,
                "roe": 0.30, "profit_margin": 0.30, "debt_to_equity": 0.1,
            },
        }

        np.random.seed(42)
        n = 252
        dates = pd.bdate_range(start="2023-01-02", periods=n)

        def _mock_prices(symbol, **kwargs):
            close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
            return pd.DataFrame(
                {"Close": close, "Open": close, "High": close, "Low": close,
                 "Volume": 1_000_000, "Adj Close": close},
                index=dates,
            )

        # Heavy value weight should favor Stock A (low PE/PB, high div)
        value_weights = {"value": 0.80, "momentum": 0.10, "quality": 0.10}
        # Heavy quality weight should favor Stock B (high ROE, margin, low debt)
        quality_weights = {"value": 0.10, "momentum": 0.10, "quality": 0.80}

        with patch("quant_tools.factors.scoring.fetch_ticker_info") as mock_info, \
             patch("quant_tools.factors.scoring.fetch_prices") as mock_prices:
            mock_info.side_effect = lambda sym: mock_infos[sym]
            mock_prices.side_effect = _mock_prices

            value_results = calculate_factor_scores(["A", "B"], weights=value_weights)
            quality_results = calculate_factor_scores(["A", "B"], weights=quality_weights)

        # With value weights, Stock A should rank #1
        assert value_results[0].symbol == "A"
        # With quality weights, Stock B should rank #1
        assert quality_results[0].symbol == "B"

    def test_partial_symbol_failure(self):
        """Some symbols failing should still return valid results."""
        mock_infos = {
            "GOOD": {
                "symbol": "GOOD", "name": "Good Stock", "sector": "Tech",
                "pe_ratio": 20.0, "pb_ratio": 3.0, "dividend_yield": 0.02,
                "roe": 0.15, "profit_margin": 0.20, "debt_to_equity": 1.0,
            },
        }

        np.random.seed(42)
        n = 252
        dates = pd.bdate_range(start="2023-01-02", periods=n)

        def _mock_prices(symbol, **kwargs):
            close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
            return pd.DataFrame(
                {"Close": close, "Open": close, "High": close, "Low": close,
                 "Volume": 1_000_000, "Adj Close": close},
                index=dates,
            )

        def _mock_info(sym):
            if sym in mock_infos:
                return mock_infos[sym]
            raise Exception(f"No data for {sym}")

        with patch("quant_tools.factors.scoring.fetch_ticker_info") as mock_info, \
             patch("quant_tools.factors.scoring.fetch_prices") as mock_prices:
            mock_info.side_effect = _mock_info
            mock_prices.side_effect = _mock_prices

            results = calculate_factor_scores(["GOOD", "BAD1", "BAD2"])

        assert len(results) == 1
        assert results[0].symbol == "GOOD"
