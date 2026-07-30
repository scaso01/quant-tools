"""Regression tests for sector exposure when the portfolio holds funds.

The donut read "Unknown 80%" on a real portfolio. The cause was not the
rate-limiting the code's comments assumed: ``yf.Ticker().info`` answers fine, an
ETF simply has no ``sector`` field, so every fund fell into one opaque slice.
Equity funds do publish an underlying sector split, and bond and commodity funds
at least publish a category.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.data.market_data import _fund_sector_weights
from quant_tools.errors import DataError
from quant_tools.portfolio.analyzer import _sector_split, calculate_sector_exposure

# Trimmed from a live SPY response, weights as yfinance returns them.
SPY_WEIGHTS = {
    "technology": 0.3855,
    "financial_services": 0.1155,
    "healthcare": 0.0889,
    "realestate": 0.0183,
}

EQUITY = {"symbol": "MSFT", "sector": "Technology", "quote_type": "EQUITY"}
EQUITY_FUND = {
    "symbol": "SPY",
    "sector": "",
    "quote_type": "ETF",
    "category": "Large Blend",
    "sector_weights": {
        "Technology": 0.60,
        "Financial Services": 0.25,
        "Real Estate": 0.15,
    },
}
BOND_FUND = {
    "symbol": "BND",
    "sector": "",
    "quote_type": "ETF",
    "category": "Intermediate Core Bond",
    "sector_weights": {},
}


def _patch_info(mapping):
    return patch(
        "quant_tools.portfolio.analyzer.fetch_ticker_info",
        side_effect=lambda s: mapping[s],
    )


class TestFundSectorWeights:
    """The market_data helper that reads a fund's published breakdown."""

    def _ticker(self, weights):
        ticker = MagicMock()
        ticker.funds_data.sector_weightings = weights
        return ticker

    def test_equity_is_not_queried_for_fund_data(self):
        """A stock has no funds_data, so it must not be touched."""
        ticker = MagicMock()
        ticker.funds_data.sector_weightings = SPY_WEIGHTS
        assert _fund_sector_weights(ticker, {"quoteType": "EQUITY"}) == {}

    def test_etf_weights_are_title_cased(self):
        out = _fund_sector_weights(self._ticker(SPY_WEIGHTS), {"quoteType": "ETF"})
        assert out["Technology"] == pytest.approx(0.3855)
        assert out["Financial Services"] == pytest.approx(0.1155)

    def test_realestate_maps_to_the_equity_spelling(self):
        """yfinance says "realestate"; an equity's sector field says "Real Estate"."""
        out = _fund_sector_weights(self._ticker(SPY_WEIGHTS), {"quoteType": "ETF"})
        assert "Real Estate" in out
        assert "Realestate" not in out

    def test_zero_weight_sectors_are_dropped(self):
        weights = {"technology": 0.9, "energy": 0.0, "utilities": 0.1}
        out = _fund_sector_weights(self._ticker(weights), {"quoteType": "ETF"})
        assert "Energy" not in out

    def test_mutual_funds_are_included(self):
        out = _fund_sector_weights(self._ticker(SPY_WEIGHTS), {"quoteType": "MUTUALFUND"})
        assert out

    def test_missing_fund_data_is_not_fatal(self):
        """Not every fund answers. An empty split is fine, an exception is not."""
        ticker = MagicMock()
        type(ticker).funds_data = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no funds data"))
        )
        assert _fund_sector_weights(ticker, {"quoteType": "ETF"}) == {}


class TestSectorSplit:
    """How one holding divides across sectors."""

    def test_stock_is_one_sector(self):
        with _patch_info({"MSFT": EQUITY}):
            assert _sector_split("MSFT") == {"Technology": 1.0}

    def test_equity_fund_is_broken_out(self):
        with _patch_info({"SPY": EQUITY_FUND}):
            split = _sector_split("SPY")
        assert len(split) == 3
        assert split["Technology"] == pytest.approx(0.60)
        assert sum(split.values()) == pytest.approx(1.0)

    def test_rounded_fund_weights_are_normalized(self):
        """Published weightings are rounded and need not total exactly 1."""
        fund = dict(EQUITY_FUND, sector_weights={"Technology": 0.30, "Energy": 0.30})
        with _patch_info({"SPY": fund}):
            split = _sector_split("SPY")
        assert sum(split.values()) == pytest.approx(1.0)
        assert split["Technology"] == pytest.approx(0.5)

    def test_bond_fund_falls_back_to_category(self):
        """The bug: this used to be "Unknown"."""
        with _patch_info({"BND": BOND_FUND}):
            assert _sector_split("BND") == {"Intermediate Core Bond": 1.0}

    def test_no_sector_and_no_category_is_unknown(self):
        bare = {"symbol": "???", "sector": "", "quote_type": "ETF"}
        with _patch_info({"???": bare}):
            assert _sector_split("???") == {"Unknown": 1.0}

    def test_fetch_failure_is_unknown(self):
        with patch(
            "quant_tools.portfolio.analyzer.fetch_ticker_info",
            side_effect=DataError("delisted"),
        ):
            assert _sector_split("FAKE") == {"Unknown": 1.0}

    def test_unexpected_errors_still_surface(self):
        """A bug in the fetch path is not the same as a bad symbol."""
        with patch(
            "quant_tools.portfolio.analyzer.fetch_ticker_info",
            side_effect=TypeError("bug"),
        ), pytest.raises(TypeError):
            _sector_split("MSFT")


class TestExposureWithFunds:
    """The donut itself."""

    def test_fund_holding_no_longer_reads_unknown(self):
        holdings = {"MSFT": 0.5, "SPY": 0.5}
        with _patch_info({"MSFT": EQUITY, "SPY": EQUITY_FUND}):
            exposure = calculate_sector_exposure(holdings)
        assert "Unknown" not in exposure
        # 0.5 direct plus 0.5 * 0.60 through the fund.
        assert exposure["Technology"] == pytest.approx(0.80)
        assert exposure["Real Estate"] == pytest.approx(0.075)

    def test_weights_are_conserved(self):
        holdings = {"MSFT": 0.4, "SPY": 0.4, "BND": 0.2}
        with _patch_info({"MSFT": EQUITY, "SPY": EQUITY_FUND, "BND": BOND_FUND}):
            exposure = calculate_sector_exposure(holdings)
        assert sum(exposure.values()) == pytest.approx(1.0)

    def test_all_fund_portfolio_is_fully_resolved(self):
        """The reported case: mostly ETFs, 80% of the chart unlabelled."""
        holdings = {"SPY": 0.8, "BND": 0.2}
        with _patch_info({"SPY": EQUITY_FUND, "BND": BOND_FUND}):
            exposure = calculate_sector_exposure(holdings)
        assert exposure.get("Unknown", 0.0) == 0.0
        assert exposure["Technology"] == pytest.approx(0.48)
