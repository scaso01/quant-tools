"""Tests for the shared helpers behind risk, stress-test, tearsheet and ask.

``_portfolio_returns`` used to throw the dates away. Every caller still got a
number out, so nothing looked wrong, but the tearsheet builds its monthly and
yearly tables by resampling and both came out empty: a report covering two
years of data said "No yearly returns available". Dropping the index also meant
holdings were paired by row number rather than by day.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from quant_tools.cli import _parse_holdings, _portfolio_returns, _yf_crypto_symbol
from quant_tools.errors import DataError


def _prices(dates: pd.DatetimeIndex, start: float, step: float) -> pd.DataFrame:
    """A price frame rising by a fixed step, so returns are easy to reason about."""
    closes = [start + step * i for i in range(len(dates))]
    return pd.DataFrame({"Close": closes}, index=dates)


class TestPortfolioReturns:
    def test_returns_are_indexed_by_date(self):
        """Without a DatetimeIndex the tearsheet's resampled tables are empty."""
        dates = pd.bdate_range("2024-01-01", periods=40)
        frames = {"AAA": _prices(dates, 100.0, 1.0), "BBB": _prices(dates, 50.0, 0.5)}

        with patch(
            "quant_tools.data.market_data.fetch_prices",
            side_effect=lambda sym, **kw: frames[sym],
        ):
            result = _portfolio_returns({"AAA": 0.5, "BBB": 0.5})

        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.DatetimeIndex)
        # One row is consumed producing the first return.
        assert len(result) == len(dates) - 1

    def test_holdings_are_aligned_by_date_not_position(self):
        """A shorter history must line up on its own dates.

        The old version truncated each symbol to a common length from the
        front, so a holding that started later had its first day paired with
        the other holding's first day rather than the same calendar day.
        """
        long_dates = pd.bdate_range("2024-01-01", periods=40)
        short_dates = long_dates[20:]
        frames = {
            "AAA": _prices(long_dates, 100.0, 1.0),
            "BBB": _prices(short_dates, 50.0, 0.5),
        }

        with patch(
            "quant_tools.data.market_data.fetch_prices",
            side_effect=lambda sym, **kw: frames[sym],
        ):
            result = _portfolio_returns({"AAA": 0.5, "BBB": 0.5})

        # Only the overlapping window survives, and it starts where BBB does.
        assert result.index.min() > short_dates[0]
        assert result.index.max() == long_dates[-1]
        assert len(result) == len(short_dates) - 1

    def test_weights_are_applied(self):
        """An all-in weight on one symbol reproduces that symbol's returns."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        frames = {"AAA": _prices(dates, 100.0, 1.0), "BBB": _prices(dates, 50.0, 5.0)}

        with patch(
            "quant_tools.data.market_data.fetch_prices",
            side_effect=lambda sym, **kw: frames[sym],
        ):
            result = _portfolio_returns({"AAA": 1.0, "BBB": 0.0})

        expected = frames["AAA"]["Close"].pct_change().dropna()
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestParseHoldings:
    def test_valid_json(self):
        assert _parse_holdings('{"AAPL": 0.5, "BND": 0.5}') == {"AAPL": 0.5, "BND": 0.5}

    def test_malformed_json_is_rejected(self):
        with pytest.raises(DataError):
            _parse_holdings("{not json}")

    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        with pytest.raises(DataError):
            _parse_holdings('{"AAPL": 0.9, "BND": 0.9}')


class TestYfCryptoSymbol:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("BTC/USDT", "BTC-USD"),
            ("BTC/USD", "BTC-USD"),
            ("ETH/USDC", "ETH-USD"),
            ("BTC-USD", "BTC-USD"),
            ("AAPL", "AAPL"),
        ],
    )
    def test_pairs_are_rewritten_for_yfinance(self, given: str, expected: str):
        assert _yf_crypto_symbol(given) == expected
