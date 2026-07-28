"""Tests for the market scanner (universe, signals, engine)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.data.universe import (
    SECTOR_MAP,
    SP500_TOP100,
    get_sector_tickers,
    get_ticker_sector,
    get_universe,
)
from quant_tools.models import ScanResult, SignalType
from quant_tools.models import AssetClass
from quant_tools.scanner.signals import (
    commodity_signal,
    crypto_momentum_signal,
    forex_signal,
    mean_reversion_signal,
    momentum_signal,
    quality_signal,
    value_signal,
)


# =============================================================================
# Universe Tests
# =============================================================================


class TestUniverse:
    """Tests for universe.py."""

    def test_sp500_has_100_tickers(self):
        assert len(SP500_TOP100) == 100

    def test_get_universe_returns_list(self):
        result = get_universe("sp500")
        assert isinstance(result, list)
        assert len(result) == 100
        assert "AAPL" in result
        assert "MSFT" in result

    def test_get_universe_case_insensitive(self):
        result = get_universe("SP500")
        assert len(result) == 100

    def test_get_universe_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nasdaq100")

    def test_all_11_sectors_represented(self):
        sectors = set(SP500_TOP100.values())
        expected = {
            "Technology", "Healthcare", "Financials",
            "Consumer Discretionary", "Consumer Staples",
            "Industrials", "Energy", "Materials",
            "Utilities", "Real Estate", "Communication Services",
        }
        assert sectors == expected

    def test_sector_map_covers_all_sectors(self):
        assert set(SECTOR_MAP.keys()) == set(SP500_TOP100.values())

    def test_get_sector_tickers_technology(self):
        tech = get_sector_tickers("Technology")
        assert isinstance(tech, list)
        assert "AAPL" in tech
        assert "MSFT" in tech
        assert len(tech) > 0

    def test_get_sector_tickers_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sector"):
            get_sector_tickers("Fake Sector")

    def test_get_ticker_sector(self):
        assert get_ticker_sector("AAPL") == "Technology"
        assert get_ticker_sector("JPM") == "Financials"

    def test_get_ticker_sector_case_insensitive(self):
        assert get_ticker_sector("aapl") == "Technology"

    def test_get_ticker_sector_unknown_raises(self):
        with pytest.raises(KeyError, match="not in SP500_TOP100"):
            get_ticker_sector("ZZZZZ")

    def test_get_sector_tickers_returns_copy(self):
        """Ensure returned list is a copy, not the internal list."""
        t1 = get_sector_tickers("Technology")
        t2 = get_sector_tickers("Technology")
        assert t1 is not t2


# =============================================================================
# Signal Tests
# =============================================================================


class TestMomentumSignal:
    """Tests for momentum_signal."""

    def test_returns_float_in_range(self, sample_close_array):
        score = momentum_signal(sample_close_array)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_short_array_returns_50(self):
        close = np.array([100.0, 101.0, 99.0])
        assert momentum_signal(close) == 50.0

    def test_with_volume(self, sample_prices):
        close = sample_prices["Close"].values
        volume = sample_prices["Volume"].values.astype(np.float64)
        score = momentum_signal(close, volume)
        assert 0 <= score <= 100

    def test_uptrend_scores_higher(self):
        """A strong uptrend should score higher than a downtrend."""
        n = 100
        uptrend = np.linspace(100, 150, n)
        downtrend = np.linspace(150, 100, n)
        up_score = momentum_signal(uptrend)
        down_score = momentum_signal(downtrend)
        assert up_score > down_score


class TestValueSignal:
    """Tests for value_signal."""

    def test_returns_float_in_range(self, sample_ticker_info):
        score = value_signal(sample_ticker_info)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_empty_info_returns_50(self):
        assert value_signal({}) == 50.0

    def test_low_pe_scores_higher(self):
        low_pe = {"pe_ratio": 8, "pb_ratio": 1.0}
        high_pe = {"pe_ratio": 50, "pb_ratio": 10.0}
        assert value_signal(low_pe) > value_signal(high_pe)

    def test_none_values_handled(self):
        info = {"pe_ratio": None, "pb_ratio": None}
        score = value_signal(info)
        assert score == 50.0


class TestQualitySignal:
    """Tests for quality_signal."""

    def test_returns_float_in_range(self, sample_ticker_info):
        score = quality_signal(sample_ticker_info)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_empty_info_returns_50(self):
        assert quality_signal({}) == 50.0

    def test_high_quality_scores_higher(self):
        high_q = {"roe": 0.30, "profit_margin": 0.35, "debt_to_equity": 20}
        low_q = {"roe": 0.02, "profit_margin": 0.02, "debt_to_equity": 200}
        assert quality_signal(high_q) > quality_signal(low_q)


class TestMeanReversionSignal:
    """Tests for mean_reversion_signal."""

    def test_returns_float_in_range(self, sample_close_array):
        score = mean_reversion_signal(sample_close_array)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_short_array_returns_50(self):
        close = np.array([100.0, 101.0, 99.0])
        assert mean_reversion_signal(close) == 50.0

    def test_oversold_scores_higher(self):
        """A sharply declining series (oversold) should score higher for reversion."""
        n = 60
        # Steady price then sharp drop
        steady = np.full(n, 100.0)
        drop = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 20)])
        drop_score = mean_reversion_signal(drop)
        steady_score = mean_reversion_signal(steady)
        # Dropped stock has more reversion potential
        assert drop_score > steady_score


# =============================================================================
# Engine Tests
# =============================================================================


class TestScanUniverse:
    """Tests for engine.scan_universe."""

    def _make_mock_df(self, n: int = 252) -> pd.DataFrame:
        """Create a mock price DataFrame."""
        np.random.seed(42)
        dates = pd.bdate_range(start="2023-01-02", periods=n)
        returns = np.random.normal(0.0005, 0.015, n)
        close = 100 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 50_000_000, n),
                "Adj Close": close,
            },
            index=dates,
        )

    def _make_mock_info(self, symbol: str = "AAPL") -> dict:
        """Create a mock ticker info dict."""
        return {
            "symbol": symbol,
            "name": f"{symbol} Inc.",
            "sector": "Technology",
            "pe_ratio": 25.0,
            "forward_pe": 22.0,
            "pb_ratio": 8.0,
            "roe": 0.15,
            "profit_margin": 0.25,
            "debt_to_equity": 1.5,
        }

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_returns_scan_results(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL", "MSFT", "GOOGL"],
            signals=[SignalType.MOMENTUM],
            max_workers=2,
            top_n=10,
        )

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ScanResult) for r in results)

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_results_sorted_by_score(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
            signals=[SignalType.MOMENTUM],
            max_workers=2,
        )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_top_n_limits_results(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            signals=[SignalType.MOMENTUM],
            max_workers=2,
            top_n=2,
        )

        assert len(results) <= 2

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_with_value_signal(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL"],
            signals=[SignalType.VALUE],
            max_workers=1,
        )

        assert len(results) == 1
        assert results[0].signal == SignalType.VALUE

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_with_all_signals(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL"],
            signals=None,  # defaults to all
            max_workers=1,
        )

        assert len(results) == 1
        # Details should have scores for all signal types + breakdown
        details = results[0].details
        for st in SignalType:
            assert st.value in details
        assert "breakdown" in details

    def test_empty_universe_returns_empty(self):
        from quant_tools.scanner.engine import scan_universe

        results = scan_universe(symbols=[], signals=[SignalType.MOMENTUM])
        assert results == []

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_handles_symbol_failure_gracefully(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        def side_effect(symbol, **kwargs):
            if symbol == "BAD":
                raise RuntimeError("Simulated failure")
            return self._make_mock_df()

        mock_prices.side_effect = side_effect
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL", "BAD", "MSFT"],
            signals=[SignalType.MOMENTUM],
            max_workers=2,
        )

        # BAD should be skipped, AAPL and MSFT should succeed
        symbols_returned = {r.symbol for r in results}
        assert "BAD" not in symbols_returned
        assert len(results) == 2

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_all_symbols_fail_returns_empty(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.side_effect = RuntimeError("All fail")
        mock_info.side_effect = RuntimeError("All fail")

        results = scan_universe(
            symbols=["AAPL", "MSFT"],
            signals=[SignalType.MOMENTUM],
            max_workers=1,
        )

        assert results == []

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_scan_result_has_price_data(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = self._make_mock_info()

        results = scan_universe(
            symbols=["AAPL"],
            signals=[SignalType.MOMENTUM],
            max_workers=1,
        )

        assert len(results) == 1
        r = results[0]
        assert r.price > 0
        assert r.symbol == "AAPL"
        assert isinstance(r.change_1d, float)
        assert isinstance(r.change_1m, float)


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for cli.py."""

    def test_build_parser(self):
        from quant_tools.scanner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.universe == "sp500"
        assert args.top_n == 20
        assert args.period == "1y"

    def test_parser_with_args(self):
        from quant_tools.scanner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--universe", "sp500",
            "--sector", "Technology",
            "--signals", "momentum", "value",
            "--top-n", "10",
            "--period", "6mo",
        ])
        assert args.sector == "Technology"
        assert args.signals == ["momentum", "value"]
        assert args.top_n == 10
        assert args.period == "6mo"

    @patch("quant_tools.scanner.cli.scan_universe")
    @patch("quant_tools.scanner.cli.get_universe")
    def test_main_runs_successfully(self, mock_get_universe, mock_scan):
        from quant_tools.scanner.cli import main

        mock_get_universe.return_value = ["AAPL", "MSFT"]
        mock_scan.return_value = [
            ScanResult(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                signal=SignalType.MOMENTUM,
                score=75.0,
                price=150.0,
                change_1d=1.5,
                change_1m=5.0,
                rsi_14=55.0,
            ),
        ]

        exit_code = main(["--top-n", "5"])
        assert exit_code == 0

    @patch("quant_tools.scanner.cli.scan_universe")
    @patch("quant_tools.scanner.cli.get_universe")
    def test_main_no_results(self, mock_get_universe, mock_scan):
        from quant_tools.scanner.cli import main

        mock_get_universe.return_value = ["AAPL"]
        mock_scan.return_value = []

        exit_code = main([])
        assert exit_code == 0

    def test_main_bad_sector(self):
        from quant_tools.scanner.cli import main

        exit_code = main(["--sector", "FakeSector"])
        assert exit_code == 1

    def test_parser_asset_class_default(self):
        from quant_tools.scanner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.asset_class == "equity"

    def test_parser_asset_class_crypto(self):
        from quant_tools.scanner.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--asset-class", "crypto"])
        assert args.asset_class == "crypto"


# =============================================================================
# Multi-Asset Signal Tests
# =============================================================================


class TestCryptoSignal:
    """Tests for crypto_momentum_signal."""

    def test_basic_crypto_signal(self):
        np.random.seed(42)
        close = np.random.normal(50000, 2000, 100).cumsum() + 40000
        close = np.abs(close)
        result = crypto_momentum_signal(close)
        assert 0 <= result <= 100

    def test_short_data(self):
        assert crypto_momentum_signal(np.array([100.0, 101.0, 102.0])) == 50.0

    def test_with_volume(self):
        np.random.seed(42)
        close = np.linspace(100, 150, 100)
        volume = np.random.randint(1_000_000, 50_000_000, 100).astype(np.float64)
        result = crypto_momentum_signal(close, volume)
        assert 0 <= result <= 100

    def test_returns_float(self):
        close = np.linspace(100, 200, 60)
        result = crypto_momentum_signal(close)
        assert isinstance(result, float)


class TestForexSignal:
    """Tests for forex_signal."""

    def test_basic_forex_signal(self):
        close = np.linspace(1.10, 1.15, 100)
        result = forex_signal(close)
        assert 0 <= result <= 100

    def test_short_data(self):
        assert forex_signal(np.array([1.10, 1.11, 1.12])) == 50.0

    def test_stable_pair(self):
        close = np.full(100, 1.10)
        result = forex_signal(close)
        assert 0 <= result <= 100

    def test_returns_float(self):
        close = np.linspace(1.10, 1.15, 60)
        result = forex_signal(close)
        assert isinstance(result, float)


class TestCommoditySignal:
    """Tests for commodity_signal."""

    def test_basic_commodity_signal(self):
        np.random.seed(42)
        close = np.random.normal(100, 10, 100).cumsum() + 500
        close = np.abs(close)
        result = commodity_signal(close)
        assert 0 <= result <= 100

    def test_short_data(self):
        assert commodity_signal(np.array([100.0] * 30)) == 50.0

    def test_with_volume(self):
        np.random.seed(42)
        close = np.linspace(100, 120, 100)
        volume = np.random.randint(1_000_000, 50_000_000, 100).astype(np.float64)
        result = commodity_signal(close, volume)
        assert 0 <= result <= 100

    def test_returns_float(self):
        close = np.linspace(100, 120, 60)
        result = commodity_signal(close)
        assert isinstance(result, float)


# =============================================================================
# Multi-Asset Engine Tests
# =============================================================================


class TestMultiAssetScan:
    """Tests for scan_universe with asset_class parameter."""

    def _make_mock_df(self, n: int = 252) -> pd.DataFrame:
        """Create a mock price DataFrame."""
        np.random.seed(42)
        dates = pd.bdate_range(start="2023-01-02", periods=n)
        returns = np.random.normal(0.0005, 0.015, n)
        close = 100 * np.cumprod(1 + returns)
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 50_000_000, n),
                "Adj Close": close,
            },
            index=dates,
        )

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_crypto_asset_class(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {}

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_universe(
                symbols=["BTC/USDT", "ETH/USDT"],
                signals=[SignalType.MOMENTUM],
                max_workers=1,
                asset_class=AssetClass.CRYPTO,
            )

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert r.sector == "Crypto"

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_forex_asset_class(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {}

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_universe(
                symbols=["EUR/USD"],
                signals=[SignalType.MOMENTUM],
                max_workers=1,
                asset_class=AssetClass.FOREX,
            )

        assert len(results) == 1
        assert results[0].sector == "Forex"

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_commodity_asset_class(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {}

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_universe(
                symbols=["GOLD"],
                signals=[SignalType.MOMENTUM],
                max_workers=1,
                asset_class=AssetClass.COMMODITY,
            )

        assert len(results) == 1
        assert results[0].sector == "Commodity"

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_value_signal_returns_50_for_non_equity(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {}

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_universe(
                symbols=["BTC/USDT"],
                signals=[SignalType.VALUE],
                max_workers=1,
                asset_class=AssetClass.CRYPTO,
            )

        assert len(results) == 1
        assert results[0].details["value"] == 50.0

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_quality_signal_returns_50_for_non_equity(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_universe

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {}

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_universe(
                symbols=["EUR/USD"],
                signals=[SignalType.QUALITY],
                max_workers=1,
                asset_class=AssetClass.FOREX,
            )

        assert len(results) == 1
        assert results[0].details["quality"] == 50.0

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_scan_multi_asset(self, mock_prices, mock_info):
        from quant_tools.scanner.engine import scan_multi_asset

        mock_df = self._make_mock_df()
        mock_prices.return_value = mock_df
        mock_info.return_value = {"name": "Test", "sector": "Technology"}

        assets = [
            ("AAPL", AssetClass.EQUITY),
            ("BTC/USDT", AssetClass.CRYPTO),
            ("EUR/USD", AssetClass.FOREX),
        ]

        mock_registry = MagicMock()
        mock_registry.fetch_ohlcv.return_value = mock_df
        with patch(
            "quant_tools.data.providers.registry.ProviderRegistry",
            return_value=mock_registry,
        ):
            results = scan_multi_asset(
                assets=assets,
                signals=[SignalType.MOMENTUM],
                max_workers=1,
            )

        assert len(results) == 3
        sectors = {r.sector for r in results}
        assert "Crypto" in sectors
        assert "Forex" in sectors

    @patch("quant_tools.scanner.engine.fetch_ticker_info")
    @patch("quant_tools.scanner.engine.fetch_prices")
    def test_equity_default_unchanged(self, mock_prices, mock_info):
        """Verify that not passing asset_class preserves existing equity behavior."""
        from quant_tools.scanner.engine import scan_universe

        mock_prices.return_value = self._make_mock_df()
        mock_info.return_value = {
            "name": "Apple", "sector": "Technology",
            "pe_ratio": 25.0, "pb_ratio": 8.0,
            "roe": 0.15, "profit_margin": 0.25, "debt_to_equity": 1.5,
        }

        results = scan_universe(
            symbols=["AAPL"],
            signals=None,
            max_workers=1,
        )

        assert len(results) == 1
        details = results[0].details
        for st in SignalType:
            assert st.value in details
        assert "breakdown" in details
