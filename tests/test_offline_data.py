"""Tests for the offline market data generator.

These call the generator directly rather than going through ``activate()``, so
they do not fight the session-scoped yfinance guard in ``conftest.py``.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from quant_tools.testing import offline


class TestGeneratePrices:
    """Shape and invariants of the generated OHLCV frame."""

    def test_columns_match_yfinance_layout(self) -> None:
        df = offline.generate_prices("AAPL", period="1mo")
        assert list(df.columns) == offline.OHLCV_COLUMNS

    def test_deterministic_for_the_same_symbol(self) -> None:
        first = offline.generate_prices("AAPL", start="2025-01-01", end="2025-06-30")
        second = offline.generate_prices("AAPL", start="2025-01-01", end="2025-06-30")
        pd.testing.assert_frame_equal(first, second)

    def test_different_symbols_differ(self) -> None:
        aapl = offline.generate_prices("AAPL", period="6mo")
        msft = offline.generate_prices("MSFT", period="6mo")
        assert not aapl["Close"].equals(msft["Close"])

    def test_ohlc_invariants_hold(self) -> None:
        df = offline.generate_prices("NVDA", period="1y")
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
        assert (df["Low"] > 0).all()
        assert (df["Volume"] > 0).all()

    def test_explicit_range_is_respected(self) -> None:
        df = offline.generate_prices("TSLA", start="2025-03-03", end="2025-03-07")
        assert df.index.min() >= pd.Timestamp("2025-03-03")
        assert df.index.max() <= pd.Timestamp("2025-03-07")
        assert len(df) == 5  # one trading week

    @pytest.mark.parametrize(
        "period,low,high",
        [("5d", 3, 8), ("1mo", 18, 26), ("1y", 240, 265), ("2y", 480, 530)],
    )
    def test_period_lengths_are_plausible(self, period: str, low: int, high: int) -> None:
        assert low <= len(offline.generate_prices("SPY", period=period)) <= high

    def test_empty_range_returns_empty_frame(self) -> None:
        df = offline.generate_prices("AAPL", start="2025-01-10", end="2025-01-05")
        assert df.empty
        assert list(df.columns) == offline.OHLCV_COLUMNS


class TestGenerateInfo:
    """Ticker metadata shaped the way ``fetch_ticker_info`` reads it."""

    def test_has_the_fields_the_app_reads(self) -> None:
        info = offline.generate_info("AAPL")
        for field in ("longName", "sector", "marketCap", "trailingPE", "beta"):
            assert field in info

    def test_deterministic(self) -> None:
        assert offline.generate_info("AAPL") == offline.generate_info("AAPL")

    def test_52_week_band_contains_the_last_price(self) -> None:
        info = offline.generate_info("MSFT")
        assert info["fiftyTwoWeekLow"] <= info["regularMarketPrice"]
        assert info["regularMarketPrice"] <= info["fiftyTwoWeekHigh"]


class TestDownloadShim:
    """The ``yf.download`` stand-in, single and multi symbol."""

    def test_single_symbol_is_flat(self) -> None:
        df = offline._download("AAPL", period="1mo")
        assert list(df.columns) == offline.OHLCV_COLUMNS

    def test_multiple_symbols_are_multiindex(self) -> None:
        df = offline._download("AAPL MSFT", period="1mo")
        assert isinstance(df.columns, pd.MultiIndex)

    def test_accepts_a_list(self) -> None:
        df = offline._download(["AAPL", "MSFT"], period="1mo")
        assert isinstance(df.columns, pd.MultiIndex)


class TestActivation:
    """The env-var gate. Never fake data unless explicitly asked."""

    def test_inactive_without_the_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(offline.ENV_VAR, raising=False)
        assert offline.is_active() is False
        assert offline.activate() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_truthy_values_activate(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(offline.ENV_VAR, value)
        assert offline.is_active() is True

    def test_other_values_do_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(offline.ENV_VAR, "0")
        assert offline.is_active() is False


class TestNullCache:
    """Offline mode has to bypass the on-disk price cache.

    Not calling ``activate()`` here on purpose: it patches yfinance globally and
    would outlive this test, blunting the session-scoped network guard.
    """

    def test_reads_always_miss(self) -> None:
        cache = offline._NullCache()
        assert cache.get_prices("AAPL", "2025-01-01", "2025-06-30") is None
        assert cache.get_metadata("AAPL") is None

    def test_writes_are_dropped(self) -> None:
        cache = offline._NullCache()
        assert cache.store_prices("AAPL", [{"date": "2025-01-02"}]) is None
        assert cache.store_metadata("AAPL", {"sector": "Technology"}) is None

    def test_covers_what_market_data_calls(self) -> None:
        from quant_tools.data import market_data

        used = {"get_prices", "store_prices", "get_metadata", "store_metadata"}
        assert used <= set(dir(offline._NullCache))
        assert used <= set(dir(type(market_data._cache)))


class TestOfflineTicker:
    """The ``yf.Ticker`` stand-in."""

    def test_news_matches_the_shape_fetch_news_parses(self) -> None:
        items = offline._OfflineTicker("AAPL").news
        assert items
        content = items[0]["content"]
        assert content["title"]
        assert content["provider"]["displayName"]
        assert content["canonicalUrl"]["url"]

    def test_history_returns_prices(self) -> None:
        assert not offline._OfflineTicker("AAPL").history(period="1mo").empty


class TestOfflineCrypto:
    """Crypto has to be covered too, not just yfinance.

    Called directly rather than through ``activate()``, which patches the
    provider class globally and would leak into other tests.
    """

    def test_columns_match_the_ccxt_provider(self) -> None:
        df = offline._offline_crypto_ohlcv(None, "BTC/USD", period="1mo")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_index_is_dated(self) -> None:
        df = offline._offline_crypto_ohlcv(None, "ETH/USD", period="1mo")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_same_pair_gives_the_same_series(self) -> None:
        """Assertions on rendered prices need the generator to be stable."""
        first = offline._offline_crypto_ohlcv(None, "SOL/USD", period="1mo")
        second = offline._offline_crypto_ohlcv(None, "SOL/USD", period="1mo")
        pd.testing.assert_frame_equal(first, second)

    def test_different_pairs_differ(self) -> None:
        btc = offline._offline_crypto_ohlcv(None, "BTC/USD", period="1mo")
        eth = offline._offline_crypto_ohlcv(None, "ETH/USD", period="1mo")
        assert not btc["Close"].equals(eth["Close"])

    def test_activate_patches_the_crypto_provider(self) -> None:
        """Without this the dashboard's crypto scans still call a live exchange.

        activate() only replaced yfinance, so QUANT_TOOLS_OFFLINE left ccxt
        talking to Kraken, which is the situation the module exists to avoid.

        Run in a subprocess because activate() patches yfinance, the provider
        class and the price cache all at module scope. Restoring them by hand
        is the sort of thing that works until it silently does not, and a
        half-restored patch leaks into whatever test runs next.
        """
        code = (
            "from quant_tools.testing import offline;"
            "from quant_tools.data.providers.crypto import CryptoProvider;"
            "assert offline.activate(force=True) is True;"
            "assert CryptoProvider.fetch_ohlcv is offline._offline_crypto_ohlcv;"
            "df = CryptoProvider().fetch_ohlcv('BTC/USD', period='1mo');"
            "assert not df.empty;"
            "print('PATCHED')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert "PATCHED" in result.stdout
