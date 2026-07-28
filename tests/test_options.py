"""Tests for OptionsProvider.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from quant_tools.data.providers.options import (
    OptionsChain,
    OptionsContract,
    OptionsProvider,
)
from quant_tools.errors import DataError, NetworkError


def _make_contract(
    strike: float = 150.0,
    expiration: str = "2026-04-17",
    option_type: str = "call",
    last_price: float = 5.0,
    bid: float = 4.5,
    ask: float = 5.5,
    volume: int = 100,
    open_interest: int = 500,
    implied_volatility: float = 0.3,
) -> OptionsContract:
    """Helper to create a test OptionsContract."""
    return OptionsContract(
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        last_price=last_price,
        bid=bid,
        ask=ask,
        volume=volume,
        open_interest=open_interest,
        implied_volatility=implied_volatility,
    )


def _make_yf_chain_df(option_type: str = "call", count: int = 3) -> pd.DataFrame:
    """Build a mock yfinance options DataFrame."""
    rows = []
    for i in range(count):
        rows.append({
            "strike": 140.0 + i * 5,
            "lastPrice": 5.0 - i * 1.0,
            "bid": 4.5 - i * 1.0,
            "ask": 5.5 - i * 1.0,
            "volume": 100 + i * 50,
            "openInterest": 500 + i * 100,
            "impliedVolatility": 0.3 + i * 0.05,
        })
    return pd.DataFrame(rows)


class TestFetchChain:
    """Tests for OptionsProvider.fetch_chain."""

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_fetch_chain_success(self, mock_breaker):
        """Mock yfinance and verify OptionsChain structure."""
        mock_breaker.is_open = False

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-04-17", "2026-05-15")

        calls_df = _make_yf_chain_df("call", 3)
        puts_df = _make_yf_chain_df("put", 2)
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_chain.puts = puts_df
        mock_ticker.option_chain.return_value = mock_chain

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = OptionsProvider()
            result = provider.fetch_chain("AAPL")

        assert isinstance(result, OptionsChain)
        assert result.symbol == "AAPL"
        assert result.expiration == "2026-04-17"
        assert len(result.calls) == 3
        assert len(result.puts) == 2
        assert all(isinstance(c, OptionsContract) for c in result.calls)
        mock_breaker.record_success.assert_called_once()

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_fetch_chain_specific_expiration(self, mock_breaker):
        """Fetch chain for a specific expiration date."""
        mock_breaker.is_open = False

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-04-17", "2026-05-15")

        calls_df = _make_yf_chain_df("call", 1)
        puts_df = _make_yf_chain_df("put", 1)
        mock_chain = MagicMock()
        mock_chain.calls = calls_df
        mock_chain.puts = puts_df
        mock_ticker.option_chain.return_value = mock_chain

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = OptionsProvider()
            result = provider.fetch_chain("AAPL", expiration="2026-05-15")

        assert result.expiration == "2026-05-15"
        mock_ticker.option_chain.assert_called_once_with("2026-05-15")

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_empty_chain(self, mock_breaker):
        """No expirations raises DataError."""
        mock_breaker.is_open = False

        mock_ticker = MagicMock()
        mock_ticker.options = ()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = OptionsProvider()
            with pytest.raises(DataError, match="No options expirations"):
                provider.fetch_chain("ZZZZZ")

        mock_breaker.record_failure.assert_called_once()

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_circuit_breaker_open(self, mock_breaker):
        """NetworkError when breaker is open."""
        mock_breaker.is_open = True

        provider = OptionsProvider()
        with pytest.raises(NetworkError, match="circuit breaker is open"):
            provider.fetch_chain("AAPL")


class TestFetchExpirations:
    """Tests for OptionsProvider.fetch_expirations."""

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_fetch_expirations(self, mock_breaker):
        """Mock yfinance.Ticker.options."""
        mock_breaker.is_open = False

        mock_ticker = MagicMock()
        mock_ticker.options = ("2026-04-17", "2026-05-15", "2026-06-19")

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = OptionsProvider()
            result = provider.fetch_expirations("AAPL")

        assert result == ["2026-04-17", "2026-05-15", "2026-06-19"]
        mock_breaker.record_success.assert_called_once()

    @patch("quant_tools.data.providers.options.options_breaker")
    def test_library_not_available(self, mock_breaker):
        """Graceful error when yfinance not installed."""
        mock_breaker.is_open = False

        with patch.dict("sys.modules", {"yfinance": None}):
            # Force reimport to trigger ImportError
            import importlib
            import quant_tools.data.providers.options as opts_mod
            # Manually simulate what happens in fetch_expirations
            provider = OptionsProvider()
            # Patch the import inside the method
            with patch("builtins.__import__", side_effect=ImportError("No module named 'yfinance'")):
                with pytest.raises(DataError, match="yfinance is required"):
                    provider.fetch_expirations("AAPL")


class TestPutCallRatio:
    """Tests for OptionsProvider.calculate_put_call_ratio."""

    def test_put_call_ratio_bullish(self):
        """Volume skewed to calls = ratio < 1."""
        chain = OptionsChain(
            symbol="AAPL",
            expiration="2026-04-17",
            calls=[_make_contract(volume=1000, option_type="call")],
            puts=[_make_contract(volume=300, option_type="put")],
        )
        provider = OptionsProvider()
        ratio = provider.calculate_put_call_ratio(chain)
        assert ratio < 1.0
        assert ratio == pytest.approx(0.3)

    def test_put_call_ratio_bearish(self):
        """Volume skewed to puts = ratio > 1."""
        chain = OptionsChain(
            symbol="AAPL",
            expiration="2026-04-17",
            calls=[_make_contract(volume=200, option_type="call")],
            puts=[_make_contract(volume=800, option_type="put")],
        )
        provider = OptionsProvider()
        ratio = provider.calculate_put_call_ratio(chain)
        assert ratio > 1.0
        assert ratio == pytest.approx(4.0)

    def test_put_call_ratio_no_volume(self):
        """Zero volume raises DataError."""
        chain = OptionsChain(
            symbol="AAPL",
            expiration="2026-04-17",
            calls=[_make_contract(volume=0, option_type="call")],
            puts=[_make_contract(volume=0, option_type="put")],
        )
        provider = OptionsProvider()
        with pytest.raises(DataError, match="No volume data"):
            provider.calculate_put_call_ratio(chain)


class TestMaxPain:
    """Tests for OptionsProvider.calculate_max_pain."""

    def test_max_pain_calculation(self):
        """Known strike data with predictable max pain."""
        # Create chain with known data:
        # Strikes: 140, 150, 160
        # Call OI: heavy at 140 (1000), light at 150 (100), none at 160
        # Put OI: none at 140, light at 150 (100), heavy at 160 (1000)
        # Max pain should be around 150 (middle), minimizing total pain
        calls = [
            _make_contract(strike=140.0, open_interest=1000, option_type="call"),
            _make_contract(strike=150.0, open_interest=100, option_type="call"),
            _make_contract(strike=160.0, open_interest=10, option_type="call"),
        ]
        puts = [
            _make_contract(strike=140.0, open_interest=10, option_type="put"),
            _make_contract(strike=150.0, open_interest=100, option_type="put"),
            _make_contract(strike=160.0, open_interest=1000, option_type="put"),
        ]
        chain = OptionsChain(
            symbol="AAPL", expiration="2026-04-17", calls=calls, puts=puts,
        )
        provider = OptionsProvider()
        max_pain = provider.calculate_max_pain(chain)

        # Max pain should be 150 — balanced point
        assert max_pain == 150.0

    def test_max_pain_empty_chain(self):
        """Empty chain raises DataError."""
        chain = OptionsChain(symbol="AAPL", expiration="2026-04-17")
        provider = OptionsProvider()
        with pytest.raises(DataError, match="No contracts"):
            provider.calculate_max_pain(chain)


class TestUnusualVolume:
    """Tests for OptionsProvider.scan_unusual_volume."""

    def test_unusual_volume_detection(self):
        """Contracts above threshold are flagged."""
        chain = OptionsChain(
            symbol="AAPL",
            expiration="2026-04-17",
            calls=[
                _make_contract(strike=150.0, volume=500, open_interest=100),  # 5.0 ratio
                _make_contract(strike=155.0, volume=50, open_interest=100),   # 0.5 ratio
            ],
            puts=[
                _make_contract(strike=145.0, volume=300, open_interest=100, option_type="put"),  # 3.0
            ],
        )
        provider = OptionsProvider()
        unusual = provider.scan_unusual_volume(chain, threshold=2.0)

        assert len(unusual) == 2
        assert unusual[0]["vol_oi_ratio"] == 5.0
        assert unusual[0]["strike"] == 150.0
        assert unusual[1]["vol_oi_ratio"] == 3.0

    def test_unusual_volume_none(self):
        """No unusual activity returns empty list."""
        chain = OptionsChain(
            symbol="AAPL",
            expiration="2026-04-17",
            calls=[_make_contract(volume=10, open_interest=500)],
            puts=[_make_contract(volume=5, open_interest=500, option_type="put")],
        )
        provider = OptionsProvider()
        unusual = provider.scan_unusual_volume(chain, threshold=2.0)
        assert unusual == []


class TestOptionsContractFrozen:
    """Test dataclass immutability."""

    def test_options_contract_frozen(self):
        """OptionsContract is frozen — cannot mutate fields."""
        contract = _make_contract()
        with pytest.raises(AttributeError):
            contract.strike = 999.0  # type: ignore[misc]
