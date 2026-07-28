"""Tests for enhanced indicators (~25 tests).

Mocks pandas_ta to avoid dependency on the library during tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Helpers
# =============================================================================


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    """Build a sample OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-02", periods=n)
    close = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


def _mock_pandas_ta():
    """Create a mock pandas_ta module with realistic return values."""
    mock_ta = MagicMock()

    # RSI returns a Series
    mock_ta.rsi.return_value = pd.Series(np.random.uniform(20, 80, 100), name="RSI_14")

    # MACD returns a DataFrame with 3 columns
    mock_ta.macd.return_value = pd.DataFrame({
        "MACD_12_26_9": np.random.randn(100),
        "MACDh_12_26_9": np.random.randn(100),
        "MACDs_12_26_9": np.random.randn(100),
    })

    # Bollinger Bands returns a DataFrame
    mock_ta.bbands.return_value = pd.DataFrame({
        "BBL_20_2.0": np.random.uniform(95, 100, 100),
        "BBM_20_2.0": np.random.uniform(99, 101, 100),
        "BBU_20_2.0": np.random.uniform(100, 105, 100),
    })

    # ADX returns a DataFrame
    mock_ta.adx.return_value = pd.DataFrame({
        "ADX_14": np.random.uniform(10, 50, 100),
        "DMP_14": np.random.uniform(10, 40, 100),
        "DMN_14": np.random.uniform(10, 40, 100),
    })

    # OBV returns a Series
    mock_ta.obv.return_value = pd.Series(np.random.randint(0, 1_000_000, 100), name="OBV")

    # Stochastic returns a DataFrame
    mock_ta.stoch.return_value = pd.DataFrame({
        "STOCHk_14_3_3": np.random.uniform(0, 100, 100),
        "STOCHd_14_3_3": np.random.uniform(0, 100, 100),
    })

    # CCI returns a Series
    mock_ta.cci.return_value = pd.Series(np.random.uniform(-200, 200, 100), name="CCI_20")

    # Williams %R returns a Series
    mock_ta.willr.return_value = pd.Series(np.random.uniform(-100, 0, 100), name="WILLR_14")

    # ATR returns a Series
    mock_ta.atr.return_value = pd.Series(np.random.uniform(0.5, 3.0, 100), name="ATR_14")

    return mock_ta


# =============================================================================
# build_indicator_set Tests
# =============================================================================


class TestBuildIndicatorSet:
    """Tests for build_indicator_set function."""

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_default_indicators_added(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df)
        # Should have more columns than the original
        assert len(result.columns) > len(df.columns)

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_rsi_column_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["rsi_14"])
        assert "RSI_14" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_macd_columns_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["macd_12_26_9"])
        macd_cols = [c for c in result.columns if "MACD" in c]
        assert len(macd_cols) >= 1

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_bbands_columns_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["bbands_20"])
        bb_cols = [c for c in result.columns if "BB" in c]
        assert len(bb_cols) >= 1

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_obv_column_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["obv"])
        assert "OBV" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_stoch_columns_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["stoch_14_3_3"])
        stoch_cols = [c for c in result.columns if "STOCH" in c]
        assert len(stoch_cols) >= 1

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_cci_column_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["cci_20"])
        assert "CCI_20" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_willr_column_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["willr_14"])
        assert "WILLR_14" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_atr_column_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["atr_14"])
        assert "ATR_14" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_adx_columns_present(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["adx_14"])
        adx_cols = [c for c in result.columns if "ADX" in c or "DM" in c]
        assert len(adx_cols) >= 1

    def test_empty_dataframe_returned_unchanged(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = pd.DataFrame()
        result = build_indicator_set(df)
        assert result.empty

    def test_none_dataframe_returned_as_none(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        result = build_indicator_set(None)
        assert result is None

    def test_short_dataframe_returned_unchanged(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv(1)
        result = build_indicator_set(df)
        assert len(result) == 1

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_original_columns_preserved(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_custom_indicator_list(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df, indicators=["rsi_14", "obv"])
        assert "RSI_14" in result.columns
        assert "OBV" in result.columns

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_does_not_modify_input(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        original_cols = list(df.columns)
        build_indicator_set(df)
        assert list(df.columns) == original_cols

    def test_missing_pandas_ta_returns_unchanged(self):
        """When pandas_ta is not installed, return df without indicators."""
        import importlib
        import sys

        # Temporarily remove pandas_ta from sys.modules
        saved = sys.modules.pop("pandas_ta", None)
        # Also prevent import from succeeding
        with patch.dict("sys.modules", {"pandas_ta": None}):
            # Force reimport
            if "quant_tools.data.indicators_enhanced" in sys.modules:
                del sys.modules["quant_tools.data.indicators_enhanced"]
            from quant_tools.data.indicators_enhanced import build_indicator_set

            df = _sample_ohlcv()
            result = build_indicator_set(df)
            # Should return with same columns (no indicators added)
            assert len(result.columns) == len(df.columns)

        # Restore
        if saved is not None:
            sys.modules["pandas_ta"] = saved

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_dataframe_row_count_unchanged(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df)
        assert len(result) == len(df)

    @patch.dict("sys.modules", {"pandas_ta": _mock_pandas_ta()})
    def test_index_preserved(self):
        from quant_tools.data.indicators_enhanced import build_indicator_set

        df = _sample_ohlcv()
        result = build_indicator_set(df)
        assert result.index.equals(df.index)


# =============================================================================
# _parse_param Tests
# =============================================================================


class TestParseParam:
    """Tests for the _parse_param helper."""

    def test_parse_rsi_14(self):
        from quant_tools.data.indicators_enhanced import _parse_param

        assert _parse_param("rsi_14", "rsi", 14) == 14

    def test_parse_rsi_default(self):
        from quant_tools.data.indicators_enhanced import _parse_param

        assert _parse_param("rsi", "rsi", 14) == 14

    def test_parse_cci_20(self):
        from quant_tools.data.indicators_enhanced import _parse_param

        assert _parse_param("cci_20", "cci", 20) == 20

    def test_parse_atr_10(self):
        from quant_tools.data.indicators_enhanced import _parse_param

        assert _parse_param("atr_10", "atr", 14) == 10

    def test_parse_non_numeric_returns_default(self):
        from quant_tools.data.indicators_enhanced import _parse_param

        assert _parse_param("rsi_abc", "rsi", 14) == 14


# =============================================================================
# Universe Updates Tests
# =============================================================================


class TestUniverseUpdates:
    """Tests for the universe.py multi-asset additions."""

    def test_crypto_universe_exists(self):
        from quant_tools.data.universe import CRYPTO_UNIVERSE

        assert "BTC/USDT" in CRYPTO_UNIVERSE
        assert len(CRYPTO_UNIVERSE) == 20

    def test_forex_universe_exists(self):
        from quant_tools.data.universe import FOREX_UNIVERSE

        assert "EUR/USD" in FOREX_UNIVERSE
        assert len(FOREX_UNIVERSE) == 8

    def test_commodity_universe_exists(self):
        from quant_tools.data.universe import COMMODITY_UNIVERSE

        assert "GC=F" in COMMODITY_UNIVERSE
        assert len(COMMODITY_UNIVERSE) == 21

    def test_get_universe_crypto(self):
        from quant_tools.data.universe import get_universe

        symbols = get_universe(asset_class="crypto")
        assert "BTC/USDT" in symbols

    def test_get_universe_forex(self):
        from quant_tools.data.universe import get_universe

        symbols = get_universe(asset_class="forex")
        assert "EUR/USD" in symbols

    def test_get_universe_commodity(self):
        from quant_tools.data.universe import get_universe

        symbols = get_universe(asset_class="commodity")
        assert "GC=F" in symbols

    def test_get_universe_invalid_asset_class(self):
        from quant_tools.data.universe import get_universe

        with pytest.raises(ValueError, match="Unknown asset_class"):
            get_universe(asset_class="invalid")

    def test_get_universe_sp500_still_works(self):
        from quant_tools.data.universe import get_universe

        symbols = get_universe("sp500")
        assert "AAPL" in symbols
