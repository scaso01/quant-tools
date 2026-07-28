"""Tests for data/indicators.py."""

from __future__ import annotations

import numpy as np
import pytest

from quant_tools.data.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    volume_ratio,
)
from quant_tools.errors import ComputeError


class TestSMA:
    """Test Simple Moving Average."""

    def test_basic_sma(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(prices, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_period_1(self):
        prices = np.array([10.0, 20.0, 30.0])
        result = sma(prices, 1)
        np.testing.assert_array_almost_equal(result, prices)

    def test_sma_period_equals_length(self):
        prices = np.array([1.0, 2.0, 3.0])
        result = sma(prices, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)

    def test_sma_period_exceeds_length(self):
        prices = np.array([1.0, 2.0])
        result = sma(prices, 5)
        assert all(np.isnan(result))

    def test_sma_invalid_period(self):
        with pytest.raises(ComputeError):
            sma(np.array([1.0]), 0)

    def test_sma_realistic(self, sample_close_array):
        result = sma(sample_close_array, 20)
        # First 19 should be NaN
        assert all(np.isnan(result[:19]))
        # Rest should be valid
        assert not np.isnan(result[19])
        # SMA should be smooth (less volatile than raw prices)
        valid = result[~np.isnan(result)]
        assert np.std(valid) < np.std(sample_close_array)


class TestEMA:
    """Test Exponential Moving Average."""

    def test_basic_ema(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(prices, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)  # SMA seed
        assert result[3] > result[2]  # Should increase

    def test_ema_reacts_faster_than_sma(self, sample_close_array):
        ema_result = ema(sample_close_array, 20)
        sma_result = sma(sample_close_array, 20)
        # EMA reacts faster — larger absolute differences from price
        valid_mask = ~np.isnan(ema_result) & ~np.isnan(sma_result)
        ema_diff = np.abs(sample_close_array[valid_mask] - ema_result[valid_mask])
        sma_diff = np.abs(sample_close_array[valid_mask] - sma_result[valid_mask])
        # EMA should generally be closer to current price
        assert np.mean(ema_diff) <= np.mean(sma_diff) + 1.0

    def test_ema_invalid_period(self):
        with pytest.raises(ComputeError):
            ema(np.array([1.0]), 0)


class TestRSI:
    """Test Relative Strength Index."""

    def test_rsi_range(self, sample_close_array):
        result = rsi(sample_close_array, 14)
        valid = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid)

    def test_rsi_all_up(self):
        prices = np.arange(1.0, 20.0)
        result = rsi(prices, 5)
        valid = result[~np.isnan(result)]
        assert all(v > 90 for v in valid)  # Should be near 100

    def test_rsi_all_down(self):
        prices = np.arange(20.0, 1.0, -1.0)
        result = rsi(prices, 5)
        valid = result[~np.isnan(result)]
        assert all(v < 10 for v in valid)  # Should be near 0

    def test_rsi_invalid_period(self):
        with pytest.raises(ComputeError):
            rsi(np.array([1.0]), 0)

    def test_rsi_insufficient_data(self):
        prices = np.array([100.0, 101.0])
        result = rsi(prices, 14)
        assert all(np.isnan(result))


class TestBollingerBands:
    """Test Bollinger Bands."""

    def test_bands_relationship(self, sample_close_array):
        upper, middle, lower = bollinger_bands(sample_close_array, 20, 2.0)
        valid_mask = ~np.isnan(upper) & ~np.isnan(lower)
        assert all(upper[valid_mask] >= middle[valid_mask])
        assert all(middle[valid_mask] >= lower[valid_mask])

    def test_bands_width(self):
        # Constant prices should have zero bandwidth
        prices = np.full(30, 100.0)
        upper, middle, lower = bollinger_bands(prices, 20, 2.0)
        valid_mask = ~np.isnan(upper)
        np.testing.assert_array_almost_equal(
            upper[valid_mask], lower[valid_mask]
        )

    def test_bands_invalid_period(self):
        with pytest.raises(ComputeError):
            bollinger_bands(np.array([1.0]), 0)

    def test_middle_equals_sma(self, sample_close_array):
        _, middle, _ = bollinger_bands(sample_close_array, 20)
        sma_20 = sma(sample_close_array, 20)
        valid = ~np.isnan(middle) & ~np.isnan(sma_20)
        np.testing.assert_array_almost_equal(middle[valid], sma_20[valid])


class TestMACD:
    """Test MACD."""

    def test_macd_components(self, sample_close_array):
        macd_line, signal_line, histogram = macd(sample_close_array)
        # Histogram = MACD - Signal
        valid = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        np.testing.assert_array_almost_equal(
            histogram[valid], macd_line[valid] - signal_line[valid]
        )

    def test_macd_trending_up(self):
        prices = np.arange(1.0, 100.0)
        macd_line, _, _ = macd(prices, 5, 10, 3)
        valid = macd_line[~np.isnan(macd_line)]
        # In uptrend, MACD should be positive eventually
        assert valid[-1] > 0


class TestATR:
    """Test Average True Range."""

    def test_atr_positive(self, sample_prices):
        high = sample_prices["High"].values
        low = sample_prices["Low"].values
        close = sample_prices["Close"].values
        result = atr(high, low, close, 14)
        valid = result[~np.isnan(result)]
        assert all(v > 0 for v in valid)

    def test_atr_invalid_period(self):
        with pytest.raises(ComputeError):
            atr(np.array([1.0]), np.array([1.0]), np.array([1.0]), 0)


class TestVolumeRatio:
    """Test volume ratio."""

    def test_volume_ratio_basic(self):
        volumes = np.array([100.0] * 25)
        volumes[-1] = 200.0
        result = volume_ratio(volumes, 20)
        assert result[-1] == pytest.approx(2.0, rel=0.1)

    def test_volume_ratio_realistic(self, sample_prices):
        vol = sample_prices["Volume"].values.astype(np.float64)
        result = volume_ratio(vol, 20)
        valid = result[~np.isnan(result)]
        # Average ratio should be around 1.0
        assert 0.5 < np.mean(valid) < 2.0
