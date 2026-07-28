"""Tests for quant_tools.predict.features — FeatureEngineer (~45 tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_tools.models import AssetClass, FeatureSet
from quant_tools.predict.features import FeatureEngineer, _bars_per_day


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture
def short_prices() -> pd.DataFrame:
    """Very short DataFrame (30 rows) to test edge cases."""
    np.random.seed(99)
    n = 30
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════
# build_features integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFeatures:
    def test_returns_feature_set(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices, symbol="AAPL")
        assert isinstance(fs, FeatureSet)

    def test_symbol_set(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices, symbol="MSFT")
        assert fs.symbol == "MSFT"

    def test_feature_count_at_least_50(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert len(fs.feature_names) >= 50

    def test_feature_names_match_columns(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert list(fs.features_df.columns) == fs.feature_names

    def test_no_nan_in_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert fs.features_df.notna().all().all()

    def test_no_nan_in_target(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert np.all(np.isfinite(fs.target))

    def test_target_length_matches_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert len(fs.target) == len(fs.features_df)

    def test_target_is_forward_returns(self, engineer, sample_prices):
        """Target should be percentage forward returns."""
        fs = engineer.build_features(sample_prices, horizon_days=5)
        # Forward returns should be small-ish numbers, not prices
        assert np.abs(fs.target).max() < 1.0  # less than 100%

    def test_horizon_days_changes_target(self, engineer, sample_prices):
        fs5 = engineer.build_features(sample_prices, horizon_days=5)
        fs20 = engineer.build_features(sample_prices, horizon_days=20)
        # Different horizons produce different targets
        min_len = min(len(fs5.target), len(fs20.target))
        assert not np.allclose(fs5.target[:min_len], fs20.target[:min_len])

    def test_ohlcv_columns_excluded(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        excluded = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        assert excluded.isdisjoint(set(fs.feature_names))

    def test_rows_fewer_than_input(self, engineer, sample_prices):
        """NaN warmup rows get dropped so output < input."""
        fs = engineer.build_features(sample_prices)
        assert len(fs.features_df) < len(sample_prices)

    def test_asset_class_param_accepted(self, engineer, sample_prices):
        """asset_class param doesn't crash (future-proofing)."""
        fs = engineer.build_features(
            sample_prices, asset_class=AssetClass.CRYPTO
        )
        assert isinstance(fs, FeatureSet)

    def test_short_dataframe_produces_some_features(self, engineer, short_prices):
        """Even short data should produce a FeatureSet (possibly with few rows)."""
        fs = engineer.build_features(short_prices)
        assert isinstance(fs, FeatureSet)

    def test_features_df_is_dataframe(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert isinstance(fs.features_df, pd.DataFrame)

    def test_target_is_numpy_array(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert isinstance(fs.target, np.ndarray)


# ═══════════════════════════════════════════════════════════════════════
# Feature presence tests
# ═══════════════════════════════════════════════════════════════════════


class TestFeaturePresence:
    def test_rsi_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for p in [7, 14, 21]:
            assert f"rsi_{p}" in fs.feature_names

    def test_macd_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "macd" in fs.feature_names
        assert "macd_signal" in fs.feature_names
        assert "macd_hist" in fs.feature_names

    def test_bollinger_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "bb_upper_20" in fs.feature_names
        assert "bb_mid_20" in fs.feature_names
        assert "bb_lower_20" in fs.feature_names
        assert "bb_pct_20" in fs.feature_names

    def test_adx_feature(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "adx_14" in fs.feature_names

    def test_atr_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "atr_7" in fs.feature_names
        assert "atr_14" in fs.feature_names

    def test_stochastic_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "stoch_k" in fs.feature_names
        assert "stoch_d" in fs.feature_names

    def test_cci_feature(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "cci_20" in fs.feature_names

    def test_williams_r_feature(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "willr_14" in fs.feature_names

    def test_return_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for p in [1, 5, 10, 20, 60]:
            assert f"return_{p}d" in fs.feature_names

    def test_volatility_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for w in [5, 20, 60]:
            assert f"volatility_{w}d" in fs.feature_names

    def test_ma_ratio_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for p in [10, 20, 50, 200]:
            assert f"close_sma{p}_ratio" in fs.feature_names

    def test_ema_ratio_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for p in [12, 26]:
            assert f"close_ema{p}_ratio" in fs.feature_names

    def test_volume_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "volume_sma20_ratio" in fs.feature_names
        assert "obv" in fs.feature_names
        assert "obv_slope_5" in fs.feature_names
        assert "ad_line" in fs.feature_names
        assert "ad_slope_5" in fs.feature_names

    def test_calendar_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "day_of_week" in fs.feature_names
        assert "month" in fs.feature_names
        assert "quarter" in fs.feature_names

    def test_roc_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        for p in [5, 10, 20]:
            assert f"roc_{p}" in fs.feature_names

    def test_hl_range_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "hl_range" in fs.feature_names
        assert "hl_range_20d_avg" in fs.feature_names

    def test_additional_price_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "close_to_high" in fs.feature_names
        assert "close_to_low" in fs.feature_names
        assert "gap" in fs.feature_names

    def test_volume_roc_feature(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "volume_roc_5" in fs.feature_names

    def test_distribution_features(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "return_skew_20" in fs.feature_names
        assert "return_kurt_20" in fs.feature_names

    def test_mfi_feature(self, engineer, sample_prices):
        fs = engineer.build_features(sample_prices)
        assert "mfi_14" in fs.feature_names


# ═══════════════════════════════════════════════════════════════════════
# Individual indicator unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestRSI:
    def test_rsi_range(self, sample_prices):
        rsi = FeatureEngineer._rsi(sample_prices["Close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self, sample_prices):
        rsi = FeatureEngineer._rsi(sample_prices["Close"], 14)
        assert len(rsi) == len(sample_prices)

    def test_rsi_short_period(self, sample_prices):
        rsi = FeatureEngineer._rsi(sample_prices["Close"], 2)
        valid = rsi.dropna()
        assert len(valid) > 0


class TestMACD:
    def test_macd_returns_three_series(self, sample_prices):
        macd, signal, hist = FeatureEngineer._macd(sample_prices["Close"])
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_macd_hist_is_difference(self, sample_prices):
        macd, signal, hist = FeatureEngineer._macd(sample_prices["Close"])
        np.testing.assert_allclose(hist.values, (macd - signal).values)

    def test_macd_length(self, sample_prices):
        macd, _, _ = FeatureEngineer._macd(sample_prices["Close"])
        assert len(macd) == len(sample_prices)


class TestBollinger:
    def test_upper_above_lower(self, sample_prices):
        upper, mid, lower = FeatureEngineer._bollinger(
            sample_prices["Close"], 20
        )
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()

    def test_mid_is_sma(self, sample_prices):
        _, mid, _ = FeatureEngineer._bollinger(sample_prices["Close"], 20)
        sma = sample_prices["Close"].rolling(20).mean()
        valid = mid.dropna()
        np.testing.assert_allclose(valid.values, sma.dropna().values)

    def test_returns_three_series(self, sample_prices):
        result = FeatureEngineer._bollinger(sample_prices["Close"])
        assert len(result) == 3


class TestADX:
    def test_adx_non_negative(self, sample_prices):
        adx = FeatureEngineer._adx(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = adx.dropna()
        assert (valid >= 0).all()


class TestATR:
    def test_atr_positive(self, sample_prices):
        atr = FeatureEngineer._atr(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = atr.dropna()
        assert (valid > 0).all()


class TestStochastic:
    def test_k_range(self, sample_prices):
        k, d = FeatureEngineer._stochastic(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = k.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100.01).all()  # small epsilon

    def test_d_is_smoothed_k(self, sample_prices):
        k, d = FeatureEngineer._stochastic(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
            14,
            3,
        )
        expected_d = k.rolling(3).mean()
        valid_mask = d.notna() & expected_d.notna()
        np.testing.assert_allclose(
            d[valid_mask].values, expected_d[valid_mask].values
        )


class TestCCI:
    def test_cci_has_values(self, sample_prices):
        cci = FeatureEngineer._cci(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = cci.dropna()
        assert len(valid) > 0

    def test_cci_centered_around_zero(self, sample_prices):
        cci = FeatureEngineer._cci(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = cci.dropna()
        # Mean should be somewhat near zero for random walk data
        assert abs(valid.mean()) < 200


class TestWilliamsR:
    def test_range(self, sample_prices):
        wr = FeatureEngineer._williams_r(
            sample_prices["High"],
            sample_prices["Low"],
            sample_prices["Close"],
        )
        valid = wr.dropna()
        assert (valid <= 0.01).all()  # small epsilon
        assert (valid >= -100.01).all()


class TestOBV:
    def test_obv_length(self, sample_prices):
        obv = FeatureEngineer._obv(
            sample_prices["Close"], sample_prices["Volume"]
        )
        assert len(obv) == len(sample_prices)

    def test_obv_is_cumulative(self, sample_prices):
        obv = FeatureEngineer._obv(
            sample_prices["Close"], sample_prices["Volume"]
        )
        # OBV should not be constant (with varying prices and volumes)
        assert obv.std() > 0


# ═══════════════════════════════════════════════════════════════════════
# Intraday feature tests
# ═══════════════════════════════════════════════════════════════════════


class TestBarsPerDayMapping:
    def test_1h_returns_24(self):
        assert _bars_per_day("1h") == 24

    def test_4h_returns_6(self):
        assert _bars_per_day("4h") == 6

    def test_15m_returns_96(self):
        assert _bars_per_day("15m") == 96

    def test_1d_returns_1(self):
        assert _bars_per_day("1d") == 1

    def test_unknown_returns_1(self):
        assert _bars_per_day("unknown") == 1


class TestBuildFeaturesIntraday:
    @pytest.fixture
    def hourly_prices(self) -> pd.DataFrame:
        """Mock DataFrame with ~6000 rows simulating hourly bars.

        With bpd=24 the longest indicator window is SMA200 which needs
        200*24=4800 bars for warmup, so we generate 6000 rows to have
        enough non-NaN rows after warmup.
        """
        np.random.seed(123)
        n = 6000
        dates = pd.date_range(start="2023-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
        return pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.003,
                "Low": close * 0.997,
                "Close": close,
                "Volume": np.random.randint(100_000, 5_000_000, n).astype(float),
            },
            index=dates,
        )

    def test_build_features_intraday_1h(self, hourly_prices):
        engineer = FeatureEngineer()
        fs = engineer.build_features(hourly_prices, interval="1h")
        assert isinstance(fs, FeatureSet)
        assert len(fs.feature_names) >= 50

    def test_build_features_intraday_feature_count_matches_daily(
        self, hourly_prices, sample_prices
    ):
        engineer = FeatureEngineer()
        fs_intraday = engineer.build_features(hourly_prices, interval="1h")
        fs_daily = engineer.build_features(sample_prices, interval="1d")
        assert len(fs_intraday.feature_names) == len(fs_daily.feature_names)

    def test_build_features_horizon_bars(self, hourly_prices):
        engineer = FeatureEngineer()
        fs_24 = engineer.build_features(
            hourly_prices, interval="1h", horizon_bars=24
        )
        assert isinstance(fs_24, FeatureSet)
        # With horizon_bars=24 the target uses shift(-24); the default
        # (horizon_days=5) also shifts by 5 bars.  They should differ.
        fs_default = engineer.build_features(hourly_prices, interval="1h")
        min_len = min(len(fs_24.target), len(fs_default.target))
        assert min_len > 0, "Both FeatureSets should have rows"
        assert not np.allclose(fs_24.target[:min_len], fs_default.target[:min_len])

    def test_build_features_default_backward_compat(self, sample_prices):
        engineer = FeatureEngineer()
        fs_no_param = engineer.build_features(sample_prices)
        fs_explicit = engineer.build_features(sample_prices, interval="1d")
        # Results should be identical
        assert fs_no_param.feature_names == fs_explicit.feature_names
        np.testing.assert_array_equal(fs_no_param.target, fs_explicit.target)
        pd.testing.assert_frame_equal(
            fs_no_param.features_df, fs_explicit.features_df
        )
