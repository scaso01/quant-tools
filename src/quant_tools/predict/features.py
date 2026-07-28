"""Feature engineering for ML prediction models."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_tools.models import AssetClass, FeatureSet

log = logging.getLogger("quant_tools")


def _bars_per_day(interval: str) -> int:
    """Return the number of bars that fit in one trading day for *interval*.

    Unmapped intervals default to 1 (daily-equivalent).
    """
    return {"1h": 24, "4h": 6, "15m": 96, "1d": 1}.get(interval, 1)


class FeatureEngineer:
    """Builds feature matrices from OHLCV price data."""

    def build_features(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str = "",
        asset_class: AssetClass | None = None,
        horizon_days: int = 5,
        interval: str = "1d",
        horizon_bars: int | None = None,
    ) -> FeatureSet:
        """Build 50+ features from OHLCV DataFrame.

        Args:
            ohlcv_df: DataFrame with Open, High, Low, Close, Volume columns
                      and DatetimeIndex.
            symbol: Ticker symbol.
            asset_class: Asset class for class-specific features.
            horizon_days: Forward return horizon for target variable.
            interval: Bar interval (e.g. ``"1d"``, ``"1h"``, ``"15m"``).
                      Used to scale indicator windows so they represent the
                      same calendar duration regardless of bar frequency.
            horizon_bars: If provided, overrides *horizon_days* for the
                          target computation (``close.shift(-horizon_bars)``).

        Returns:
            FeatureSet with features_df, target array, and feature names.
        """
        bpd = _bars_per_day(interval)

        df = ohlcv_df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # -- Technical indicators --

        # RSI
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = self._rsi(close, period * bpd)

        # MACD
        macd, signal, hist = self._macd(
            close, fast=12 * bpd, slow=26 * bpd, signal=9 * bpd
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # Bollinger Bands
        for period in [20]:
            upper, mid, lower = self._bollinger(close, period * bpd)
            df[f"bb_upper_{period}"] = upper
            df[f"bb_mid_{period}"] = mid
            df[f"bb_lower_{period}"] = lower
            df[f"bb_pct_{period}"] = (close - lower) / (upper - lower + 1e-10)

        # ADX
        df["adx_14"] = self._adx(high, low, close, 14 * bpd)

        # ATR
        for period in [7, 14]:
            df[f"atr_{period}"] = self._atr(high, low, close, period * bpd)

        # Stochastic
        k, d = self._stochastic(high, low, close, 14 * bpd, 3 * bpd)
        df["stoch_k"] = k
        df["stoch_d"] = d

        # CCI
        df["cci_20"] = self._cci(high, low, close, 20 * bpd)

        # Williams %R
        df["willr_14"] = self._williams_r(high, low, close, 14 * bpd)

        # -- Price-derived features --

        # Log returns at multiple periods
        for period in [1, 5, 10, 20, 60]:
            df[f"return_{period}d"] = np.log(close / close.shift(period * bpd))

        # Rolling volatility
        log_ret = np.log(close / close.shift(1))
        for window in [5, 20, 60]:
            df[f"volatility_{window}d"] = (
                log_ret.rolling(window * bpd).std() * np.sqrt(252 * bpd)
            )

        # MA ratios
        for period in [10, 20, 50, 200]:
            sma = close.rolling(period * bpd).mean()
            df[f"close_sma{period}_ratio"] = close / sma

        # EMA ratios
        for period in [12, 26]:
            ema = close.ewm(span=period * bpd, adjust=False).mean()
            df[f"close_ema{period}_ratio"] = close / ema

        # High-low range
        df["hl_range"] = (high - low) / close
        df["hl_range_20d_avg"] = df["hl_range"].rolling(20 * bpd).mean()

        # -- Volume features --
        df["volume_sma20_ratio"] = volume / volume.rolling(20 * bpd).mean()
        df["obv"] = self._obv(close, volume)
        df["obv_slope_5"] = (
            df["obv"].diff(5 * bpd) / (df["obv"].shift(5 * bpd).abs() + 1e-10)
        )

        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        df["ad_line"] = (clv * volume).cumsum()
        df["ad_slope_5"] = df["ad_line"].diff(5 * bpd)

        # -- Calendar features --
        if hasattr(df.index, "dayofweek"):
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter

        # -- Momentum features --
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (close - close.shift(period * bpd))
                / close.shift(period * bpd)
                * 100
            )

        # -- Additional price features --
        df["close_to_high"] = close / high
        df["close_to_low"] = close / low
        df["gap"] = df["Open"] / close.shift(1) - 1

        # Volume rate of change
        df["volume_roc_5"] = volume.pct_change(5 * bpd)

        # Return distribution features (rolling)
        df["return_skew_20"] = log_ret.rolling(20 * bpd).skew()
        df["return_kurt_20"] = log_ret.rolling(20 * bpd).kurt()

        # MFI approximation (money flow index)
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        tp_diff = tp.diff()
        pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(14 * bpd).sum()
        neg_mf = raw_mf.where(tp_diff <= 0, 0.0).rolling(14 * bpd).sum()
        df["mfi_14"] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

        # -- Target: forward returns --
        _shift = horizon_bars if horizon_bars is not None else horizon_days
        target = (close.shift(-_shift) / close - 1).values

        # Drop NaN rows
        feature_cols = [
            c
            for c in df.columns
            if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        ]
        features = df[feature_cols].copy()
        mask = features.notna().all(axis=1) & pd.notna(target)
        features = features[mask]
        target = target[mask]

        return FeatureSet(
            symbol=symbol,
            features_df=features,
            target=target,
            feature_names=list(features.columns),
            close_prices=close[mask],
        )

    # -- Indicator implementations (pure numpy/pandas) --

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: upper, mid, lower."""
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        return dx.rolling(period).mean()

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range."""
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator: %K, %D."""
        lowest = low.rolling(k_period).min()
        highest = high.rolling(k_period).max()
        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def _cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (tp - sma) / (0.015 * mad + 1e-10)

    @staticmethod
    def _williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Williams %R."""
        highest = high.rolling(period).max()
        lowest = low.rolling(period).min()
        return -100 * (highest - close) / (highest - lowest + 1e-10)

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()
