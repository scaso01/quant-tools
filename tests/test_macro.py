"""Tests for factors/macro.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quant_tools.factors.macro import _classify_regime, detect_regime
from quant_tools.models import MacroRegime, MacroSnapshot


# =============================================================================
# _classify_regime (pure logic, no mocking needed)
# =============================================================================


class TestClassifyRegime:
    """Test regime classification logic with known inputs."""

    def test_expansion_regime(self):
        """Strong GDP + low unemployment + positive spread -> EXPANSION."""
        regime, signals = _classify_regime(gdp=3.5, unemployment=3.8, yield_spread=1.5)
        assert regime == MacroRegime.EXPANSION
        assert "gdp" in signals
        assert "unemployment" in signals

    def test_expansion_moderate_unemployment(self):
        """Strong GDP + moderate unemployment + positive spread -> EXPANSION."""
        regime, _ = _classify_regime(gdp=2.5, unemployment=4.5, yield_spread=0.5)
        assert regime == MacroRegime.EXPANSION

    def test_contraction_negative_gdp(self):
        """Negative GDP -> CONTRACTION regardless of other indicators."""
        regime, signals = _classify_regime(gdp=-1.5, unemployment=4.0, yield_spread=1.0)
        assert regime == MacroRegime.CONTRACTION
        assert "contraction" in signals.get("regime_reason", "").lower()

    def test_contraction_high_unemp_inverted(self):
        """High unemployment + inverted curve -> CONTRACTION."""
        regime, _ = _classify_regime(gdp=1.0, unemployment=7.0, yield_spread=-0.8)
        assert regime == MacroRegime.CONTRACTION

    def test_slowdown_moderate_gdp_flat_curve(self):
        """Moderate GDP + flat/narrowing curve -> SLOWDOWN."""
        regime, _ = _classify_regime(gdp=1.5, unemployment=4.0, yield_spread=-0.2)
        assert regime == MacroRegime.SLOWDOWN

    def test_slowdown_strong_gdp_inverted(self):
        """Strong GDP + inverted curve -> SLOWDOWN (warning signal)."""
        regime, _ = _classify_regime(gdp=3.0, unemployment=3.5, yield_spread=-0.8)
        assert regime == MacroRegime.SLOWDOWN

    def test_slowdown_elevated_unemp_flat_curve(self):
        """Elevated unemployment + flat curve -> SLOWDOWN."""
        regime, _ = _classify_regime(gdp=None, unemployment=5.5, yield_spread=-0.3)
        assert regime == MacroRegime.SLOWDOWN

    def test_recovery_moderate_gdp_elevated_unemp(self):
        """Moderate GDP + elevated unemployment -> RECOVERY."""
        regime, _ = _classify_regime(gdp=1.5, unemployment=5.5, yield_spread=0.5)
        assert regime == MacroRegime.RECOVERY

    def test_recovery_moderate_gdp_steep_curve(self):
        """Moderate GDP + steep curve -> RECOVERY."""
        regime, _ = _classify_regime(gdp=1.5, unemployment=4.0, yield_spread=1.5)
        assert regime == MacroRegime.RECOVERY

    def test_recovery_moderate_gdp_high_unemp(self):
        """Moderate GDP + high unemployment -> RECOVERY."""
        regime, _ = _classify_regime(gdp=1.0, unemployment=7.5, yield_spread=1.0)
        assert regime == MacroRegime.RECOVERY

    def test_all_none_defaults_slowdown(self):
        """All None data -> defaults to SLOWDOWN (mixed signals)."""
        regime, signals = _classify_regime(gdp=None, unemployment=None, yield_spread=None)
        assert regime == MacroRegime.SLOWDOWN
        assert "mixed" in signals.get("regime_reason", "").lower()

    def test_strong_gdp_defaults_expansion(self):
        """Strong GDP with unknown other signals -> EXPANSION default."""
        regime, _ = _classify_regime(gdp=3.0, unemployment=None, yield_spread=None)
        assert regime == MacroRegime.EXPANSION

    def test_signals_contain_data_descriptions(self):
        """All signals should have descriptive text."""
        _, signals = _classify_regime(gdp=2.5, unemployment=4.5, yield_spread=0.5)
        assert "gdp" in signals
        assert "unemployment" in signals
        assert "yield_spread" in signals
        assert "regime_reason" in signals

    def test_zero_gdp_is_contraction(self):
        """GDP exactly 0 -> CONTRACTION (not positive)."""
        regime, _ = _classify_regime(gdp=0.0, unemployment=5.0, yield_spread=0.5)
        assert regime == MacroRegime.CONTRACTION


# =============================================================================
# detect_regime (mocked FRED calls)
# =============================================================================


class TestDetectRegime:
    """Test full detect_regime with mocked FRED data."""

    @patch("quant_tools.factors.macro.fetch_yield_spread_10y2y")
    @patch("quant_tools.factors.macro.fetch_fed_funds_rate")
    @patch("quant_tools.factors.macro.fetch_cpi_yoy")
    @patch("quant_tools.factors.macro.fetch_unemployment")
    @patch("quant_tools.factors.macro.fetch_gdp_growth")
    def test_expansion_snapshot(
        self, mock_gdp, mock_unemp, mock_cpi, mock_ffr, mock_spread,
    ):
        """Full expansion scenario produces correct MacroSnapshot."""
        mock_gdp.return_value = 3.0
        mock_unemp.return_value = 3.7
        mock_cpi.return_value = 2.5
        mock_ffr.return_value = 5.25
        mock_spread.return_value = 0.5

        result = detect_regime()

        assert isinstance(result, MacroSnapshot)
        assert result.regime == MacroRegime.EXPANSION
        assert result.gdp_growth == 3.0
        assert result.unemployment == 3.7
        assert result.cpi_yoy == 2.5
        assert result.fed_funds_rate == 5.25
        assert result.yield_spread_10y2y == 0.5
        assert result.timestamp is not None

    @patch("quant_tools.factors.macro.fetch_yield_spread_10y2y")
    @patch("quant_tools.factors.macro.fetch_fed_funds_rate")
    @patch("quant_tools.factors.macro.fetch_cpi_yoy")
    @patch("quant_tools.factors.macro.fetch_unemployment")
    @patch("quant_tools.factors.macro.fetch_gdp_growth")
    def test_contraction_snapshot(
        self, mock_gdp, mock_unemp, mock_cpi, mock_ffr, mock_spread,
    ):
        """Contraction scenario with negative GDP."""
        mock_gdp.return_value = -2.0
        mock_unemp.return_value = 6.5
        mock_cpi.return_value = 1.0
        mock_ffr.return_value = 2.0
        mock_spread.return_value = -0.5

        result = detect_regime()

        assert result.regime == MacroRegime.CONTRACTION
        assert result.gdp_growth == -2.0

    @patch("quant_tools.factors.macro.fetch_yield_spread_10y2y")
    @patch("quant_tools.factors.macro.fetch_fed_funds_rate")
    @patch("quant_tools.factors.macro.fetch_cpi_yoy")
    @patch("quant_tools.factors.macro.fetch_unemployment")
    @patch("quant_tools.factors.macro.fetch_gdp_growth")
    def test_all_none_snapshot(
        self, mock_gdp, mock_unemp, mock_cpi, mock_ffr, mock_spread,
    ):
        """All None data produces valid snapshot with SLOWDOWN default."""
        mock_gdp.return_value = None
        mock_unemp.return_value = None
        mock_cpi.return_value = None
        mock_ffr.return_value = None
        mock_spread.return_value = None

        result = detect_regime()

        assert isinstance(result, MacroSnapshot)
        assert result.regime == MacroRegime.SLOWDOWN
        assert result.gdp_growth is None
        assert result.unemployment is None

    @patch("quant_tools.factors.macro.fetch_yield_spread_10y2y")
    @patch("quant_tools.factors.macro.fetch_fed_funds_rate")
    @patch("quant_tools.factors.macro.fetch_cpi_yoy")
    @patch("quant_tools.factors.macro.fetch_unemployment")
    @patch("quant_tools.factors.macro.fetch_gdp_growth")
    def test_recovery_snapshot(
        self, mock_gdp, mock_unemp, mock_cpi, mock_ffr, mock_spread,
    ):
        """Recovery scenario with moderate GDP and elevated unemployment."""
        mock_gdp.return_value = 1.5
        mock_unemp.return_value = 6.0
        mock_cpi.return_value = 3.0
        mock_ffr.return_value = 3.0
        mock_spread.return_value = 1.5

        result = detect_regime()

        assert result.regime == MacroRegime.RECOVERY

    @patch("quant_tools.factors.macro.fetch_yield_spread_10y2y")
    @patch("quant_tools.factors.macro.fetch_fed_funds_rate")
    @patch("quant_tools.factors.macro.fetch_cpi_yoy")
    @patch("quant_tools.factors.macro.fetch_unemployment")
    @patch("quant_tools.factors.macro.fetch_gdp_growth")
    def test_snapshot_includes_cpi_signal(
        self, mock_gdp, mock_unemp, mock_cpi, mock_ffr, mock_spread,
    ):
        """CPI data should appear in signals when available."""
        mock_gdp.return_value = 2.5
        mock_unemp.return_value = 4.0
        mock_cpi.return_value = 3.5
        mock_ffr.return_value = 5.0
        mock_spread.return_value = 0.8

        result = detect_regime()

        assert "cpi" in result.signals
        assert "3.5" in result.signals["cpi"]
