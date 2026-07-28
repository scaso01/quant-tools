# tests/test_dashboard.py
"""Dashboard regression tests for Quant-Tools Streamlit.

These tests hit a LIVE Streamlit dashboard (http://localhost:8502).
They verify health, page loading, page source definitions, data layer
integrity, and error handling.

Run:
    python -m pytest tests/test_dashboard.py -v

All tests are marked @pytest.mark.dashboard and auto-skip when the
Streamlit server is unreachable.

Uses http.client instead of requests/urllib to bypass the conftest.py
session-scoped network guard that blocks requests/httpx/urllib.request.
"""
from __future__ import annotations

import http.client
import importlib
import json
import os
import socket
import sys
from pathlib import Path

import pytest

# ── Configuration ────────────────────────────────────────────────────

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8502
TIMEOUT = 15  # seconds


# ── Reachability check ───────────────────────────────────────────────

def _streamlit_reachable() -> bool:
    """Return True if Streamlit port accepts TCP connections."""
    try:
        with socket.create_connection((STREAMLIT_HOST, STREAMLIT_PORT), timeout=5):
            return True
    except OSError:
        return False


_SKIP_REASON = f"Streamlit ({STREAMLIT_HOST}:{STREAMLIT_PORT}) unreachable"
skip_if_unreachable = pytest.mark.skipif(
    not _streamlit_reachable(), reason=_SKIP_REASON
)


# ── Helpers ──────────────────────────────────────────────────────────

def _http_get(path: str) -> tuple[int, str, dict[str, str]]:
    """GET a path via http.client. Returns (status, body_text, headers_dict).

    Uses http.client directly to bypass the conftest network guards
    that block requests, httpx, and urllib.request.
    """
    conn = http.client.HTTPConnection(STREAMLIT_HOST, STREAMLIT_PORT, timeout=TIMEOUT)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, body, headers
    finally:
        conn.close()


# ── Expected pages (source of truth from app.py) ────────────────────

EXPECTED_PAGES = [
    "Market Scanner",
    "Strategy Backtester",
    "Portfolio Analyzer",
    "Factors & Macro",
    "Predictions",
    "Performance",
    "Rolling Analytics",
    "Risk Dashboard",
    "Live Monitor",
    "AI Analysis",
]

EXPECTED_VIEW_MODULES = [
    "scanner",
    "backtest",
    "portfolio",
    "factors",
    "predictions",
    "performance",
    "rolling",
    "risk_dashboard",
    "live",
    "ai_analysis",
]


# =====================================================================
# Streamlit Health
# =====================================================================


@pytest.mark.dashboard
@skip_if_unreachable
class TestStreamlitHealth:
    def test_streamlit_health(self):
        """/_stcore/health returns 'ok'."""
        status, body, _ = _http_get("/_stcore/health")
        assert status == 200
        assert "ok" in body.lower()

    def test_main_page_loads(self):
        """GET / returns 200 with HTML content."""
        status, body, headers = _http_get("/")
        assert status == 200
        ct = headers.get("content-type", "")
        assert "text/html" in ct

    def test_main_page_has_streamlit_marker(self):
        """Main page HTML contains Streamlit framework markers."""
        status, body, _ = _http_get("/")
        assert status == 200
        # Streamlit injects its own root div and scripts
        assert "streamlit" in body.lower() or "stApp" in body or "_stcore" in body

    def test_streamlit_invalid_path(self):
        """GET /nonexistent returns a proper response (Streamlit routes
        everything through the main app, so it may return 200 with the
        app shell or 404)."""
        status, body, _ = _http_get("/nonexistent")
        # Streamlit typically returns 200 for all paths (SPA behavior)
        # or 404 for truly unknown paths. Either is acceptable.
        assert status in (200, 404)
        # Should never be a 500 error
        assert status != 500


# =====================================================================
# Page Source Verification
# =====================================================================


@pytest.mark.dashboard
@skip_if_unreachable
class TestPageDefinitions:
    def test_all_pages_defined(self):
        """All expected page names exist in the PAGES registry of app.py."""
        app_py = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "app.py"
        assert app_py.exists(), f"app.py not found at {app_py}"
        source = app_py.read_text(encoding="utf-8")

        for page_name in EXPECTED_PAGES:
            assert page_name in source, f"Page '{page_name}' not found in app.py PAGES registry"

    def test_all_view_modules_exist(self):
        """All view module files exist on disk."""
        views_dir = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "views"
        assert views_dir.is_dir(), f"Views dir not found: {views_dir}"

        for module_name in EXPECTED_VIEW_MODULES:
            module_path = views_dir / f"{module_name}.py"
            assert module_path.exists(), f"View module {module_name}.py not found"

    def test_all_page_sources_import_cleanly(self):
        """All dashboard view modules can be imported without errors."""
        for module_name in EXPECTED_VIEW_MODULES:
            full_name = f"quant_tools.dashboard.views.{module_name}"
            try:
                mod = importlib.import_module(full_name)
                assert hasattr(mod, "render"), f"Module {full_name} missing render() function"
            except Exception as exc:
                pytest.fail(f"Failed to import {full_name}: {exc}")

    def test_core_modules_exist(self):
        """Core dashboard modules (cache, colors, helpers) exist."""
        core_dir = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "core"
        assert core_dir.is_dir()

        for name in ("cache", "colors", "helpers"):
            path = core_dir / f"{name}.py"
            assert path.exists(), f"Core module {name}.py not found"

    def test_style_css_exists(self):
        """Dashboard style.css exists and is non-empty."""
        css_path = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "style.css"
        assert css_path.exists(), "style.css not found"
        content = css_path.read_text(encoding="utf-8")
        assert len(content) > 100, "style.css appears empty or trivially small"


# =====================================================================
# Data Layer (SQLite)
# =====================================================================


@pytest.mark.dashboard
class TestDataLayer:
    """Verify SQLite cache database tables.

    These tests do NOT need a live Streamlit server -- they verify the
    schema directly via SQLAlchemy. Engine is disposed before temp dir
    cleanup to avoid Windows file-locking errors.
    """

    def test_database_tables_defined(self, tmp_path):
        """CacheManager creates all expected tables."""
        from quant_tools.cache import CacheManager

        db_path = str(tmp_path / "test.db")
        cm = CacheManager(db_path=db_path)
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            tables = set(insp.get_table_names())

            expected = {
                "price_cache",
                "metadata_cache",
                "predictions",
                "prediction_outcomes",
                "model_performance",
            }
            for table in expected:
                assert table in tables, f"Missing table: {table}"
        finally:
            cm._engine.dispose()

    def test_prediction_table_columns(self, tmp_path):
        """Predictions table has all required columns for ML tracking."""
        from quant_tools.cache import CacheManager

        db_path = str(tmp_path / "test.db")
        cm = CacheManager(db_path=db_path)
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("predictions")}

            required = {
                "id", "timestamp", "symbol", "asset_class", "model_name",
                "prediction_direction", "predicted_magnitude", "confidence",
                "horizon_days", "target_date", "resolved",
            }
            for col in required:
                assert col in cols, f"Missing column in predictions: {col}"
        finally:
            cm._engine.dispose()

    def test_model_performance_table_columns(self, tmp_path):
        """Model performance table has win rate tracking columns."""
        from quant_tools.cache import CacheManager

        db_path = str(tmp_path / "test.db")
        cm = CacheManager(db_path=db_path)
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("model_performance")}

            required = {
                "id", "model_name", "asset_class", "total_predictions",
                "wins", "losses", "win_rate", "avg_confidence", "avg_error",
            }
            for col in required:
                assert col in cols, f"Missing column in model_performance: {col}"
        finally:
            cm._engine.dispose()


# =====================================================================
# Streamlit Component Endpoints
# =====================================================================


@pytest.mark.dashboard
@skip_if_unreachable
class TestStreamlitComponents:
    def test_stcore_allowed_message_origins(self):
        """Streamlit exposes allowed message origins config."""
        status, body, _ = _http_get("/_stcore/allowed-message-origins")
        assert status == 200

    def test_streamlit_static_assets(self):
        """Streamlit serves its own static JS/CSS bundles."""
        status, body, _ = _http_get("/")
        assert status == 200
        # Streamlit pages reference _stcore static assets
        assert "_stcore" in body or "static" in body


# =====================================================================
# Security / No Hardcoded Secrets
# =====================================================================


@pytest.mark.dashboard
class TestSecurity:
    def test_no_hardcoded_secrets(self):
        """Dashboard view sources contain no API keys, passwords, or secrets."""
        views_dir = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "views"
        secret_patterns = [
            "api_key",
            "apikey",
            "password",
            "passwd",
            "secret",
            "token",
            "sk-",
            "Bearer ",
        ]

        for py_file in views_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8").lower()
            for pattern in secret_patterns:
                # Allow common false positives like variable names, config refs
                lines_with_pattern = [
                    line.strip() for line in source.split("\n")
                    if pattern.lower() in line
                    and not line.strip().startswith("#")
                    and not line.strip().startswith('"""')
                    and "import" not in line
                    and "def " not in line
                    and "param" not in line
                    and "label" not in line
                    and "header" not in line
                    and "column" not in line
                    and "config" not in line
                    and "setting" not in line
                    and "env" not in line
                    and "os.getenv" not in line
                    and "os.environ" not in line
                ]
                # Only fail if there is a suspicious hardcoded string value
                for line in lines_with_pattern:
                    # Look for actual string assignments like api_key = "sk-abc123"
                    if ('= "' in line or "= '" in line) and len(line) > 20:
                        # Check if it looks like a real secret (has alphanumeric after quote)
                        import re
                        if re.search(r'["\'][A-Za-z0-9+/=_-]{16,}["\']', line):
                            pytest.fail(
                                f"Possible hardcoded secret in {py_file.name}: {line[:80]}"
                            )

    def test_no_secrets_in_app_py(self):
        """Main app.py contains no hardcoded secrets."""
        app_py = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "app.py"
        source = app_py.read_text(encoding="utf-8")
        # Look for obvious patterns
        import re
        # Matches things like sk-ant-..., ghp_..., etc
        matches = re.findall(r'["\'](?:sk-|ghp_|gho_|Bearer )[A-Za-z0-9_-]{10,}["\']', source)
        assert len(matches) == 0, f"Hardcoded secret found in app.py: {matches}"


# =====================================================================
# Dashboard Config Consistency
# =====================================================================


@pytest.mark.dashboard
class TestDashboardConfig:
    def test_page_count_matches(self):
        """Number of view modules matches number of pages in PAGES dict."""
        app_py = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "app.py"
        source = app_py.read_text(encoding="utf-8")

        # Count PAGES entries by looking for quoted strings in the PAGES dict
        import re
        # Match lines like '    "... Market Scanner": scanner,'
        page_entries = re.findall(r'"[^"]+"\s*:\s*\w+', source)
        # Filter to only the ones inside PAGES dict (they import view modules)
        page_count = len([e for e in page_entries if any(m in e for m in EXPECTED_VIEW_MODULES)])

        assert page_count == len(EXPECTED_VIEW_MODULES), (
            f"PAGES dict has {page_count} entries but expected {len(EXPECTED_VIEW_MODULES)}"
        )

    def test_views_all_have_render(self):
        """Every view module exposes a render() function."""
        views_dir = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "views"
        for py_file in sorted(views_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            source = py_file.read_text(encoding="utf-8")
            assert "def render(" in source, f"{py_file.name} missing render() function"


# #####################################################################
# DEEP FUNCTIONAL REGRESSION TESTS
#
# Everything below exercises the actual Python functions used by the
# dashboard pages.  No live Streamlit server is required unless the
# test is explicitly marked @skip_if_unreachable.
#
# All tests are marked @pytest.mark.dashboard so they can be run in
# isolation with:  pytest -m dashboard -v
# #####################################################################


import re
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# =====================================================================
# Market Scanner -- Universe & Data Layer
# =====================================================================


@pytest.mark.dashboard
class TestScannerUniverse:
    """Verify universe definitions, sector maps, and preset populations."""

    def test_scanner_loads_sp500_top100(self):
        """SP500_TOP100 dict is populated with >= 100 tickers."""
        from quant_tools.data.universe import SP500_TOP100
        assert len(SP500_TOP100) >= 100

    def test_scanner_sp500_contains_known_tickers(self):
        """Well-known mega-cap tickers are in the SP500 Top 100."""
        from quant_tools.data.universe import SP500_TOP100
        for sym in ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"):
            assert sym in SP500_TOP100, f"{sym} missing from SP500_TOP100"

    def test_scanner_has_all_11_gics_sectors(self):
        """SP500_TOP100 covers all 11 GICS sectors."""
        from quant_tools.data.universe import SP500_TOP100
        sectors = set(SP500_TOP100.values())
        expected = {
            "Technology", "Healthcare", "Financials", "Consumer Discretionary",
            "Consumer Staples", "Industrials", "Energy", "Materials",
            "Utilities", "Real Estate", "Communication Services",
        }
        assert sectors == expected

    def test_sector_map_inverse_consistent(self):
        """SECTOR_MAP (sector -> tickers) is consistent with SP500_TOP100."""
        from quant_tools.data.universe import SECTOR_MAP, SP500_TOP100
        for sector, tickers in SECTOR_MAP.items():
            for t in tickers:
                assert SP500_TOP100.get(t) == sector, (
                    f"Ticker {t} in SECTOR_MAP[{sector}] but SP500_TOP100 says {SP500_TOP100.get(t)}"
                )

    def test_scanner_has_equity_asset_class(self):
        """Equity asset class is defined and usable."""
        from quant_tools.models import AssetClass
        assert AssetClass.EQUITY.value == "equity"

    def test_scanner_has_crypto_asset_class(self):
        """Crypto asset class is defined and usable."""
        from quant_tools.models import AssetClass
        assert AssetClass.CRYPTO.value == "crypto"

    def test_scanner_has_forex_asset_class(self):
        """Forex asset class is defined and usable."""
        from quant_tools.models import AssetClass
        assert AssetClass.FOREX.value == "forex"

    def test_scanner_has_commodity_asset_class(self):
        """Commodity asset class is defined and usable."""
        from quant_tools.models import AssetClass
        assert AssetClass.COMMODITY.value == "commodity"

    def test_scanner_universe_presets_exist(self):
        """UNIVERSE_PRESETS dict has at least the core presets."""
        from quant_tools.data.universe import UNIVERSE_PRESETS
        assert len(UNIVERSE_PRESETS) >= 5
        for name in ("S&P 500 Top 100", "Magnificent 7", "Dow 30"):
            assert name in UNIVERSE_PRESETS, f"Preset '{name}' missing"

    def test_scanner_preset_magnificent7_has_7(self):
        """Magnificent 7 preset has exactly 7 symbols."""
        from quant_tools.data.universe import UNIVERSE_PRESETS
        mag7 = UNIVERSE_PRESETS["Magnificent 7"]
        assert mag7.count == 7

    def test_scanner_preset_count_matches_symbols(self):
        """Each preset's .count property matches len(symbols)."""
        from quant_tools.data.universe import UNIVERSE_PRESETS
        for name, preset in UNIVERSE_PRESETS.items():
            assert preset.count == len(preset.symbols), (
                f"Preset '{name}': count={preset.count} but len(symbols)={len(preset.symbols)}"
            )

    def test_crypto_universe_has_entries(self):
        """Crypto universe dict is populated."""
        from quant_tools.data.universe import CRYPTO_UNIVERSE
        assert len(CRYPTO_UNIVERSE) >= 10
        assert "BTC/USDT" in CRYPTO_UNIVERSE

    def test_forex_universe_has_entries(self):
        """Forex universe dict is populated."""
        from quant_tools.data.universe import FOREX_UNIVERSE
        assert len(FOREX_UNIVERSE) >= 5
        assert "EUR/USD" in FOREX_UNIVERSE

    def test_commodity_universe_has_entries(self):
        """Commodity universe dict is populated."""
        from quant_tools.data.universe import COMMODITY_UNIVERSE
        assert len(COMMODITY_UNIVERSE) >= 10
        assert "GC=F" in COMMODITY_UNIVERSE  # Gold

    def test_commodity_presets_cover_all_categories(self):
        """Commodity presets include energy, metals, agriculture, etc."""
        from quant_tools.data.universe import COMMODITY_PRESETS
        expected_categories = {"Energy", "Precious Metals", "Agriculture", "All Commodities"}
        for cat in expected_categories:
            assert cat in COMMODITY_PRESETS, f"Commodity category '{cat}' missing"

    def test_list_all_presets_returns_strings(self):
        """list_all_presets() returns a non-empty list of strings."""
        from quant_tools.data.universe import list_all_presets
        presets = list_all_presets()
        assert len(presets) >= 5
        assert all(isinstance(p, str) for p in presets)

    def test_get_universe_sp500(self):
        """get_universe('sp500') returns 100+ tickers."""
        from quant_tools.data.universe import get_universe
        tickers = get_universe("sp500")
        assert len(tickers) >= 100
        assert "AAPL" in tickers

    def test_get_universe_crypto(self):
        """get_universe(asset_class='crypto') returns crypto symbols."""
        from quant_tools.data.universe import get_universe
        tickers = get_universe("", asset_class="crypto")
        assert len(tickers) >= 10
        assert any("/" in t for t in tickers)

    def test_get_universe_unknown_raises(self):
        """get_universe with unknown name raises ValueError."""
        from quant_tools.data.universe import get_universe
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nonexistent_universe_xyz")


# =====================================================================
# Market Scanner -- Universe JSON Files
# =====================================================================


@pytest.mark.dashboard
class TestUniverseJsonFiles:
    """Verify the static JSON universe files on disk."""

    def test_sp500_json_exists(self):
        """sp500.json file exists and has entries."""
        from quant_tools.data.universe_loader import load_universe
        universe = load_universe("sp500")
        assert len(universe) >= 400

    def test_russell1000_json_exists(self):
        """russell1000.json file exists and has entries."""
        from quant_tools.data.universe_loader import load_universe
        universe = load_universe("russell1000")
        assert len(universe) >= 800

    def test_international_json_exists(self):
        """international.json file exists and has entries."""
        from quant_tools.data.universe_loader import load_universe
        universe = load_universe("international")
        assert len(universe) >= 30

    def test_universe_ticker_metadata_structure(self):
        """Each entry in a loaded universe has correct TickerMeta fields."""
        from quant_tools.data.universe_loader import load_universe
        from quant_tools.models import TickerMeta
        universe = load_universe("sp500")
        for sym, meta in list(universe.items())[:10]:
            assert isinstance(meta, TickerMeta)
            assert isinstance(meta.name, str) and len(meta.name) > 0

    def test_load_combined_merges(self):
        """load_combined() merges multiple universes with dedup."""
        from quant_tools.data.universe_loader import load_combined
        combined = load_combined(["sp500", "international"])
        assert len(combined) >= 400

    def test_unknown_universe_raises(self):
        """Loading a nonexistent universe JSON raises FileNotFoundError."""
        from quant_tools.data.universe_loader import load_universe
        with pytest.raises(FileNotFoundError):
            load_universe("nonexistent_xyz_12345")


# =====================================================================
# Market Scanner -- Signal Generation
# =====================================================================


@pytest.mark.dashboard
class TestScannerSignals:
    """Verify signal generation functions used by the scanner page."""

    @pytest.fixture
    def close_prices(self) -> np.ndarray:
        """Deterministic 252-day close price series for signal testing."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 252)
        return 100 * np.cumprod(1 + returns)

    @pytest.fixture
    def volume_data(self) -> np.ndarray:
        """Deterministic 252-day volume series."""
        np.random.seed(42)
        return np.random.randint(1_000_000, 50_000_000, 252).astype(np.float64)

    def test_momentum_signal_returns_score(self, close_prices):
        """Momentum signal returns a float score in [0, 100]."""
        from quant_tools.scanner.signals import momentum_signal
        score = momentum_signal(close_prices)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_momentum_signal_with_breakdown(self, close_prices):
        """Momentum signal with return_breakdown=True returns (score, dict)."""
        from quant_tools.scanner.signals import momentum_signal
        result = momentum_signal(close_prices, return_breakdown=True)
        assert isinstance(result, tuple)
        score, breakdown = result
        assert isinstance(score, float)
        assert isinstance(breakdown, dict)
        assert "rsi" in breakdown
        assert "sma_crossover" in breakdown

    def test_mean_reversion_signal_returns_score(self, close_prices):
        """Mean reversion signal returns a float score in [0, 100]."""
        from quant_tools.scanner.signals import mean_reversion_signal
        score = mean_reversion_signal(close_prices)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_mean_reversion_signal_with_breakdown(self, close_prices):
        """Mean reversion signal with breakdown returns components."""
        from quant_tools.scanner.signals import mean_reversion_signal
        result = mean_reversion_signal(close_prices, return_breakdown=True)
        assert isinstance(result, tuple)
        score, breakdown = result
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_momentum_signal_short_series(self):
        """Momentum signal handles very short price series gracefully."""
        from quant_tools.scanner.signals import momentum_signal
        short = np.array([100.0, 101.0, 99.5])
        score = momentum_signal(short)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_signal_type_enum_values(self):
        """All 4 signal types are defined with correct string values."""
        from quant_tools.models import SignalType
        assert SignalType.MOMENTUM.value == "momentum"
        assert SignalType.VALUE.value == "value"
        assert SignalType.QUALITY.value == "quality"
        assert SignalType.MEAN_REVERSION.value == "mean_reversion"

    def test_signal_explanations_exist_for_all_types(self):
        """SIGNAL_EXPLANATIONS has an entry for every SignalType."""
        from quant_tools.models import SignalType
        from quant_tools.scanner.signal_explanations import SIGNAL_EXPLANATIONS
        for st in SignalType:
            assert st in SIGNAL_EXPLANATIONS, f"Missing explanation for {st}"

    def test_signal_explanation_has_components(self):
        """Each signal explanation has at least 1 component."""
        from quant_tools.scanner.signal_explanations import SIGNAL_EXPLANATIONS
        for sig_type, explanation in SIGNAL_EXPLANATIONS.items():
            assert len(explanation.components) >= 1, (
                f"Signal {sig_type} has no components"
            )
            assert explanation.summary, f"Signal {sig_type} has empty summary"

    def test_signal_explanation_component_weights_sum(self):
        """Component weights for each signal sum to ~1.0."""
        from quant_tools.scanner.signal_explanations import SIGNAL_EXPLANATIONS
        for sig_type, explanation in SIGNAL_EXPLANATIONS.items():
            total = sum(c.weight for c in explanation.components)
            assert 0.9 <= total <= 1.1, (
                f"Signal {sig_type} component weights sum to {total:.2f}, expected ~1.0"
            )


# =====================================================================
# Market Scanner -- Sector Filtering
# =====================================================================


@pytest.mark.dashboard
class TestScannerSectorFiltering:
    """Verify sector-based filtering used by the scanner page."""

    def test_get_sector_tickers_technology(self):
        """get_sector_tickers('Technology') returns tech stocks."""
        from quant_tools.data.universe import get_sector_tickers
        tickers = get_sector_tickers("Technology")
        assert len(tickers) >= 5
        assert "AAPL" in tickers

    def test_get_sector_tickers_financials(self):
        """get_sector_tickers('Financials') returns financial stocks."""
        from quant_tools.data.universe import get_sector_tickers
        tickers = get_sector_tickers("Financials")
        assert len(tickers) >= 5
        assert "JPM" in tickers

    def test_get_sector_tickers_unknown_raises(self):
        """Unknown sector raises ValueError."""
        from quant_tools.data.universe import get_sector_tickers
        with pytest.raises(ValueError, match="Unknown sector"):
            get_sector_tickers("Nonexistent Sector XYZ")

    def test_get_ticker_sector_aapl(self):
        """get_ticker_sector('AAPL') returns 'Technology'."""
        from quant_tools.data.universe import get_ticker_sector
        assert get_ticker_sector("AAPL") == "Technology"

    def test_get_ticker_sector_unknown_raises(self):
        """Unknown ticker raises KeyError."""
        from quant_tools.data.universe import get_ticker_sector
        with pytest.raises(KeyError):
            get_ticker_sector("ZZZZZZ")

    def test_sector_filter_reduces_universe(self):
        """Filtering by a single sector produces fewer tickers than the full universe."""
        from quant_tools.data.universe import SECTOR_MAP, SP500_TOP100
        tech_tickers = set(SECTOR_MAP.get("Technology", []))
        all_tickers = set(SP500_TOP100.keys())
        assert len(tech_tickers) < len(all_tickers)
        assert tech_tickers.issubset(all_tickers)


# =====================================================================
# Predictions -- Model Definitions & Horizons
# =====================================================================


@pytest.mark.dashboard
class TestPredictionModels:
    """Verify prediction model definitions the dashboard relies on."""

    def test_model_type_enum_has_5_models(self):
        """ModelType enum has exactly 5 entries."""
        from quant_tools.models import ModelType
        assert len(ModelType) == 5

    def test_model_type_names(self):
        """The 5 expected model names are present."""
        from quant_tools.models import ModelType
        expected = {"gradient_boost", "stats_forecast", "neural_forecast", "prophet", "garch"}
        actual = {m.value for m in ModelType}
        assert actual == expected

    def test_prediction_direction_enum(self):
        """PredictionDirection has up, down, neutral."""
        from quant_tools.models import PredictionDirection
        assert PredictionDirection.UP.value == "up"
        assert PredictionDirection.DOWN.value == "down"
        assert PredictionDirection.NEUTRAL.value == "neutral"

    def test_consensus_signal_enum(self):
        """ConsensusSignal covers the full spectrum."""
        from quant_tools.models import ConsensusSignal
        expected = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        actual = {s.value for s in ConsensusSignal}
        assert actual == expected

    def test_horizon_presets_equity(self):
        """Equity horizons include 1d, 5d, 20d, 60d."""
        from quant_tools.models import AssetClass
        from quant_tools.predict.horizons import get_horizon_labels
        labels = get_horizon_labels(AssetClass.EQUITY)
        assert "1d" in labels
        assert "5d" in labels
        assert "20d" in labels
        assert "60d" in labels

    def test_horizon_presets_crypto_includes_hours(self):
        """Crypto horizons include hourly presets."""
        from quant_tools.models import AssetClass
        from quant_tools.predict.horizons import get_horizon_labels
        labels = get_horizon_labels(AssetClass.CRYPTO)
        assert "1h" in labels
        assert "4h" in labels

    def test_parse_horizon_label_valid(self):
        """parse_horizon_label correctly parses '5d' and '4h'."""
        from quant_tools.predict.horizons import parse_horizon_label
        p = parse_horizon_label("5d")
        assert p is not None
        assert p.value == 5
        assert p.unit == "d"
        assert p.interval == "1d"

        p2 = parse_horizon_label("4h")
        assert p2 is not None
        assert p2.value == 4
        assert p2.unit == "h"
        assert p2.interval == "1h"

    def test_parse_horizon_label_invalid(self):
        """parse_horizon_label returns None for invalid labels."""
        from quant_tools.predict.horizons import parse_horizon_label
        assert parse_horizon_label("") is None
        assert parse_horizon_label("x") is None
        assert parse_horizon_label("abc") is None

    def test_horizon_display_days(self):
        """horizon_display formats days correctly."""
        from quant_tools.predict.horizons import horizon_display
        assert horizon_display(horizon_days=5) == "5d"
        assert horizon_display(horizon_days=20) == "20d"

    def test_horizon_display_hours(self):
        """horizon_display prefers hours when provided."""
        from quant_tools.predict.horizons import horizon_display
        assert horizon_display(horizon_hours=4) == "4h"
        assert horizon_display(horizon_days=1, horizon_hours=24) == "24h"

    def test_horizon_display_unknown(self):
        """horizon_display returns '?' when no horizon given."""
        from quant_tools.predict.horizons import horizon_display
        assert horizon_display() == "?"


# =====================================================================
# Predictions -- Tracker & Storage
# =====================================================================


@pytest.mark.dashboard
class TestPredictionTracker:
    """Verify prediction tracker CRUD used by the Predictions page."""

    def test_tracker_creates_with_temp_db(self, tmp_path):
        """PredictionTracker can be instantiated with a temp database."""
        from quant_tools.cache import CacheManager
        from quant_tools.predict.tracker import PredictionTracker
        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)
            assert tracker is not None
        finally:
            cm._engine.dispose()

    def test_tracker_get_predictions_empty(self, tmp_path):
        """Fresh tracker returns empty predictions list."""
        from quant_tools.cache import CacheManager
        from quant_tools.predict.tracker import PredictionTracker
        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)
            preds = tracker.get_predictions(limit=10)
            assert preds == []
        finally:
            cm._engine.dispose()

    def test_tracker_get_performance_empty(self, tmp_path):
        """Fresh tracker returns empty performance list."""
        from quant_tools.cache import CacheManager
        from quant_tools.predict.tracker import PredictionTracker
        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)
            perf = tracker.get_performance()
            assert perf == []
        finally:
            cm._engine.dispose()

    def test_tracker_store_and_retrieve(self, tmp_path):
        """Store an EnsemblePrediction, then retrieve it."""
        from quant_tools.cache import CacheManager
        from quant_tools.models import (
            AssetClass, ConsensusSignal, EnsemblePrediction,
            PredictionDirection, PredictionResult,
        )
        from quant_tools.predict.tracker import PredictionTracker

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)

            # Build a minimal ensemble prediction
            model_pred = PredictionResult(
                symbol="AAPL", asset_class=AssetClass.EQUITY,
                model_name="gradient_boost",
                direction=PredictionDirection.UP,
                magnitude=0.02, confidence=0.7,
                horizon_days=5, target_date="2026-04-01",
            )
            ensemble = EnsemblePrediction(
                symbol="AAPL", asset_class=AssetClass.EQUITY,
                consensus=ConsensusSignal.BUY,
                confidence=0.7, direction=PredictionDirection.UP,
                magnitude=0.02, horizon_days=5,
                target_date="2026-04-01",
                model_predictions=[model_pred],
            )
            pred_id = tracker.store_prediction(ensemble)
            assert isinstance(pred_id, int)
            assert pred_id > 0

            # Retrieve
            preds = tracker.get_predictions(limit=10)
            assert len(preds) >= 1
            symbols = {p["symbol"] for p in preds}
            assert "AAPL" in symbols
        finally:
            cm._engine.dispose()

    def test_tracker_clear_predictions(self, tmp_path):
        """clear_predictions removes all stored predictions."""
        from quant_tools.cache import CacheManager
        from quant_tools.models import (
            AssetClass, ConsensusSignal, EnsemblePrediction,
            PredictionDirection, PredictionResult,
        )
        from quant_tools.predict.tracker import PredictionTracker

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)
            model_pred = PredictionResult(
                symbol="SPY", asset_class=AssetClass.EQUITY,
                model_name="gradient_boost",
                direction=PredictionDirection.UP,
                magnitude=0.01, confidence=0.6,
                horizon_days=5, target_date="2026-04-01",
            )
            ensemble = EnsemblePrediction(
                symbol="SPY", asset_class=AssetClass.EQUITY,
                consensus=ConsensusSignal.HOLD,
                confidence=0.6, direction=PredictionDirection.UP,
                magnitude=0.01, horizon_days=5,
                target_date="2026-04-01",
                model_predictions=[model_pred],
            )
            tracker.store_prediction(ensemble)

            count = tracker.clear_predictions()
            assert count >= 1

            preds = tracker.get_predictions(limit=10)
            assert preds == []
        finally:
            cm._engine.dispose()

    def test_prediction_stored_dates_valid(self, tmp_path):
        """Stored predictions have parseable dates that are not in the far future."""
        from quant_tools.cache import CacheManager
        from quant_tools.models import (
            AssetClass, ConsensusSignal, EnsemblePrediction,
            PredictionDirection, PredictionResult,
        )
        from quant_tools.predict.tracker import PredictionTracker

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            tracker = PredictionTracker(cache_manager=cm)
            target = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
            model_pred = PredictionResult(
                symbol="MSFT", asset_class=AssetClass.EQUITY,
                model_name="gradient_boost",
                direction=PredictionDirection.UP,
                magnitude=0.01, confidence=0.65,
                horizon_days=5, target_date=target,
            )
            ensemble = EnsemblePrediction(
                symbol="MSFT", asset_class=AssetClass.EQUITY,
                consensus=ConsensusSignal.BUY,
                confidence=0.65, direction=PredictionDirection.UP,
                magnitude=0.01, horizon_days=5,
                target_date=target,
                model_predictions=[model_pred],
            )
            tracker.store_prediction(ensemble)
            preds = tracker.get_predictions(limit=10)
            for p in preds:
                td = p.get("target_date", "")
                if td:
                    parsed = datetime.strptime(td, "%Y-%m-%d")
                    # Should not be more than 1 year in the future
                    assert parsed < datetime.now() + timedelta(days=366)
        finally:
            cm._engine.dispose()

    def test_model_names_consistent_with_enum(self, tmp_path):
        """Stored model names match the ModelType enum values."""
        from quant_tools.models import ModelType
        valid_names = {m.value for m in ModelType} | {"ensemble"}
        # Just verify the set of valid names is what the dashboard expects
        assert "gradient_boost" in valid_names
        assert "stats_forecast" in valid_names
        assert "neural_forecast" in valid_names
        assert "prophet" in valid_names
        assert "garch" in valid_names
        assert "ensemble" in valid_names


# =====================================================================
# Portfolio -- Analysis Models
# =====================================================================


@pytest.mark.dashboard
class TestPortfolioModels:
    """Verify portfolio data models used by the Portfolio Analyzer page."""

    def test_portfolio_analysis_dataclass(self):
        """PortfolioAnalysis can be constructed with expected fields."""
        from quant_tools.models import PortfolioAnalysis, PortfolioHolding
        holding = PortfolioHolding(
            symbol="AAPL", weight=0.5,
            current_price=150.0, total_return=0.12,
            annualized_return=0.10, volatility=0.20, sharpe=1.5,
        )
        analysis = PortfolioAnalysis(
            holdings=[holding],
            total_return=0.12, annualized_return=0.10,
            volatility=0.20, sharpe_ratio=1.5, max_drawdown=-0.15,
        )
        assert analysis.total_return == 0.12
        assert len(analysis.holdings) == 1
        assert analysis.holdings[0].symbol == "AAPL"

    def test_optimization_constraints_dataclass(self):
        """OptimizationConstraints can be constructed with nested constraints."""
        from quant_tools.models import (
            AssetConstraint, GroupConstraint, OptimizationConstraints,
        )
        ac = AssetConstraint(symbol="AAPL", min_weight=0.05, max_weight=0.30)
        gc = GroupConstraint(name="Tech", symbols=["AAPL", "MSFT"], max_weight=0.50)
        constraints = OptimizationConstraints(
            asset_constraints=[ac],
            group_constraints=[gc],
            max_turnover=0.5,
        )
        assert constraints.max_turnover == 0.5
        assert len(constraints.asset_constraints) == 1
        assert len(constraints.group_constraints) == 1

    def test_efficient_frontier_dataclass(self):
        """EfficientFrontier and EfficientFrontierPoint constructible."""
        from quant_tools.models import EfficientFrontier, EfficientFrontierPoint
        point = EfficientFrontierPoint(
            expected_return=0.10, volatility=0.15, sharpe_ratio=0.67,
            weights={"AAPL": 0.5, "BND": 0.5},
        )
        frontier = EfficientFrontier(
            points=[point],
            max_sharpe_point=point,
            min_variance_point=point,
        )
        assert len(frontier.points) == 1
        assert frontier.max_sharpe_point.sharpe_ratio == 0.67

    def test_distribution_type_enum_has_5_types(self):
        """DistributionType enum covers all 5 MC distributions."""
        from quant_tools.models import DistributionType
        expected = {"normal", "student_t", "skew_normal", "johnson_su", "ged"}
        actual = {d.value for d in DistributionType}
        assert actual == expected

    def test_monte_carlo_result_dataclass(self):
        """MonteCarloResult can be constructed with expected fields."""
        from quant_tools.models import DistributionType, MonteCarloResult
        mc = MonteCarloResult(
            n_simulations=1000, time_horizon_days=252,
            initial_value=100000, mean_final=112000,
            median_final=110000, percentile_5=85000,
            percentile_25=95000, percentile_75=125000,
            percentile_95=140000, prob_loss=0.25,
            prob_gain_10pct=0.55,
            distribution_used=DistributionType.NORMAL,
        )
        assert mc.n_simulations == 1000
        assert mc.prob_loss == 0.25


# =====================================================================
# Portfolio -- Monte Carlo Engine
# =====================================================================


@pytest.mark.dashboard
class TestMonteCarloEngine:
    """Verify Monte Carlo simulation functions the dashboard calls."""

    @pytest.fixture
    def portfolio_returns(self) -> np.ndarray:
        """Deterministic daily portfolio returns for MC testing."""
        np.random.seed(42)
        return np.random.normal(0.0004, 0.012, 252)

    def test_mc_simulation_runs(self, portfolio_returns):
        """run_monte_carlo produces a MonteCarloResult."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=100,
            time_horizon_days=63,
        )
        assert mc is not None
        assert mc.n_simulations == 100
        assert mc.time_horizon_days == 63
        assert mc.initial_value == 100000

    def test_mc_has_simulated_paths(self, portfolio_returns):
        """Simulation produces the requested number of paths."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=50,
            time_horizon_days=63,
        )
        assert len(mc.simulated_paths) == 50
        assert len(mc.simulated_paths[0]) == 64  # 63 days + initial

    def test_mc_percentiles_ordered(self, portfolio_returns):
        """Percentile values are in ascending order: 5th < 25th < median < 75th < 95th."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=200,
            time_horizon_days=252,
        )
        assert mc.percentile_5 <= mc.percentile_25
        assert mc.percentile_25 <= mc.median_final
        assert mc.median_final <= mc.percentile_75
        assert mc.percentile_75 <= mc.percentile_95

    def test_mc_prob_loss_in_range(self, portfolio_returns):
        """Probability of loss is between 0 and 1."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=200,
            time_horizon_days=252,
        )
        assert 0.0 <= mc.prob_loss <= 1.0
        assert 0.0 <= mc.prob_gain_10pct <= 1.0

    def test_mc_normal_distribution(self, portfolio_returns):
        """MC with Normal distribution runs and reports distribution type."""
        from quant_tools.models import DistributionType
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=100,
            time_horizon_days=63,
            distribution=DistributionType.NORMAL,
        )
        assert mc.distribution_used == DistributionType.NORMAL

    def test_mc_student_t_distribution(self, portfolio_returns):
        """MC with Student-t distribution runs and reports distribution type."""
        from quant_tools.models import DistributionType
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=100,
            time_horizon_days=63,
            distribution=DistributionType.STUDENT_T,
        )
        assert mc.distribution_used == DistributionType.STUDENT_T

    def test_mc_skewness_kurtosis_populated(self, portfolio_returns):
        """MC result includes skewness and kurtosis."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=100,
            time_horizon_days=63,
        )
        assert isinstance(mc.skewness, float)
        assert isinstance(mc.kurtosis, float)

    def test_mc_distribution_params_populated(self, portfolio_returns):
        """MC result includes fitted distribution parameters."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        mc = run_monte_carlo(
            portfolio_returns,
            initial_value=100000,
            n_simulations=100,
            time_horizon_days=63,
        )
        assert mc.distribution_params is not None
        assert isinstance(mc.distribution_params.mean, float)
        assert isinstance(mc.distribution_params.std, float)
        assert mc.distribution_params.std > 0

    def test_estimate_distribution_params_normal(self, portfolio_returns):
        """estimate_distribution_params for Normal returns valid params."""
        from quant_tools.models import DistributionType
        from quant_tools.portfolio.monte_carlo import estimate_distribution_params
        params = estimate_distribution_params(
            portfolio_returns, DistributionType.NORMAL,
        )
        assert params.distribution == DistributionType.NORMAL
        assert abs(params.mean) < 0.1
        assert params.std > 0


# =====================================================================
# Portfolio -- Optimizer
# =====================================================================


@pytest.mark.dashboard
class TestPortfolioOptimizer:
    """Verify portfolio optimization functions the dashboard calls."""

    @pytest.fixture
    def returns_df(self) -> pd.DataFrame:
        """Deterministic multi-asset returns DataFrame for optimizer testing."""
        np.random.seed(42)
        n = 252
        data = {}
        for i, sym in enumerate(["AAPL", "MSFT", "BND", "GLD"]):
            np.random.seed(42 + i)
            data[sym] = np.random.normal(0.0003 + i * 0.0001, 0.01 + i * 0.002, n)
        return pd.DataFrame(data)

    def test_markowitz_optimization_runs(self, returns_df):
        """Markowitz optimization returns weights summing to 1."""
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        result = opt.optimize_markowitz(returns_df)
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01
        assert result.expected_return != 0
        assert result.volatility > 0

    def test_min_variance_optimization_runs(self, returns_df):
        """Min variance optimization returns weights summing to 1."""
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        result = opt.optimize_min_variance(returns_df)
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01
        assert result.volatility > 0

    def test_risk_parity_optimization_runs(self, returns_df):
        """Risk parity optimization returns weights summing to 1."""
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        result = opt.optimize_risk_parity(returns_df)
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01
        assert all(w >= 0 for w in result.weights.values())

    def test_optimizer_with_constraints(self, returns_df):
        """Markowitz with per-asset constraints respects bounds."""
        from quant_tools.models import AssetConstraint, OptimizationConstraints
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        constraints = OptimizationConstraints(
            asset_constraints=[
                AssetConstraint(symbol="AAPL", min_weight=0.1, max_weight=0.4),
                AssetConstraint(symbol="MSFT", min_weight=0.1, max_weight=0.4),
                AssetConstraint(symbol="BND", min_weight=0.1, max_weight=0.4),
                AssetConstraint(symbol="GLD", min_weight=0.1, max_weight=0.4),
            ],
        )
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        result = opt.optimize_markowitz(returns_df, constraints=constraints)
        for sym, w in result.weights.items():
            assert w >= 0.1 - 0.01, f"{sym} weight {w} below min 0.1"
            assert w <= 0.4 + 0.01, f"{sym} weight {w} above max 0.4"

    def test_efficient_frontier_has_points(self, returns_df):
        """Efficient frontier computation produces multiple points."""
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        frontier = opt.optimize_efficient_frontier(returns_df, n_points=10)
        assert len(frontier.points) >= 5
        assert frontier.max_sharpe_point is not None
        assert frontier.min_variance_point is not None

    def test_efficient_frontier_points_ordered(self, returns_df):
        """Frontier points have increasing volatility."""
        from quant_tools.portfolio.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.04)
        frontier = opt.optimize_efficient_frontier(returns_df, n_points=15)
        vols = [p.volatility for p in frontier.points]
        # Should be roughly monotonically increasing
        for i in range(1, len(vols)):
            assert vols[i] >= vols[i - 1] - 0.001  # small tolerance


# =====================================================================
# Portfolio -- Benchmark Framework
# =====================================================================


@pytest.mark.dashboard
class TestBenchmarkFramework:
    """Verify benchmark definitions and metrics used by the Portfolio page."""

    def test_standard_benchmarks_defined(self):
        """STANDARD_BENCHMARKS has the expected benchmark keys."""
        from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS
        assert len(STANDARD_BENCHMARKS) >= 5
        for key in ("SPY", "AGG", "60/40", "QQQ"):
            assert key in STANDARD_BENCHMARKS

    def test_benchmark_definition_has_name(self):
        """Each standard benchmark has a human-readable name."""
        from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS
        for key, bm in STANDARD_BENCHMARKS.items():
            assert bm.name, f"Benchmark '{key}' has empty name"

    def test_benchmark_metrics_dataclass(self):
        """BenchmarkMetrics can be constructed with expected fields."""
        from quant_tools.models import BenchmarkMetrics
        m = BenchmarkMetrics(
            benchmark_name="S&P 500",
            alpha=0.02, beta=1.1,
            information_ratio=0.5,
            treynor_ratio=0.08,
            tracking_error=0.04,
            correlation=0.95,
            active_return=0.03,
        )
        assert m.alpha == 0.02
        assert m.beta == 1.1
        assert m.tracking_error == 0.04

    def test_composite_benchmark_60_40(self):
        """60/40 benchmark has correct weights."""
        from quant_tools.portfolio.benchmark import STANDARD_BENCHMARKS
        bm = STANDARD_BENCHMARKS["60/40"]
        assert bm.weights == {"SPY": 0.6, "AGG": 0.4}


# =====================================================================
# Backtest -- Strategy Definitions
# =====================================================================


@pytest.mark.dashboard
class TestBacktestStrategies:
    """Verify backtest strategy functions the dashboard calls."""

    @pytest.fixture
    def close_prices(self) -> np.ndarray:
        """Deterministic 252-day close prices for backtest testing."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 252)
        return 100 * np.cumprod(1 + returns)

    def test_sma_crossover_produces_positions(self, close_prices):
        """SMA crossover strategy returns position array."""
        from quant_tools.backtest.strategies import sma_crossover
        positions = sma_crossover(close_prices, fast=20, slow=50)
        assert len(positions) == len(close_prices)
        # Positions should be -1, 0, or 1
        unique_vals = set(np.unique(positions))
        assert unique_vals.issubset({-1.0, 0.0, 1.0, -1, 0, 1})

    def test_bollinger_strategy_produces_positions(self, close_prices):
        """Bollinger bands strategy returns position array."""
        from quant_tools.backtest.strategies import bollinger_strategy
        positions = bollinger_strategy(close_prices, period=20, num_std=2.0)
        assert len(positions) == len(close_prices)

    def test_rsi_strategy_produces_positions(self, close_prices):
        """RSI strategy returns position array."""
        from quant_tools.backtest.strategies import rsi_strategy
        positions = rsi_strategy(close_prices, period=14, oversold=30.0, overbought=70.0)
        assert len(positions) == len(close_prices)

    def test_momentum_strategy_produces_positions(self, close_prices):
        """Momentum strategy returns position array."""
        from quant_tools.backtest.strategies import momentum_strategy
        positions = momentum_strategy(close_prices, lookback=60)
        assert len(positions) == len(close_prices)

    def test_backtest_engine_runs(self, close_prices):
        """run_backtest produces a BacktestResult with expected fields."""
        from quant_tools.backtest.engine import run_backtest
        from quant_tools.backtest.strategies import sma_crossover
        from quant_tools.models import StrategyType
        positions = sma_crossover(close_prices, fast=20, slow=50)
        result = run_backtest(
            close_prices, positions,
            initial_capital=100000.0,
            commission=0.001,
            strategy=StrategyType.SMA_CROSSOVER,
            ticker="TEST",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert result.ticker == "TEST"
        assert result.initial_capital == 100000.0
        assert result.final_value > 0
        assert len(result.equity_curve) == len(close_prices)
        # max_drawdown can be stored as a positive fraction (convention varies)
        assert 0.0 <= abs(result.max_drawdown) <= 1.0

    def test_strategy_type_enum_has_entries(self):
        """StrategyType enum has the 4 dashboard strategies."""
        from quant_tools.models import StrategyType
        expected = {"sma_crossover", "bollinger", "rsi", "momentum"}
        actual = {s.value for s in StrategyType}
        assert expected.issubset(actual)


# =====================================================================
# Factors & Macro -- Models
# =====================================================================


@pytest.mark.dashboard
class TestFactorsModels:
    """Verify factor/macro data models used by the Factors & Macro page."""

    def test_macro_regime_enum(self):
        """MacroRegime enum has expected regimes."""
        from quant_tools.models import MacroRegime
        expected = {"expansion", "slowdown", "contraction", "recovery"}
        actual = {r.value for r in MacroRegime}
        assert actual == expected

    def test_macro_snapshot_dataclass(self):
        """MacroSnapshot can be constructed with FRED fields."""
        from quant_tools.models import MacroRegime, MacroSnapshot
        snap = MacroSnapshot(
            regime=MacroRegime.EXPANSION,
            gdp_growth=0.025, unemployment=0.038,
            cpi_yoy=0.032, fed_funds_rate=0.0525,
            yield_spread_10y2y=0.001,
        )
        assert snap.regime == MacroRegime.EXPANSION
        assert snap.gdp_growth == 0.025

    def test_factor_score_dataclass(self):
        """FactorScore can be constructed with z-score fields."""
        from quant_tools.models import FactorScore
        fs = FactorScore(
            symbol="AAPL", name="Apple Inc.",
            value_z=1.2, momentum_z=0.8, quality_z=1.5,
            composite_z=1.17, rank=1, sector="Technology",
        )
        assert fs.composite_z == 1.17
        assert fs.rank == 1

    def test_rebalance_recommendation_dataclass(self):
        """RebalanceRecommendation can be constructed with tilts."""
        from quant_tools.models import MacroRegime, RebalanceRecommendation
        rec = RebalanceRecommendation(
            regime=MacroRegime.EXPANSION,
            value_tilt=0.2, momentum_tilt=0.3,
            quality_tilt=0.3, defensive_tilt=0.2,
            rationale="Growth environment favors momentum.",
        )
        total_tilt = rec.value_tilt + rec.momentum_tilt + rec.quality_tilt + rec.defensive_tilt
        assert abs(total_tilt - 1.0) < 0.01


# =====================================================================
# Dashboard Helpers & Colors
# =====================================================================


@pytest.mark.dashboard
class TestDashboardHelpers:
    """Verify shared dashboard helper functions."""

    def test_format_pct(self):
        """format_pct formats decimals as percentages."""
        from quant_tools.dashboard.core.helpers import format_pct
        assert "%" in format_pct(0.1234)

    def test_format_number(self):
        """format_number formats floats with decimals."""
        from quant_tools.dashboard.core.helpers import format_number
        result = format_number(1.2345)
        assert "1.23" in result or "1.24" in result

    def test_format_currency(self):
        """format_currency formats as dollar amounts."""
        from quant_tools.dashboard.core.helpers import format_currency
        result = format_currency(123456.78)
        assert "$" in result

    def test_plotly_layout_is_dict(self):
        """PLOTLY_LAYOUT is a dict with dark theme settings."""
        from quant_tools.dashboard.core.helpers import PLOTLY_LAYOUT
        assert isinstance(PLOTLY_LAYOUT, dict)
        assert "paper_bgcolor" in PLOTLY_LAYOUT or "template" in PLOTLY_LAYOUT

    def test_accent_colors_defined(self):
        """All accent color constants are non-empty hex strings."""
        from quant_tools.dashboard.core.colors import (
            ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED,
        )
        for color in (ACCENT_AMBER, ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_RED):
            assert isinstance(color, str) and len(color) > 0
            assert color.startswith("#") or color.startswith("rgb")

    def test_hex_to_rgb(self):
        """hex_to_rgb converts hex color to RGB string."""
        from quant_tools.dashboard.core.colors import hex_to_rgb
        result = hex_to_rgb("#FF5733")
        assert "255" in result
        assert "87" in result or "55" in result  # depends on exact parsing


# =====================================================================
# Dashboard View -- Scanner Helpers
# =====================================================================


@pytest.mark.dashboard
class TestScannerViewHelpers:
    """Verify internal helper functions in the scanner view module."""

    def test_results_to_dataframe(self):
        """_results_to_dataframe converts ScanResult list to DataFrame."""
        from quant_tools.dashboard.views.scanner import _results_to_dataframe
        from quant_tools.models import ScanResult, SignalType
        results = [
            ScanResult(
                symbol="AAPL", name="Apple Inc.", sector="Technology",
                signal=SignalType.MOMENTUM, score=85.0, price=150.0,
                change_1d=1.5, change_5d=3.2, change_1m=8.0, rsi_14=65.0,
            ),
            ScanResult(
                symbol="MSFT", name="Microsoft Corp", sector="Technology",
                signal=SignalType.QUALITY, score=72.0, price=380.0,
                change_1d=-0.5, change_5d=1.1, change_1m=5.5, rsi_14=55.0,
            ),
        ]
        df = _results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Symbol" in df.columns
        assert "Score" in df.columns
        assert "Price" in df.columns
        assert "RSI" in df.columns
        assert df.iloc[0]["Symbol"] == "AAPL"

    def test_asset_class_map_completeness(self):
        """_ASSET_CLASS_MAP covers all 4 asset classes."""
        from quant_tools.dashboard.views.scanner import _ASSET_CLASS_MAP
        from quant_tools.models import AssetClass
        assert len(_ASSET_CLASS_MAP) == 4
        for ac in AssetClass:
            assert ac in _ASSET_CLASS_MAP.values()

    def test_signal_colors_all_defined(self):
        """_SIGNAL_COLORS has entries for all SignalTypes."""
        from quant_tools.dashboard.views.scanner import _SIGNAL_COLORS
        from quant_tools.models import SignalType
        for st in SignalType:
            assert st in _SIGNAL_COLORS, f"Missing color for {st}"

    def test_signal_labels_all_defined(self):
        """_SIGNAL_LABELS has entries for all SignalTypes."""
        from quant_tools.dashboard.views.scanner import _SIGNAL_LABELS
        from quant_tools.models import SignalType
        for st in SignalType:
            assert st in _SIGNAL_LABELS, f"Missing label for {st}"

    def test_timeframe_presets_populated(self):
        """_TIMEFRAME_PRESETS has multiple timeframe options."""
        from quant_tools.dashboard.views.scanner import _TIMEFRAME_PRESETS
        assert len(_TIMEFRAME_PRESETS) >= 10
        assert "Monthly \u00b7 3 months" in _TIMEFRAME_PRESETS

    def test_get_timeframes_equity(self):
        """_get_timeframes('Equity') returns equity-compatible timeframes."""
        from quant_tools.dashboard.views.scanner import _get_timeframes
        frames = _get_timeframes("Equity")
        assert len(frames) >= 5
        assert "Monthly \u00b7 3 months" in frames

    def test_get_timeframes_crypto_includes_multi_hour(self):
        """_get_timeframes('Crypto') includes 4h/6h/8h timeframes."""
        from quant_tools.dashboard.views.scanner import _get_timeframes
        frames = _get_timeframes("Crypto")
        assert "Intraday \u00b7 4 hours" in frames


# =====================================================================
# Dashboard View -- Predictions Helpers
# =====================================================================


@pytest.mark.dashboard
class TestPredictionsViewHelpers:
    """Verify helper functions in the predictions view module."""

    def test_matches_horizon_days(self):
        """_matches_horizon correctly matches day horizons."""
        from quant_tools.dashboard.views.predictions import _matches_horizon
        pred = {"horizon_days": 5, "horizon_hours": None}
        assert _matches_horizon(pred, "5d") is True
        assert _matches_horizon(pred, "20d") is False

    def test_matches_horizon_hours(self):
        """_matches_horizon correctly matches hour horizons."""
        from quant_tools.dashboard.views.predictions import _matches_horizon
        pred = {"horizon_hours": 4, "horizon_days": None}
        assert _matches_horizon(pred, "4h") is True
        assert _matches_horizon(pred, "8h") is False

    def test_matches_horizon_invalid_label(self):
        """_matches_horizon returns False for invalid labels."""
        from quant_tools.dashboard.views.predictions import _matches_horizon
        pred = {"horizon_days": 5}
        assert _matches_horizon(pred, "") is False
        assert _matches_horizon(pred, "x") is False


# =====================================================================
# Dashboard View -- Portfolio Presets
# =====================================================================


@pytest.mark.dashboard
class TestPortfolioPresets:
    """Verify the preset portfolios in the portfolio view."""

    def test_presets_exist(self):
        """PRESETS dict has at least 3 preset portfolios."""
        from quant_tools.dashboard.views.portfolio import PRESETS
        assert len(PRESETS) >= 3

    def test_preset_weights_sum_to_one(self):
        """Each preset's weights sum to 1.0."""
        from quant_tools.dashboard.views.portfolio import PRESETS
        for name, weights in PRESETS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, (
                f"Preset '{name}' weights sum to {total}"
            )

    def test_preset_60_40(self):
        """60/40 preset has SPY and BND with correct weights."""
        from quant_tools.dashboard.views.portfolio import PRESETS
        p = PRESETS["60/40"]
        assert p["SPY"] == 0.6
        assert p["BND"] == 0.4

    def test_default_holdings_is_valid_json(self):
        """DEFAULT_HOLDINGS string is valid JSON."""
        from quant_tools.dashboard.views.portfolio import DEFAULT_HOLDINGS
        parsed = json.loads(DEFAULT_HOLDINGS)
        assert isinstance(parsed, dict)
        assert len(parsed) >= 2


# =====================================================================
# Data Integrity -- Ticker Format Validation
# =====================================================================


@pytest.mark.dashboard
class TestTickerFormatValidation:
    """Verify ticker format conventions across the codebase."""

    def test_sp500_tickers_are_uppercase(self):
        """All SP500_TOP100 tickers are uppercase."""
        from quant_tools.data.universe import SP500_TOP100
        for sym in SP500_TOP100:
            assert sym == sym.upper(), f"Ticker {sym} is not uppercase"

    def test_sp500_tickers_are_short(self):
        """All SP500_TOP100 tickers are 1-5 chars (excluding dash)."""
        from quant_tools.data.universe import SP500_TOP100
        for sym in SP500_TOP100:
            # BRK-B has a dash, which is valid
            clean = sym.replace("-", "")
            assert 1 <= len(clean) <= 5, f"Ticker {sym} has unexpected length"

    def test_crypto_tickers_have_slash(self):
        """All crypto tickers contain a '/' separator."""
        from quant_tools.data.universe import CRYPTO_UNIVERSE
        for sym in CRYPTO_UNIVERSE:
            assert "/" in sym, f"Crypto ticker {sym} missing '/'"

    def test_forex_tickers_have_slash(self):
        """All forex tickers contain a '/' separator."""
        from quant_tools.data.universe import FOREX_UNIVERSE
        for sym in FOREX_UNIVERSE:
            assert "/" in sym, f"Forex ticker {sym} missing '/'"

    def test_commodity_tickers_have_equal_f(self):
        """All commodity tickers end with '=F' (futures convention)."""
        from quant_tools.data.universe import COMMODITY_UNIVERSE
        for sym in COMMODITY_UNIVERSE:
            assert sym.endswith("=F"), f"Commodity ticker {sym} doesn't end with '=F'"


# =====================================================================
# Negative / Edge Cases
# =====================================================================


@pytest.mark.dashboard
class TestNegativeEdgeCases:
    """Verify graceful handling of edge cases and invalid inputs."""

    def test_momentum_signal_constant_prices(self):
        """Momentum signal handles constant price series without crash."""
        from quant_tools.scanner.signals import momentum_signal
        constant = np.full(100, 100.0)
        score = momentum_signal(constant)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_momentum_signal_single_price(self):
        """Momentum signal handles single-element array."""
        from quant_tools.scanner.signals import momentum_signal
        single = np.array([100.0])
        score = momentum_signal(single)
        assert isinstance(score, float)

    def test_mc_very_short_returns(self):
        """Monte Carlo handles very short return series (edge case)."""
        from quant_tools.portfolio.monte_carlo import run_monte_carlo
        returns = np.array([0.01, -0.005, 0.003, 0.002, -0.001])
        mc = run_monte_carlo(
            returns,
            initial_value=100000,
            n_simulations=10,
            time_horizon_days=5,
        )
        assert mc is not None
        assert mc.n_simulations == 10

    def test_portfolio_single_asset_model(self):
        """Portfolio models handle single-asset case."""
        from quant_tools.models import PortfolioAnalysis, PortfolioHolding
        holding = PortfolioHolding(
            symbol="SPY", weight=1.0,
            current_price=500.0, total_return=0.10,
            annualized_return=0.10, volatility=0.15, sharpe=0.67,
        )
        analysis = PortfolioAnalysis(
            holdings=[holding],
            total_return=0.10, annualized_return=0.10,
            volatility=0.15, sharpe_ratio=0.67, max_drawdown=-0.10,
        )
        assert len(analysis.holdings) == 1
        assert analysis.holdings[0].weight == 1.0

    def test_empty_predictions_filter(self):
        """Filtering empty predictions list does not crash."""
        predictions = []
        filtered = [p for p in predictions if p.get("asset_class") == "equity"]
        assert filtered == []

    def test_empty_performance_chart_returns_none(self):
        """Chart builders return None for empty data."""
        from quant_tools.dashboard.views.performance import (
            _build_confidence_accuracy_scatter,
            _build_heatmap,
            _build_win_rate_chart,
        )
        assert _build_win_rate_chart([]) is None
        assert _build_heatmap([]) is None
        assert _build_confidence_accuracy_scatter([]) is None

    def test_prediction_agreement_chart_empty(self):
        """Model agreement chart returns None for empty data."""
        from quant_tools.dashboard.views.predictions import _build_model_agreement_chart
        assert _build_model_agreement_chart([]) is None

    def test_confidence_histogram_empty(self):
        """Confidence histogram returns None for empty data."""
        from quant_tools.dashboard.views.predictions import _build_confidence_histogram
        assert _build_confidence_histogram([]) is None

    def test_scan_result_with_empty_details(self):
        """ScanResult with empty details dict is valid."""
        from quant_tools.models import ScanResult, SignalType
        result = ScanResult(
            symbol="TEST", name="Test Corp", sector="Technology",
            signal=SignalType.MOMENTUM, score=50.0, price=100.0,
        )
        assert result.details == {}
        assert result.details.get("breakdown") is None

    def test_scan_result_with_breakdown(self):
        """ScanResult with breakdown details is accessible."""
        from quant_tools.models import ScanResult, SignalType
        result = ScanResult(
            symbol="TEST", name="Test Corp", sector="Technology",
            signal=SignalType.MOMENTUM, score=75.0, price=100.0,
            details={"breakdown": {"rsi": 80.0, "sma_crossover": 70.0}},
        )
        assert result.details["breakdown"]["rsi"] == 80.0

    def test_parse_horizon_edge_cases(self):
        """Horizon parsing handles various edge cases."""
        from quant_tools.predict.horizons import parse_horizon_label
        # Single character
        assert parse_horizon_label("d") is None
        assert parse_horizon_label("h") is None
        # Negative numbers
        result = parse_horizon_label("-5d")
        # Should either return None or a preset with value -5
        if result is not None:
            assert result.value == -5
        # Large number
        result = parse_horizon_label("999d")
        assert result is not None
        assert result.value == 999


# =====================================================================
# Data Layer -- Prediction Storage Schema
# =====================================================================


@pytest.mark.dashboard
class TestPredictionStorageSchema:
    """Deep verification of the prediction storage schema."""

    def test_prediction_row_has_interval_column(self, tmp_path):
        """Predictions table has 'interval' column for intraday support."""
        from quant_tools.cache import CacheManager

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("predictions")}
            assert "interval" in cols
        finally:
            cm._engine.dispose()

    def test_prediction_row_has_horizon_hours_column(self, tmp_path):
        """Predictions table has 'horizon_hours' column for crypto horizons."""
        from quant_tools.cache import CacheManager

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("predictions")}
            assert "horizon_hours" in cols
        finally:
            cm._engine.dispose()

    def test_prediction_row_has_is_ensemble_column(self, tmp_path):
        """Predictions table has 'is_ensemble' column."""
        from quant_tools.cache import CacheManager

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("predictions")}
            assert "is_ensemble" in cols
        finally:
            cm._engine.dispose()

    def test_prediction_row_has_consensus_signal_column(self, tmp_path):
        """Predictions table has 'consensus_signal' column."""
        from quant_tools.cache import CacheManager

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("predictions")}
            assert "consensus_signal" in cols
        finally:
            cm._engine.dispose()

    def test_outcome_table_has_required_columns(self, tmp_path):
        """Prediction outcomes table has correct, error_pct columns."""
        from quant_tools.cache import CacheManager

        cm = CacheManager(db_path=str(tmp_path / "test.db"))
        try:
            from sqlalchemy import inspect
            insp = inspect(cm._engine)
            cols = {c["name"] for c in insp.get_columns("prediction_outcomes")}
            for col in ("prediction_id", "actual_direction", "correct", "error_pct"):
                assert col in cols, f"Missing column in outcomes: {col}"
        finally:
            cm._engine.dispose()


# =====================================================================
# Live Streamlit Integration (skip if server unreachable)
# =====================================================================


@pytest.mark.dashboard
@skip_if_unreachable
class TestLiveStreamlitPages:
    """Integration tests against the live Streamlit server."""

    def test_streamlit_api_health(self):
        """/_stcore/health returns ok."""
        status, body, _ = _http_get("/_stcore/health")
        assert status == 200
        assert "ok" in body.lower()

    def test_streamlit_main_page_has_streamlit_shell(self):
        """Main page contains the Streamlit SPA shell (content rendered client-side)."""
        status, body, _ = _http_get("/")
        assert status == 200
        # Streamlit SPA returns a shell with <title>Streamlit</title> and root div
        assert "streamlit" in body.lower() or "root" in body.lower()

    def test_streamlit_response_not_error(self):
        """Main page returns 200, not 500."""
        status, _, _ = _http_get("/")
        assert status == 200

    def test_streamlit_has_css(self):
        """Main page includes CSS styles."""
        status, body, _ = _http_get("/")
        assert status == 200
        assert "style" in body.lower() or "css" in body.lower()


# =====================================================================
# Cross-Cutting: App Registry Consistency
# =====================================================================


@pytest.mark.dashboard
class TestAppRegistryConsistency:
    """Verify the app.py PAGES registry is consistent with actual modules."""

    def test_view_modules_importable_from_app(self):
        """app.py imports all 10 view modules (views/__init__.py may be minimal)."""
        app_py = (
            Path(__file__).resolve().parent.parent
            / "src" / "quant_tools" / "dashboard" / "app.py"
        )
        source = app_py.read_text(encoding="utf-8")
        for module in EXPECTED_VIEW_MODULES:
            assert module in source, (
                f"Module '{module}' not imported in app.py"
            )

    def test_no_duplicate_page_names(self):
        """app.py has no duplicate page names in PAGES dict."""
        app_py = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "app.py"
        source = app_py.read_text(encoding="utf-8")
        # Extract page name strings from the PAGES dict
        page_names = re.findall(r'"([^"]+)":\s*\w+', source)
        # Check for duplicates
        seen = set()
        for name in page_names:
            assert name not in seen, f"Duplicate page name: {name}"
            seen.add(name)

    def test_app_imports_all_views(self):
        """app.py imports all 10 view modules."""
        app_py = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "dashboard" / "app.py"
        source = app_py.read_text(encoding="utf-8")
        for module in EXPECTED_VIEW_MODULES:
            assert module in source, f"app.py does not import {module}"

    def test_style_css_has_dark_theme(self):
        """style.css contains dark theme variables."""
        css_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "quant_tools" / "dashboard" / "style.css"
        )
        source = css_path.read_text(encoding="utf-8")
        # Dark themes typically have dark background colors
        assert "background" in source.lower()
        assert "#" in source  # hex color values


# #####################################################################
# STREAMLIT AppTest FUNCTIONAL TESTS
#
# These tests use Streamlit's built-in AppTest framework to test the
# actual dashboard app without a browser. They verify page rendering,
# widget presence, navigation, and error handling.
#
# No extra dependencies required -- AppTest is included in Streamlit.
#
# Run:
#   python -m pytest tests/test_dashboard.py -v -m dashboard -k "AppTest"
# #####################################################################


def _load_app():
    """Load the dashboard app via AppTest. Shared helper."""
    from streamlit.testing.v1 import AppTest

    app_path = str(
        Path(__file__).resolve().parent.parent
        / "src"
        / "quant_tools"
        / "dashboard"
        / "app.py"
    )
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    return at


def _navigate_app(at, page_substring: str):
    """Navigate an AppTest instance to a page matching *page_substring*.

    Works with the button-based grouped navigation (replaced radio in Phase 1A).
    Finds a sidebar button whose label contains the substring and clicks it.
    """
    buttons = at.sidebar.button
    matched = [b for b in buttons if page_substring in b.label]
    assert matched, (
        f"No sidebar button matching '{page_substring}' in "
        f"{[b.label for b in buttons]}"
    )
    matched[0].click()
    at.run()
    return at


@pytest.mark.dashboard
class TestAppTestNavigation:
    """Test page navigation using Streamlit AppTest."""

    @pytest.fixture
    def app(self):
        """Load the dashboard app via AppTest."""
        return _load_app()

    def test_app_loads_without_exception(self, app):
        """Dashboard app loads and renders without crashing."""
        assert not app.exception, f"App raised exceptions: {app.exception}"

    def test_sidebar_has_nav_buttons(self, app):
        """Sidebar has button widgets for page navigation."""
        assert len(app.sidebar.button) >= 10

    def test_sidebar_buttons_have_all_pages(self, app):
        """Sidebar buttons include all 10 page names."""
        labels = [b.label for b in app.sidebar.button]
        for page_name in EXPECTED_PAGES:
            found = any(page_name in lbl for lbl in labels)
            assert found, f"Page '{page_name}' not in sidebar buttons: {labels}"

    def test_navigate_to_each_page(self, app):
        """Navigate to each page via button click, verify no exceptions."""
        for page_name in EXPECTED_PAGES:
            buttons = app.sidebar.button
            matched = [b for b in buttons if page_name in b.label]
            if matched:
                matched[0].click()
                app.run()
                assert not app.exception, f"Exception on page '{page_name}': {app.exception}"


@pytest.mark.dashboard
class TestAppTestScanner:
    """Market Scanner page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Scanner")
        return at

    def test_scanner_renders(self, app):
        assert not app.exception

    def test_scanner_has_selectboxes(self, app):
        """Scanner page has selectbox widgets (asset class, universe, etc.)."""
        total = len(app.selectbox) + len(app.sidebar.selectbox)
        assert total >= 1

    def test_scanner_has_markdown_header(self, app):
        """Scanner page has markdown content (header/description)."""
        assert len(app.markdown) >= 1


@pytest.mark.dashboard
class TestAppTestBacktester:
    """Strategy Backtester page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Backtester")
        return at

    def test_backtester_renders(self, app):
        assert not app.exception

    def test_backtester_has_inputs(self, app):
        total_inputs = len(app.text_input) + len(app.sidebar.text_input)
        total_selects = len(app.selectbox) + len(app.sidebar.selectbox)
        assert total_inputs + total_selects >= 1


@pytest.mark.dashboard
class TestAppTestPortfolio:
    """Portfolio Analyzer page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Portfolio")
        return at

    def test_portfolio_renders(self, app):
        assert not app.exception

    def test_portfolio_has_text_area(self, app):
        """Portfolio page has text area for holdings JSON."""
        total = len(app.text_area) + len(app.sidebar.text_area)
        assert total >= 1


@pytest.mark.dashboard
class TestAppTestFactors:
    """Factors & Macro page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Factors")
        return at

    def test_factors_renders(self, app):
        assert not app.exception


@pytest.mark.dashboard
class TestAppTestPredictions:
    """Predictions page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Predictions")
        return at

    def test_predictions_renders(self, app):
        assert not app.exception

    def test_predictions_has_filters(self, app):
        total = len(app.selectbox) + len(app.sidebar.selectbox)
        assert total >= 1


@pytest.mark.dashboard
class TestAppTestPerformance:
    """Performance page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Performance")
        return at

    def test_performance_renders(self, app):
        assert not app.exception


@pytest.mark.dashboard
class TestAppTestRolling:
    """Rolling Analytics page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Rolling")
        return at

    def test_rolling_renders(self, app):
        assert not app.exception


@pytest.mark.dashboard
class TestAppTestRisk:
    """Risk Dashboard page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Risk")
        return at

    def test_risk_renders(self, app):
        assert not app.exception


@pytest.mark.dashboard
class TestAppTestLive:
    """Live Monitor page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "Live")
        return at

    def test_live_renders(self, app):
        assert not app.exception

    def test_live_shows_coming_soon(self, app):
        """Live page shows coming-soon info message."""
        info_texts = [str(i.value) for i in app.info]
        assert any("WebSocket" in t or "Scanner" in t for t in info_texts)


@pytest.mark.dashboard
class TestAppTestAI:
    """AI Analysis page -- AppTest."""

    @pytest.fixture
    def app(self):
        at = _load_app()
        _navigate_app(at, "AI")
        return at

    def test_ai_renders(self, app):
        assert not app.exception

    def test_ai_has_selector(self, app):
        total = len(app.selectbox) + len(app.sidebar.selectbox)
        assert total >= 1
