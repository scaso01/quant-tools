"""Tests for universe_loader.py and expanded universe presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_tools.data.universe_loader import (
    available_universes,
    clear_cache,
    get_ticker_metadata,
    load_combined,
    load_universe,
)
from quant_tools.data.universe import (
    SP500_TOP100,
    UNIVERSE_PRESETS,
    _EXTENDED_PRESET_BUILDERS,
    _extended_cache,
    get_universe,
)
from quant_tools.models import TickerMeta

UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "src" / "quant_tools" / "data" / "universes"


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear loader and preset caches before each test."""
    clear_cache()
    _extended_cache.clear()
    yield
    clear_cache()
    _extended_cache.clear()


# ── JSON File Tests ────────────────────────────────────────────────────


class TestJsonFiles:
    """Verify the universe JSON files exist and have correct structure."""

    def test_sp500_json_exists(self):
        assert (UNIVERSES_DIR / "sp500.json").exists()

    def test_russell1000_json_exists(self):
        assert (UNIVERSES_DIR / "russell1000.json").exists()

    def test_international_json_exists(self):
        assert (UNIVERSES_DIR / "international.json").exists()

    def test_sp500_entry_count(self):
        data = json.loads((UNIVERSES_DIR / "sp500.json").read_text())
        assert len(data) >= 500, f"S&P 500 has {len(data)} entries, expected >= 500"

    def test_russell1000_entry_count(self):
        data = json.loads((UNIVERSES_DIR / "russell1000.json").read_text())
        assert len(data) >= 900, f"Russell 1000 has {len(data)} entries, expected >= 900"

    def test_international_entry_count(self):
        data = json.loads((UNIVERSES_DIR / "international.json").read_text())
        assert len(data) >= 70, f"International has {len(data)} entries, expected >= 70"

    def test_sp500_json_schema(self):
        """Each entry must have name, sector, sub_industry, region."""
        data = json.loads((UNIVERSES_DIR / "sp500.json").read_text())
        for symbol, meta in list(data.items())[:10]:
            assert "name" in meta, f"{symbol} missing 'name'"
            assert "sector" in meta, f"{symbol} missing 'sector'"
            assert "sub_industry" in meta, f"{symbol} missing 'sub_industry'"
            assert "region" in meta, f"{symbol} missing 'region'"

    def test_international_has_multiple_regions(self):
        data = json.loads((UNIVERSES_DIR / "international.json").read_text())
        regions = {v["region"] for v in data.values()}
        assert "Europe" in regions
        assert "Emerging" in regions


# ── Loader Tests ───────────────────────────────────────────────────────


class TestLoadUniverse:
    """Test load_universe function."""

    def test_load_sp500(self):
        result = load_universe("sp500")
        assert isinstance(result, dict)
        assert len(result) >= 500

    def test_load_returns_ticker_meta(self):
        result = load_universe("sp500")
        meta = result["AAPL"]
        assert isinstance(meta, TickerMeta)
        assert meta.name == "Apple Inc."
        assert meta.sector == "Information Technology"
        assert meta.region == "US"

    def test_load_is_cached(self):
        first = load_universe("sp500")
        second = load_universe("sp500")
        assert first is second  # Same object reference

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_universe("nonexistent")

    def test_load_russell1000(self):
        result = load_universe("russell1000")
        assert len(result) >= 900

    def test_load_international(self):
        result = load_universe("international")
        assert len(result) >= 70


class TestLoadCombined:
    """Test load_combined function."""

    def test_combined_deduplicates(self):
        """Overlapping tickers between sp500 and russell1000 should be deduped."""
        sp = load_universe("sp500")
        r1 = load_universe("russell1000")
        combined = load_combined(["sp500", "russell1000"])
        # Combined should be <= sum of individuals (due to overlap)
        assert len(combined) <= len(sp) + len(r1)
        # But should have at least as many as the larger set
        assert len(combined) >= max(len(sp), len(r1))

    def test_combined_all_three(self):
        combined = load_combined(["sp500", "russell1000", "international"])
        assert len(combined) > 1000

    def test_combined_empty_list(self):
        result = load_combined([])
        assert result == {}


class TestGetTickerMetadata:
    """Test get_ticker_metadata function."""

    def test_find_known_ticker(self):
        meta = get_ticker_metadata("AAPL")
        assert meta is not None
        assert meta.name == "Apple Inc."

    def test_find_international_ticker(self):
        meta = get_ticker_metadata("TSM")
        assert meta is not None
        assert "Taiwan" in meta.name

    def test_unknown_ticker_returns_none(self):
        result = get_ticker_metadata("ZZZZZZZ")
        assert result is None


class TestAvailableUniverses:
    """Test available_universes function."""

    def test_lists_all_json_files(self):
        names = available_universes()
        assert "sp500" in names
        assert "russell1000" in names
        assert "international" in names

    def test_returns_sorted(self):
        names = available_universes()
        assert names == sorted(names)


# ── Extended Preset Tests ──────────────────────────────────────────────


class TestExtendedPresets:
    """Test the new lazy-loaded presets in universe.py."""

    def test_sp500_top100_unchanged(self):
        """SP500_TOP100 must remain exactly 100 entries."""
        assert len(SP500_TOP100) == 100

    def test_existing_presets_unchanged(self):
        """Original 6 presets must still exist."""
        assert "S&P 500 Top 100" in UNIVERSE_PRESETS
        assert "Magnificent 7" in UNIVERSE_PRESETS
        assert "Mega Cap 30" in UNIVERSE_PRESETS
        assert "Dow 30" in UNIVERSE_PRESETS
        assert "Growth & Tech" in UNIVERSE_PRESETS
        assert "Defensive & Income" in UNIVERSE_PRESETS

    def test_extended_presets_registered(self):
        assert "S&P 500" in _EXTENDED_PRESET_BUILDERS
        assert "Russell 1000" in _EXTENDED_PRESET_BUILDERS
        assert "International ADRs" in _EXTENDED_PRESET_BUILDERS
        assert "Emerging Market ADRs" in _EXTENDED_PRESET_BUILDERS
        assert "Global All" in _EXTENDED_PRESET_BUILDERS

    def test_get_universe_sp500_full(self):
        result = get_universe("S&P 500")
        assert len(result) >= 500

    def test_get_universe_russell1000(self):
        result = get_universe("Russell 1000")
        assert len(result) >= 900

    def test_get_universe_eafe(self):
        """EAFE = Europe + Asia Pacific (no Emerging)."""
        result = get_universe("International ADRs")
        assert len(result) >= 30
        # Verify no emerging market tickers leaked in
        intl = load_universe("international")
        for sym in result:
            meta = intl.get(sym)
            if meta is not None:
                assert meta.region in ("Europe", "Asia Pacific"), (
                    f"{sym} has region {meta.region}, expected Europe/Asia Pacific"
                )

    def test_get_universe_emerging(self):
        result = get_universe("Emerging Market ADRs")
        assert len(result) >= 20
        intl = load_universe("international")
        for sym in result:
            meta = intl.get(sym)
            if meta is not None:
                assert meta.region == "Emerging"

    def test_get_universe_global_all(self):
        result = get_universe("Global All")
        assert len(result) >= 1000

    def test_sp500_resolves_to_the_full_index(self):
        """'sp500' means the S&P 500, not the top hundred of it.

        It was special-cased to SP500_TOP100 while the full list sat right
        beside it, so the documented `--universe sp500` scanned 100 stocks.
        """
        result = get_universe("sp500")
        assert result == get_universe("S&P 500")
        assert len(result) > 500
        # The hundred-name cut is still available, just not under this name.
        assert len(get_universe("sp100")) == 100

    def test_unknown_universe_lists_extended(self):
        """ValueError message should include extended preset names."""
        with pytest.raises(ValueError, match="S&P 500"):
            get_universe("definitely_not_real")

    def test_ticker_meta_has_sub_industry(self):
        """Loaded TickerMeta should include sub_industry field."""
        meta = get_ticker_metadata("MSFT")
        assert meta is not None
        assert meta.sub_industry != ""

    def test_ticker_meta_index_membership_default(self):
        """index_membership should default to empty list."""
        meta = get_ticker_metadata("AAPL")
        assert meta is not None
        assert meta.index_membership == []
