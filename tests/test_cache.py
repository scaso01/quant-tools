"""Tests for cache.py."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from quant_tools.cache import CacheManager, atomic_write_json


class TestCacheManager:
    """Test SQLite cache manager."""

    def test_store_and_retrieve_prices(self, tmp_db):
        cache = CacheManager(tmp_db)
        prices = [
            {"date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000000, "adj_close": 102},
            {"date": "2024-01-03", "open": 102, "high": 106, "low": 101, "close": 105, "volume": 1200000, "adj_close": 105},
        ]
        cache.store_prices("AAPL", prices)
        result = cache.get_prices("AAPL", "2024-01-02", "2024-01-03")
        assert result is not None
        assert len(result) == 2
        assert result[0]["close"] == 102
        assert result[1]["close"] == 105

    def test_cache_miss(self, tmp_db):
        cache = CacheManager(tmp_db)
        result = cache.get_prices("FAKE", "2024-01-01", "2024-12-31")
        assert result is None

    def test_store_and_retrieve_metadata(self, tmp_db):
        cache = CacheManager(tmp_db)
        data = {"symbol": "AAPL", "sector": "Technology", "pe_ratio": 30.5}
        cache.store_metadata("AAPL", data)
        result = cache.get_metadata("AAPL")
        assert result is not None
        assert result["sector"] == "Technology"
        assert result["pe_ratio"] == 30.5

    def test_metadata_update(self, tmp_db):
        cache = CacheManager(tmp_db)
        cache.store_metadata("AAPL", {"pe": 30})
        cache.store_metadata("AAPL", {"pe": 35})
        result = cache.get_metadata("AAPL")
        assert result["pe"] == 35

    def test_clear_cache(self, tmp_db):
        cache = CacheManager(tmp_db)
        cache.store_metadata("AAPL", {"test": True})
        cache.clear()
        assert cache.get_metadata("AAPL") is None

    def test_case_insensitive_symbols(self, tmp_db):
        cache = CacheManager(tmp_db)
        cache.store_metadata("aapl", {"test": True})
        result = cache.get_metadata("AAPL")
        assert result is not None

    def test_price_overwrite(self, tmp_db):
        cache = CacheManager(tmp_db)
        prices1 = [{"date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000000, "adj_close": 102}]
        prices2 = [{"date": "2024-01-02", "open": 200, "high": 205, "low": 199, "close": 202, "volume": 2000000, "adj_close": 202}]
        cache.store_prices("AAPL", prices1)
        cache.store_prices("AAPL", prices2)
        result = cache.get_prices("AAPL", "2024-01-02", "2024-01-02")
        assert result[0]["close"] == 202


class TestAtomicWriteJson:
    """Test atomic JSON write utility."""

    def test_atomic_write(self, tmp_path):
        filepath = str(tmp_path / "test.json")
        data = {"key": "value", "number": 42}
        atomic_write_json(filepath, data)
        with open(filepath, "r") as f:
            result = json.load(f)
        assert result == data

    def test_atomic_write_overwrites(self, tmp_path):
        filepath = str(tmp_path / "test.json")
        atomic_write_json(filepath, {"old": True})
        atomic_write_json(filepath, {"new": True})
        with open(filepath, "r") as f:
            result = json.load(f)
        assert result == {"new": True}

    def test_atomic_write_no_partial(self, tmp_path):
        """Verify no temp files left on success."""
        filepath = str(tmp_path / "test.json")
        atomic_write_json(filepath, {"data": True})
        # Only the final file should exist
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "test.json"
