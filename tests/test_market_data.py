"""Tests for data/market_data.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.errors import DataError, NetworkError


class TestFetchPrices:
    """Test fetch_prices function."""

    def test_fetch_prices_from_cache(self, tmp_db):
        """Test that cached data is returned without API call."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data

        cache = CacheManager(tmp_db)
        market_data._cache = cache

        # Pre-populate cache
        prices = [
            {"date": "2024-01-02", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000000, "adj_close": 102},
            {"date": "2024-01-03", "open": 102, "high": 106, "low": 101, "close": 105, "volume": 1200000, "adj_close": 105},
        ]
        cache.store_prices("AAPL", prices)

        result = market_data.fetch_prices("AAPL", start="2024-01-02", end="2024-01-03")
        assert len(result) == 2
        assert "Close" in result.columns

    def test_fetch_prices_network_error(self, tmp_db):
        """Test network error handling."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data
        from quant_tools.retry import yfinance_breaker

        cache = CacheManager(tmp_db)
        market_data._cache = cache
        yfinance_breaker.reset()

        with patch("quant_tools.data.market_data.yf") as mock_yf:
            mock_yf.download.side_effect = ConnectionError("Network error")
            with pytest.raises(NetworkError):
                market_data.fetch_prices("AAPL", period="1d")

    def test_fetch_prices_empty_result(self, tmp_db):
        """Test empty result handling."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data
        from quant_tools.retry import yfinance_breaker

        cache = CacheManager(tmp_db)
        market_data._cache = cache
        yfinance_breaker.reset()

        with patch("quant_tools.data.market_data.yf") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            with pytest.raises(DataError, match="No price data"):
                market_data.fetch_prices("FAKE", period="1d")

    def test_fetch_prices_breaker_open(self, tmp_db):
        """Test circuit breaker blocks calls."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data
        from quant_tools.retry import yfinance_breaker

        cache = CacheManager(tmp_db)
        market_data._cache = cache

        # Force breaker open
        for _ in range(10):
            yfinance_breaker.record_failure()

        with pytest.raises(NetworkError, match="circuit breaker"):
            market_data.fetch_prices("AAPL", period="1d")

        yfinance_breaker.reset()


class TestFetchTickerInfo:
    """Test fetch_ticker_info function."""

    def test_fetch_info_from_cache(self, tmp_db, sample_ticker_info):
        """Test cached metadata retrieval."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data

        cache = CacheManager(tmp_db)
        market_data._cache = cache

        cache.store_metadata("AAPL", sample_ticker_info)
        result = market_data.fetch_ticker_info("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["sector"] == "Technology"

    def test_fetch_info_api_failure(self, tmp_db):
        """Test API failure handling."""
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data

        cache = CacheManager(tmp_db)
        market_data._cache = cache

        with patch("quant_tools.data.market_data.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("API error")
            with pytest.raises(DataError, match="Failed to fetch"):
                market_data.fetch_ticker_info("FAKE")
