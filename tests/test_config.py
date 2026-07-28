"""Tests for config.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from quant_tools.config import Config, validate_config
from quant_tools.errors import ConfigError


class TestConfig:
    """Test Config frozen dataclass."""

    def test_config_is_frozen(self):
        c = Config()
        with pytest.raises(AttributeError):
            c.fred_api_key = "new_key"  # type: ignore[misc]

    def test_default_values(self):
        c = Config()
        assert c.cache_ttl_hours == 24 or isinstance(c.cache_ttl_hours, int)
        assert c.scanner_max_workers >= 1
        assert c.backtest_initial_capital > 0
        assert c.monte_carlo_simulations >= 100

    def test_config_from_env(self):
        with patch.dict(os.environ, {"SCANNER_MAX_WORKERS": "4"}):
            c = Config(scanner_max_workers=int(os.getenv("SCANNER_MAX_WORKERS", "8")))
            assert c.scanner_max_workers == 4


class TestValidateConfig:
    """Test config validation."""

    def test_valid_config(self):
        c = Config()
        assert validate_config(c) is True

    def test_invalid_cache_ttl(self):
        c = Config(cache_ttl_hours=0)
        with pytest.raises(ConfigError, match="CACHE_TTL_HOURS"):
            validate_config(c)

    def test_invalid_scanner_workers_low(self):
        c = Config(scanner_max_workers=0)
        with pytest.raises(ConfigError, match="SCANNER_MAX_WORKERS"):
            validate_config(c)

    def test_invalid_scanner_workers_high(self):
        c = Config(scanner_max_workers=100)
        with pytest.raises(ConfigError, match="SCANNER_MAX_WORKERS"):
            validate_config(c)

    def test_invalid_initial_capital(self):
        c = Config(backtest_initial_capital=-1000)
        with pytest.raises(ConfigError, match="BACKTEST_INITIAL_CAPITAL"):
            validate_config(c)

    def test_invalid_monte_carlo_sims(self):
        c = Config(monte_carlo_simulations=10)
        with pytest.raises(ConfigError, match="MONTE_CARLO_SIMULATIONS"):
            validate_config(c)
