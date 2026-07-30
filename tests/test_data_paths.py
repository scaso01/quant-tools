"""Regression tests for where the cache and the log file are written.

Both defaulted to a bare filename, which resolves against the working
directory. Measured on a fresh install: `quant-tools web` launched from C:\\
died with PermissionError on 'C:\\quant_tools.log' before the dashboard came
up. From a writable directory it worked, but dropped a log file and a separate
copy of the price cache in whatever folder happened to be current.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from quant_tools.config import _data_dir


class TestDataDir:
    """The per-user location the defaults point at."""

    def test_is_absolute(self):
        assert _data_dir().is_absolute()

    def test_is_not_the_working_directory(self):
        assert _data_dir() != Path.cwd()

    def test_named_for_the_app(self):
        assert _data_dir().name == "quant-tools"

    @pytest.mark.skipif(os.name != "nt", reason="Windows layout")
    def test_windows_uses_localappdata(self, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\test\AppData\Local")
        assert _data_dir() == Path(r"C:\Users\test\AppData\Local\quant-tools")

    @pytest.mark.skipif(
        os.name == "nt" or sys.platform == "darwin", reason="XDG layout"
    )
    def test_linux_honours_xdg(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg")
        assert _data_dir() == Path("/tmp/xdg/quant-tools")


class TestDefaultsAreAbsolute:
    """A relative default is what made this cwd-dependent.

    Tests the default functions, not the live cfg: a developer's own .env
    legitimately overrides both, and the suite must not depend on its contents.
    """

    def test_cache_path_is_absolute(self, monkeypatch):
        from quant_tools.config import default_cache_db_path

        monkeypatch.delenv("CACHE_DB_PATH", raising=False)
        assert Path(default_cache_db_path()).is_absolute()

    def test_log_path_is_absolute(self, monkeypatch):
        from quant_tools.config import default_log_file

        monkeypatch.delenv("LOG_FILE", raising=False)
        assert Path(default_log_file()).is_absolute()

    def test_env_override_still_wins(self, monkeypatch):
        from quant_tools.config import default_cache_db_path

        monkeypatch.setenv("CACHE_DB_PATH", "/somewhere/else.db")
        assert default_cache_db_path() == "/somewhere/else.db"


class TestSecretsStayOutOfRepr:
    """A failing assertion prints the whole Config, and CI logs are public."""

    def test_api_keys_are_not_in_repr(self):
        from quant_tools.config import Config

        cfg = Config(
            fred_api_key="FRED-SECRET",
            finnhub_api_key="FINNHUB-SECRET",
            reddit_client_id="REDDIT-ID",
            reddit_client_secret="REDDIT-SECRET",
            alpaca_api_key="ALPACA-KEY",
            alpaca_secret_key="ALPACA-SECRET",
        )
        text = repr(cfg)
        for secret in (
            "FRED-SECRET",
            "FINNHUB-SECRET",
            "REDDIT-ID",
            "REDDIT-SECRET",
            "ALPACA-KEY",
            "ALPACA-SECRET",
        ):
            assert secret not in text

    def test_values_are_still_readable(self):
        from quant_tools.config import Config

        assert Config(fred_api_key="FRED-SECRET").fred_api_key == "FRED-SECRET"


class TestWritersCreateTheirDirectory:
    """Neither sqlite nor RotatingFileHandler will make a missing parent."""

    def test_cache_creates_missing_parent(self, tmp_path):
        from quant_tools.cache import CacheManager

        db = tmp_path / "nested" / "deeper" / "cache.db"
        CacheManager(str(db))
        assert db.exists()

    def test_unwritable_log_path_names_the_setting(self, tmp_path):
        """An old .env still points at a relative path. Say so, do not traceback."""
        from quant_tools.errors import ConfigError
        from quant_tools.logging_config import setup_logging

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("this is a file, so it cannot hold a log file")

        logger = logging.getLogger("quant_tools")
        saved = logger.handlers[:]
        logger.handlers = []
        try:
            with pytest.raises(ConfigError, match="LOG_FILE"):
                setup_logging(str(blocker / "quant_tools.log"))
        finally:
            for h in logger.handlers:
                h.close()
            logger.handlers = saved

    def test_logging_creates_missing_parent(self, tmp_path):
        from quant_tools.logging_config import setup_logging

        target = tmp_path / "nested" / "deeper" / "quant_tools.log"
        logger = logging.getLogger("quant_tools")
        saved = logger.handlers[:]
        logger.handlers = []
        try:
            setup_logging(str(target))
            logger.info("regression check")
            for h in logger.handlers:
                h.flush()
            assert target.exists()
        finally:
            for h in logger.handlers:
                h.close()
            logger.handlers = saved
