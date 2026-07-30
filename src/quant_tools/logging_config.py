"""Centralized logging configuration (JobHunter pattern)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from quant_tools.config import cfg
from quant_tools.errors import ConfigError


def setup_logging(log_file: str | None = None, level: str = "INFO") -> logging.Logger:
    """Configure logging with file rotation and console output.

    Args:
        log_file: Path to log file. Defaults to the configured location, which
            is a per-user data directory rather than the working directory.
        level: Console log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("quant_tools")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    path = Path(log_file or cfg.log_file)

    # File handler with rotation (5MB max, 3 backups)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError as exc:
        # Versions of .env.example before this shipped LOG_FILE=quant_tools.log,
        # a relative path that resolves against wherever you launch from. Anyone
        # who copied that template still has it, and the failure is otherwise an
        # opaque traceback out of the logging module.
        raise ConfigError(
            f"Cannot open the log file at {path}: {exc}. "
            "LOG_FILE is set to a relative path, so it lands in whatever "
            "directory you launched from. Remove LOG_FILE from your .env to use "
            "the per-user data directory, or set it to an absolute path."
            if not path.is_absolute()
            else f"Cannot open the log file at {path}: {exc}. "
            "Set LOG_FILE to a writable location."
        ) from exc
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
