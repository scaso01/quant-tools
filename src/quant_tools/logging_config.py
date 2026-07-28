"""Centralized logging configuration (JobHunter pattern)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = "quant_tools.log", level: str = "INFO") -> logging.Logger:
    """Configure logging with file rotation and console output.

    Args:
        log_file: Path to log file.
        level: Console log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("quant_tools")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # File handler with rotation (5MB max, 3 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
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
