"""Structured exceptions for quant-tools (JobHunter pattern)."""

from __future__ import annotations

from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors for appropriate handling."""
    NETWORK = "network"        # yfinance/FRED connection failures
    RATE_LIMIT = "rate_limit"  # API throttling (Yahoo, FRED)
    DATA = "data"              # Missing/malformed ticker data
    CONFIG = "config"          # Bad .env / missing keys
    COMPUTE = "compute"        # NaN in calculations, division by zero


class QuantToolsError(Exception):
    """Base exception for all quant-tools errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.DATA) -> None:
        super().__init__(message)
        self.category = category


class ConfigError(QuantToolsError):
    """Raised on invalid configuration. Fail-fast on startup."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.CONFIG)


class DataError(QuantToolsError):
    """Raised when market data is missing or malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.DATA)


class NetworkError(QuantToolsError):
    """Raised on network/API connection failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.NETWORK)


class RateLimitError(QuantToolsError):
    """Raised when an API rate limit is hit."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.RATE_LIMIT)


class ComputeError(QuantToolsError):
    """Raised on computation failures (NaN, division by zero)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.COMPUTE)


class PredictionError(QuantToolsError):
    """Raised on prediction engine failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message, ErrorCategory.COMPUTE)
