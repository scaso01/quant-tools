"""Retry logic and circuit breakers (JobHunter pattern)."""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from quant_tools.errors import ErrorCategory

log = logging.getLogger("quant_tools")

# Pre-compiled patterns for error categorization
_RATE_LIMIT_PATTERNS = [
    re.compile(r"\b" + re.escape(ind) + r"\b")
    for ind in ["rate limit", "too many requests", "429", "throttl"]
]

_NETWORK_INDICATORS = [
    "timeout", "timed out", "connection", "connect",
    "refused", "reset", "broken pipe", "network",
    "unreachable", "dns", "socket", "ssl",
]


def categorize_error(exc: Exception) -> ErrorCategory:
    """Categorize an exception for retry handling.

    Args:
        exc: The exception to categorize.

    Returns:
        ErrorCategory indicating the type of failure.
    """
    exc_name = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    if any(ind in exc_name or ind in exc_msg for ind in _NETWORK_INDICATORS):
        return ErrorCategory.NETWORK

    if any(pat.search(exc_msg) for pat in _RATE_LIMIT_PATTERNS):
        return ErrorCategory.RATE_LIMIT

    data_indicators = ["missing", "empty", "no data", "nan", "invalid", "not found"]
    if any(ind in exc_msg for ind in data_indicators):
        return ErrorCategory.DATA

    compute_indicators = ["division", "zero", "overflow", "underflow", "singular"]
    if any(ind in exc_msg for ind in compute_indicators):
        return ErrorCategory.COMPUTE

    return ErrorCategory.DATA


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
    api_name: str | None = None,
) -> Callable:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        backoff_factor: Multiplier for delay between retries.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries.
        exceptions: Tuple of exception types to catch and retry.
        on_retry: Optional callback(attempt, exception, delay).
        api_name: API name for structured logging.

    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    category = categorize_error(exc)

                    if attempt == max_attempts:
                        log.warning(
                            "[%s] All %d attempts failed. Category: %s, Error: %s",
                            api_name or func.__name__, max_attempts, category.value, exc,
                        )
                        raise

                    actual_delay = min(delay, max_delay)
                    if category == ErrorCategory.RATE_LIMIT:
                        actual_delay = min(delay * 2, max_delay)

                    log.info(
                        "[%s] Attempt %d/%d failed (%s). Retrying in %.1fs...",
                        api_name or func.__name__, attempt, max_attempts,
                        category.value, actual_delay,
                    )

                    if on_retry:
                        on_retry(attempt, exc, actual_delay)

                    time.sleep(actual_delay)
                    delay *= backoff_factor

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


@dataclass
class CircuitBreaker:
    """Per-API circuit breaker.

    Opens after `threshold` failures, recovers after `recovery_seconds`.

    Args:
        name: Identifier for this breaker.
        threshold: Number of failures before opening.
        recovery_seconds: Seconds to wait before half-open test.
    """
    name: str
    threshold: int = 5
    recovery_seconds: float = 60.0
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure: datetime | None = field(default=None, init=False, repr=False)
    _is_open: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_success(self) -> None:
        """Record a successful call. Resets failure count."""
        with self._lock:
            self._failure_count = 0
            self._is_open = False

    def record_failure(self) -> None:
        """Record a failed call. May open the breaker."""
        with self._lock:
            self._failure_count += 1
            self._last_failure = datetime.now()
            if self._failure_count >= self.threshold:
                self._is_open = True
                log.warning(
                    "Circuit breaker [%s] OPEN after %d failures",
                    self.name, self._failure_count,
                )

    @property
    def is_open(self) -> bool:
        """Check if the breaker is open (blocking calls)."""
        with self._lock:
            if not self._is_open:
                return False
            if self._last_failure is None:
                return False
            elapsed = (datetime.now() - self._last_failure).total_seconds()
            if elapsed >= self.recovery_seconds:
                log.info(
                    "Circuit breaker [%s] half-open after %.0fs recovery",
                    self.name, elapsed,
                )
                self._is_open = False
                self._failure_count = 0
                return False
            return True

    def reset(self) -> None:
        """Manually reset the breaker."""
        with self._lock:
            self._failure_count = 0
            self._is_open = False
            self._last_failure = None


# Global circuit breakers
yfinance_breaker = CircuitBreaker("yfinance", threshold=5, recovery_seconds=60.0)
fred_breaker = CircuitBreaker("fred", threshold=3, recovery_seconds=90.0)
ccxt_breaker = CircuitBreaker("ccxt", threshold=5, recovery_seconds=60.0)
coingecko_breaker = CircuitBreaker("coingecko", threshold=3, recovery_seconds=90.0)
finnhub_breaker = CircuitBreaker("finnhub", threshold=3, recovery_seconds=90.0)
reddit_breaker = CircuitBreaker("reddit", threshold=3, recovery_seconds=120.0)
worldbank_breaker = CircuitBreaker("worldbank", threshold=3, recovery_seconds=90.0)
options_breaker = CircuitBreaker("options", threshold=3, recovery_seconds=60.0)
defillama_breaker = CircuitBreaker("defillama", threshold=3, recovery_seconds=90.0)
