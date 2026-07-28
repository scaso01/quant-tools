"""Tests for retry.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from quant_tools.errors import ErrorCategory
from quant_tools.retry import (
    CircuitBreaker,
    categorize_error,
    retry_with_backoff,
)


class TestCategorizeError:
    """Test error categorization."""

    def test_network_timeout(self):
        assert categorize_error(TimeoutError("timed out")) == ErrorCategory.NETWORK

    def test_network_connection(self):
        assert categorize_error(ConnectionError("connection refused")) == ErrorCategory.NETWORK

    def test_rate_limit(self):
        assert categorize_error(Exception("429 Too Many Requests")) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_phrase(self):
        assert categorize_error(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMIT

    def test_data_missing(self):
        assert categorize_error(Exception("missing data for AAPL")) == ErrorCategory.DATA

    def test_data_empty(self):
        assert categorize_error(Exception("empty response")) == ErrorCategory.DATA

    def test_compute_division(self):
        assert categorize_error(ZeroDivisionError("division by zero")) == ErrorCategory.COMPUTE

    def test_unknown_fallback(self):
        assert categorize_error(Exception("something weird")) == ErrorCategory.DATA


class TestRetryWithBackoff:
    """Test retry decorator."""

    def test_success_no_retry(self):
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeeds() == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01, backoff_factor=1.0)
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network error")
            return "ok"

        assert fails_once() == "ok"
        assert call_count == 2

    def test_all_attempts_fail(self):
        @retry_with_backoff(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fails()

    def test_on_retry_callback(self):
        callback = MagicMock()
        call_count = 0

        @retry_with_backoff(
            max_attempts=3, initial_delay=0.01, backoff_factor=1.0,
            on_retry=callback,
        )
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        fails_twice()
        assert callback.call_count == 2

    def test_specific_exceptions(self):
        @retry_with_backoff(
            max_attempts=3, initial_delay=0.01,
            exceptions=(ValueError,),
        )
        def raises_type_error():
            raise TypeError("not caught")

        with pytest.raises(TypeError):
            raises_type_error()


class TestCircuitBreaker:
    """Test circuit breaker."""

    def test_breaker_closed_by_default(self):
        cb = CircuitBreaker("test", threshold=3, recovery_seconds=1.0)
        assert not cb.is_open

    def test_breaker_opens_after_threshold(self):
        cb = CircuitBreaker("test", threshold=3, recovery_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open

    def test_success_resets_count(self):
        cb = CircuitBreaker("test", threshold=3, recovery_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open  # Only 1 consecutive failure after reset

    def test_breaker_recovers(self):
        cb = CircuitBreaker("test", threshold=1, recovery_seconds=0.1)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.15)
        assert not cb.is_open  # Recovery time passed

    def test_manual_reset(self):
        cb = CircuitBreaker("test", threshold=1, recovery_seconds=60.0)
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert not cb.is_open
