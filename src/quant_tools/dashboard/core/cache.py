"""TTL cache with @cached decorator (JobHunter pattern)."""

from __future__ import annotations

import functools
import hashlib
import threading
import time
from typing import Any, Callable

_CACHE_MISS = object()


class CacheEntry:
    """Single cache entry with TTL expiration."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class TTLCache:
    """Thread-safe in-memory cache with per-entry TTL."""

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return _CACHE_MISS
            if entry.is_expired:
                del self._cache[key]
                return _CACHE_MISS
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        with self._lock:
            self._cache[key] = CacheEntry(value, ttl or self._default_ttl)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            # Purge expired entries
            now = time.time()
            self._cache = {
                k: v for k, v in self._cache.items() if v.expires_at > now
            }
            return len(self._cache)


# Global cache instance (5 min default for market data)
_global_cache = TTLCache(default_ttl=300.0)


def clear_cache() -> None:
    """Clear the global cache."""
    _global_cache.clear()


def get_cache() -> TTLCache:
    """Get the global cache instance."""
    return _global_cache


def cached(ttl: float = 300.0, key_prefix: str = "") -> Callable:
    """Decorator that caches function results with TTL.

    Args:
        ttl: Time-to-live in seconds (default 300 = 5 min).
        key_prefix: Prefix for cache keys to avoid collisions.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build cache key from prefix + function name + args hash
            # Skip 'self' argument for instance methods
            key_args = args
            if key_args and hasattr(key_args[0], "__dict__"):
                key_args = key_args[1:]

            raw = f"{key_prefix}:{func.__name__}:{key_args}:{sorted(kwargs.items())}"
            key = hashlib.md5(raw.encode()).hexdigest()

            result = _global_cache.get(key)
            if result is not _CACHE_MISS:
                return result

            result = func(*args, **kwargs)
            _global_cache.set(key, result, ttl)
            return result

        def refresh(*args: Any, **kwargs: Any) -> Any:
            """Bypass cache and refresh the value."""
            key_args = args
            if key_args and hasattr(key_args[0], "__dict__"):
                key_args = key_args[1:]

            raw = f"{key_prefix}:{func.__name__}:{key_args}:{sorted(kwargs.items())}"
            key = hashlib.md5(raw.encode()).hexdigest()

            result = func(*args, **kwargs)
            _global_cache.set(key, result, ttl)
            return result

        wrapper.refresh = refresh  # type: ignore[attr-defined]
        return wrapper

    return decorator
