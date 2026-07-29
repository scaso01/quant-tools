"""Regression tests for partial cache hits.

``CacheManager.get_prices`` documented "None if not fully cached" but returned
whatever cached rows happened to fall inside the requested window. Loading one
year of a symbol and then asking for five returned the one year and never
re-fetched, so the backtester scored a five year strategy on one year of bars
while the chart still drew a five year axis.
"""

from __future__ import annotations

from quant_tools.cache import COVERAGE_SLACK_DAYS, CacheManager, _covers_range


def _bars(dates: list[str]) -> list[dict[str, object]]:
    return [
        {
            "date": d, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1_000, "adj_close": 100.0,
        }
        for d in dates
    ]


class TestCoversRange:
    """The span check itself."""

    def test_exact_cover(self) -> None:
        assert _covers_range("2024-01-02", "2024-12-31", "2024-01-01", "2025-01-01")

    def test_cache_starts_too_late(self) -> None:
        assert not _covers_range(
            "2025-07-29", "2026-07-28", "2021-07-29", "2026-07-29"
        )

    def test_cache_ends_too_early(self) -> None:
        assert not _covers_range(
            "2021-08-02", "2026-06-01", "2021-07-29", "2026-07-29"
        )

    def test_weekend_and_holiday_slack_is_allowed(self) -> None:
        assert _covers_range("2024-01-08", "2024-12-27", "2024-01-05", "2024-12-31")

    def test_slack_boundary(self) -> None:
        # Exactly at the limit is still a hit, one day past it is not.
        assert _covers_range("2024-01-06", "2024-12-31", "2024-01-01", "2024-12-31")
        assert not _covers_range("2024-01-07", "2024-12-31", "2024-01-01", "2024-12-31")

    def test_unparseable_dates_are_not_a_hit(self) -> None:
        assert not _covers_range("nonsense", "2024-12-31", "2024-01-01", "2024-12-31")
        assert not _covers_range("2024-01-02", "2024-12-31", None, "2024-12-31")  # type: ignore[arg-type]

    def test_slack_constant_is_sane(self) -> None:
        # Long enough for a holiday weekend, short enough to catch a real gap.
        assert 2 <= COVERAGE_SLACK_DAYS <= 10


class TestGetPricesCoverage:
    """The behaviour through the cache manager."""

    def test_full_span_is_served_from_cache(self, tmp_db) -> None:
        cache = CacheManager(tmp_db)
        cache.store_prices("AAPL", _bars(["2024-01-02", "2024-06-14", "2024-12-30"]))
        assert cache.get_prices("AAPL", "2024-01-01", "2024-12-31") is not None

    def test_widening_the_window_is_a_miss(self, tmp_db) -> None:
        """The bug: one year cached, five years asked for."""
        cache = CacheManager(tmp_db)
        cache.store_prices("AAPL", _bars(["2025-08-01", "2026-01-05", "2026-07-27"]))
        assert cache.get_prices("AAPL", "2025-07-29", "2026-07-29") is not None
        assert cache.get_prices("AAPL", "2021-07-29", "2026-07-29") is None

    def test_narrowing_the_window_still_hits(self, tmp_db) -> None:
        cache = CacheManager(tmp_db)
        cache.store_prices(
            "AAPL", _bars(["2024-01-02", "2024-06-14", "2024-12-30"])
        )
        rows = cache.get_prices("AAPL", "2024-06-01", "2024-06-30")
        assert rows is not None
        assert len(rows) == 1

    def test_stale_forward_edge_is_a_miss(self, tmp_db) -> None:
        """Yesterday's cache must not satisfy a request that runs to today."""
        cache = CacheManager(tmp_db)
        cache.store_prices("AAPL", _bars(["2024-01-02", "2024-03-01"]))
        assert cache.get_prices("AAPL", "2024-01-01", "2024-12-31") is None

    def test_empty_cache_is_still_a_miss(self, tmp_db) -> None:
        cache = CacheManager(tmp_db)
        assert cache.get_prices("NOPE", "2024-01-01", "2024-12-31") is None
