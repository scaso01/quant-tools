"""E2E Playwright tests for the Live Monitor page.

Covers widget defaults, watchlist interaction, refresh slider,
and negative/edge-case inputs for the live feed page.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
LIVE_PAGE = "\u26a1 Live Monitor"

# Must match live.py DEFAULT_WATCHLIST joined with ", "
DEFAULT_SYMBOLS_TEXT = "AAPL, MSFT, GOOGL, AMZN, TSLA"


@pytest.fixture(autouse=True)
def _navigate_to_live(st_page) -> None:
    """Navigate to the Live Monitor page before every test."""
    st_page.navigate_to(LIVE_PAGE)


# -- TestLiveWidgets -----------------------------------------------------------


class TestLiveWidgets:
    """Verify sidebar widgets render with correct default values."""

    def test_page_loads(self, st_page) -> None:
        """Live Monitor page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Live Monitor" in visible

    def test_watchlist_default(self, st_page) -> None:
        """Watchlist text area should contain the default symbols."""
        value = st_page.get_text_area_value(
            "Symbols (comma-separated)", sidebar=True
        )
        assert "AAPL" in value
        assert "TSLA" in value

    def test_refresh_slider_default(self, st_page) -> None:
        """Refresh slider should default to 5 seconds."""
        value = st_page.get_slider_value("Refresh (seconds)", sidebar=True)
        assert value == 5

    def test_watchlist_modify(self, st_page) -> None:
        """Changing the watchlist to BTC, ETH should trigger a rerun."""
        st_page.set_text_area(
            "Symbols (comma-separated)", "BTC, ETH", sidebar=True
        )
        visible = st_page.get_visible_text()
        assert "BTC" in visible
        assert "Traceback" not in visible

    def test_watchlist_table_renders(self, st_page) -> None:
        """Refreshing the default watchlist should produce quote cards and a table."""
        st_page.click_button("Refresh Now", sidebar=True, timeout=120_000)
        visible = st_page.get_visible_text()
        assert "In Development" not in visible, "Page is still a placeholder"
        assert st_page.get_metric_count() >= 1, "Expected watchlist summary cards"
        assert st_page.has_dataframe(), "Expected the watchlist quote table"


# -- TestLiveNegative ----------------------------------------------------------


class TestLiveNegative:
    """Negative and edge-case tests for the Live Monitor page."""

    def test_empty_watchlist(self, st_page) -> None:
        """Clearing the watchlist should show a warning, not crash."""
        st_page.set_text_area(
            "Symbols (comma-separated)", "", sidebar=True
        )
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not show a raw traceback"
        # The page shows a warning when no symbols are present
        assert st_page.has_warning("Add symbols") or "watchlist" in visible.lower()

    def test_duplicate_symbols(self, st_page) -> None:
        """Entering duplicate symbols should not crash the page."""
        st_page.set_text_area(
            "Symbols (comma-separated)", "AAPL, AAPL, AAPL", sidebar=True
        )
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App should not crash on duplicates"
