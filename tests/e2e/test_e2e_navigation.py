"""E2E tests for cross-page navigation in the Streamlit dashboard.

Verifies sidebar radio navigation, query-param routing, and
basic chrome (brand, cache button) across all 10 dashboard pages.
"""

from __future__ import annotations

import re

import pytest

from .streamlit_helpers import StreamlitPage

pytestmark = [pytest.mark.e2e]

# ── Page registry (must match app.py PAGES dict order) ────────────────

ALL_PAGE_NAMES = [
    "\U0001f4e1 Market Scanner",
    "\U0001f4c8 Strategy Backtester",
    "\U0001f4bc Portfolio Analyzer",
    "\U0001f3db\ufe0f Factors & Macro",
    "\U0001f52e Predictions",
    "\U0001f4ca Performance",
    "\U0001f4c9 Rolling Analytics",
    "\U0001f6e1\ufe0f Risk Dashboard",
    "\u26a1 Live Monitor",
    "\U0001f916 AI Analysis",
]


def _strip_emoji(name: str) -> str:
    """Remove leading emoji and whitespace, returning the plain text label."""
    return re.sub(r"^[^\w]+", "", name).strip()


# ── TestNavigation ────────────────────────────────────────────────────


class TestNavigation:
    """Core sidebar-radio navigation tests."""

    def test_all_pages_listed(self, st_page: StreamlitPage) -> None:
        """Sidebar radio should expose exactly 10 page options."""
        names = st_page.get_page_names()
        # Filter out the radio group label ("Navigation") which is
        # collapsed but still present in the DOM as a label element.
        page_names = [n for n in names if n not in ("Navigation", "")]
        assert len(page_names) == 10, (
            f"Expected 10 pages, got {len(page_names)}: {page_names}"
        )

    def test_default_page_is_scanner(self, st_page: StreamlitPage) -> None:
        """First load (no query params) should land on Market Scanner."""
        visible = st_page.get_visible_text()
        assert "Market Scanner" in visible, (
            "Default page should be Market Scanner"
        )

    @pytest.mark.parametrize("page_name", ALL_PAGE_NAMES, ids=[
        _strip_emoji(n).replace(" ", "-").lower() for n in ALL_PAGE_NAMES
    ])
    def test_navigate_to_each_page(
        self, st_page: StreamlitPage, page_name: str
    ) -> None:
        """Navigate to *page_name* via sidebar radio; verify the header."""
        st_page.navigate_to(page_name)

        plain_label = _strip_emoji(page_name)
        visible = st_page.get_visible_text()
        assert plain_label in visible, (
            f"After navigating to '{page_name}', expected '{plain_label}' "
            f"in visible text but did not find it."
        )

    def test_clear_cache_button(self, st_page: StreamlitPage) -> None:
        """Clicking 'Clear Cache' in the sidebar should not cause an error."""
        st_page.click_button("Clear Cache", sidebar=True)
        # After rerun the app should still be functional (no uncaught exception)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible, "App crashed after clearing cache"

    def test_sidebar_brand_visible(self, st_page: StreamlitPage) -> None:
        """The 'Quant-Tools' brand text should appear in the sidebar."""
        sidebar_text = st_page.get_sidebar_text()
        assert "Quant-Tools" in sidebar_text, (
            "Sidebar should contain the 'Quant-Tools' brand"
        )


# ── TestQueryParamRouting ─────────────────────────────────────────────


class TestQueryParamRouting:
    """Tests that verify direct-URL (query-param) page routing."""

    def test_direct_url_navigation(
        self, st_page: StreamlitPage, streamlit_url: str
    ) -> None:
        """Loading ?page=... should route directly to the target page."""
        target = "\U0001f4c8 Strategy Backtester"
        direct_url = f"{streamlit_url}/?page={target}"
        st_page.page.goto(direct_url, wait_until="networkidle", timeout=30_000)
        st_page.wait_for_rerun()

        visible = st_page.get_visible_text()
        assert "Strategy Backtester" in visible, (
            "Direct URL with ?page= should load the Strategy Backtester page"
        )
