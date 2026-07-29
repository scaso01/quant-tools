"""E2E test fixtures for Playwright-based Streamlit dashboard testing.

Playwright uses Chrome DevTools Protocol (CDP over local WebSocket),
so the parent conftest.py network guards (requests/httpx/urllib) do NOT
interfere with browser connections.
"""

from __future__ import annotations

import os
import socket

import pytest

# Playwright ships in the optional [e2e] extra. Importing it unconditionally
# aborted collection for the whole test suite on a plain [dev] install, so no
# tests ran at all. Skipping this directory instead leaves the rest runnable.
sync_playwright = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright is not installed; install the [e2e] extra to run these",
).sync_playwright

from .streamlit_helpers import StreamlitPage  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────

DEFAULT_URL = "http://localhost:8501"
HEALTH_PATH = "/_stcore/health"
PROBE_TIMEOUT = 5  # seconds


def _is_reachable(url: str) -> bool:
    """TCP probe to check if Streamlit server is up."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        sock = socket.create_connection((host, port), timeout=PROBE_TIMEOUT)
        sock.close()
        return True
    except (OSError, TimeoutError):
        return False


# ── Session-scoped fixtures ──────────────────────────────────────────


@pytest.fixture(scope="session")
def streamlit_url() -> str:
    """Base URL for the Streamlit server under test."""
    return os.environ.get("STREAMLIT_E2E_URL", DEFAULT_URL)


@pytest.fixture(scope="session", autouse=True)
def _check_streamlit_reachable(streamlit_url: str) -> None:
    """Skip all E2E tests if the Streamlit server is unreachable."""
    if not _is_reachable(streamlit_url):
        pytest.skip(
            f"Streamlit server unreachable at {streamlit_url}. "
            "Start with 'quant-tools web' or set STREAMLIT_E2E_URL."
        )


@pytest.fixture(scope="session")
def _playwright():
    """Playwright instance for the session."""
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def browser(_playwright):
    """Headless Chromium browser for the session."""
    b = _playwright.chromium.launch(
        headless=not os.environ.get("HEADED", ""),
        args=["--disable-gpu", "--no-sandbox"],
    )
    yield b
    b.close()


# ── Function-scoped fixtures ─────────────────────────────────────────


@pytest.fixture
def page(browser, streamlit_url: str):
    """Fresh browser page (tab) per test with Streamlit loaded."""
    ctx = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
    )
    pg = ctx.new_page()
    pg.goto(streamlit_url, wait_until="networkidle", timeout=30_000)
    # Wait for Streamlit app shell
    pg.wait_for_selector(
        '[data-testid="stAppViewContainer"]', timeout=15_000
    )
    # Wait for a sidebar nav button carrying its label (signals app.py executed
    # successfully). Waiting on the container alone is not enough: on a cold
    # server the nav containers appear before Streamlit paints their labels, and
    # a test reading the nav then sees a handful of blank buttons.
    pg.wait_for_selector(
        '[data-testid="stSidebar"] [class*="st-key-nav_"]:has-text("Market Scanner")',
        timeout=30_000,
    )
    yield pg
    pg.close()
    ctx.close()


@pytest.fixture
def st_page(page) -> StreamlitPage:
    """StreamlitPage wrapper with widget interaction helpers."""
    return StreamlitPage(page)
