"""Tests for the Textual TUI dashboard.

``quant-tools dashboard`` had no tests at all. It is a full-screen app, so the
only way anyone found out it was broken was by launching it, and a mistake in
compose() or a panel's update_data() would only show up as a terminal that
opens and immediately dies.

Textual's run_test() drives the app headless, which makes all of that ordinary
to check. Every test here runs the real app, not a stand-in.
"""
from __future__ import annotations

import asyncio

import pytest

from quant_tools.models import DashboardTab
from quant_tools.tui.app import QuantToolsApp

PANEL_IDS = (
    "#scanner-panel",
    "#backtest-panel",
    "#portfolio-panel",
    "#factors-panel",
    "#status-panel",
)

# Keybinding -> the panel it should leave visible.
TAB_KEYS = {
    "1": "#scanner-panel",
    "2": "#backtest-panel",
    "3": "#portfolio-panel",
    "4": "#factors-panel",
}


def _run(coro):
    """Drive an async test body without depending on an asyncio pytest plugin."""
    return asyncio.run(coro)


class TestMount:
    """The app has to survive being started."""

    def test_app_mounts_and_composes_every_panel(self):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                for panel_id in PANEL_IDS:
                    assert app.query(panel_id), f"{panel_id} was never mounted"
                await pilot.pause()

        _run(body())

    def test_mount_leaves_the_scanner_tab_showing(self):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.query_one("#scanner-panel").display is True
                assert app.query_one("#backtest-panel").display is False

        _run(body())


class TestTabSwitching:
    """Each number key shows exactly one content panel."""

    @pytest.mark.parametrize("key,visible_id", sorted(TAB_KEYS.items()))
    def test_key_shows_only_its_own_panel(self, key: str, visible_id: str):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                await pilot.press(key)
                await pilot.pause()
                for panel_id in TAB_KEYS.values():
                    expected = panel_id == visible_id
                    assert app.query_one(panel_id).display is expected, (
                        f"pressing {key} left {panel_id} display="
                        f"{app.query_one(panel_id).display}"
                    )

        _run(body())

    def test_switching_tabs_tracks_state(self):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                await pilot.press("3")
                await pilot.pause()
                assert app.active_tab is DashboardTab.PORTFOLIO
                assert app._state.active_tab is DashboardTab.PORTFOLIO

        _run(body())

    def test_cycling_every_tab_never_crashes(self):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                for key in ("1", "2", "3", "4", "1", "4", "2"):
                    await pilot.press(key)
                    await pilot.pause()
                assert app.query_one("#backtest-panel").display is True

        _run(body())


class TestEmptyState:
    """Panels render before any data has been fetched."""

    def test_panels_render_with_no_data(self):
        """update_data is called on mount with an empty DashboardState.

        A panel that assumed it always had results would take the whole app
        down on launch, before the user could press anything.
        """

        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app._state.scan_results == []
                assert app._state.backtest_result is None
                assert app._state.portfolio_analysis is None
                # A crash in compose or on_mount surfaces here.
                app._update_ui()
                await pilot.pause()

        _run(body())


class TestQuit:
    """The documented way out has to work."""

    def test_q_exits(self):
        async def body():
            app = QuantToolsApp()
            async with app.run_test() as pilot:
                await pilot.press("q")
                await pilot.pause()
            assert app.is_running is False

        _run(body())
