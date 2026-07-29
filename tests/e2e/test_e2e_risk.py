"""E2E Playwright tests for the Risk Dashboard page.

Covers widget defaults, metric card rendering, sidebar interactions,
and boundary tests for the risk metrics page.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e]

# Page name including emoji (must match app.py PAGES dict)
RISK_PAGE = "\U0001f6e1\ufe0f Risk Dashboard"


@pytest.fixture(autouse=True)
def _navigate_to_risk(st_page) -> None:
    """Navigate to the Risk Dashboard page before every test."""
    st_page.navigate_to(RISK_PAGE)


# -- TestRiskWidgets -----------------------------------------------------------


class TestRiskWidgets:
    """Verify sidebar widgets and metric cards render correctly."""

    def test_page_loads(self, st_page) -> None:
        """Risk Dashboard page should render without errors."""
        visible = st_page.get_visible_text()
        assert "Risk Dashboard" in visible

    def test_var_confidence_slider_default(self, st_page) -> None:
        """VaR Confidence slider should default to 0.95."""
        value = st_page.get_slider_value("VaR Confidence", sidebar=True)
        assert value == pytest.approx(0.95, abs=0.005)

    def test_var_method_selectbox_options(self, st_page) -> None:
        """VaR Method selectbox should expose Historical and Parametric."""
        options = st_page.get_selectbox_options("VaR Method", sidebar=True)
        assert "Historical" in options
        assert "Parametric" in options

    def test_metric_cards_render(self, st_page) -> None:
        """The four risk metric cards should render once analysis is run."""
        st_page.click_button("Analyze Risk", sidebar=True, timeout=120_000)
        count = st_page.get_metric_count()
        assert count >= 4, f"Expected the 4 risk metric cards, got {count}"

    def test_risk_visuals_render(self, st_page) -> None:
        """The page should draw the drawdown chart and per-holding table."""
        st_page.click_button("Analyze Risk", sidebar=True, timeout=120_000)
        visible = st_page.get_visible_text()
        assert "In Development" not in visible, "Page is still a placeholder"
        assert st_page.has_plotly_chart(), "Expected the drawdown chart"
        assert st_page.has_dataframe(), "Expected the per-holding risk table"


# -- TestRiskNegative ----------------------------------------------------------


class TestRiskNegative:
    """Boundary tests for the Risk Dashboard sidebar controls."""

    def test_var_confidence_extreme_high(self, st_page) -> None:
        """Setting VaR confidence toward 0.99 should not crash."""
        original = st_page.get_slider_value("VaR Confidence", sidebar=True)
        st_page.set_slider("VaR Confidence", 0.99, sidebar=True)
        value = st_page.get_slider_value("VaR Confidence", sidebar=True)
        # Verify the slider moved (or was already at target) and no crash
        assert value >= original or value == pytest.approx(0.99, abs=0.02)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible

    def test_var_confidence_extreme_low(self, st_page) -> None:
        """Setting VaR confidence toward 0.90 should not crash."""
        original = st_page.get_slider_value("VaR Confidence", sidebar=True)
        st_page.set_slider("VaR Confidence", 0.90, sidebar=True)
        value = st_page.get_slider_value("VaR Confidence", sidebar=True)
        # Verify the slider moved (or was already at target) and no crash
        assert value <= original or value == pytest.approx(0.90, abs=0.02)
        visible = st_page.get_visible_text()
        assert "Traceback" not in visible
