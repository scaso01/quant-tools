"""Tests for the Performance Tracker dashboard view."""
from __future__ import annotations

import plotly.graph_objects as go

from quant_tools.models import AssetClass, ModelPerformance
from quant_tools.dashboard.views.performance import (
    _build_confidence_accuracy_scatter,
    _build_heatmap,
    _build_win_rate_chart,
)


# ── Sample Data ──────────────────────────────────────────────────────


def _sample_performance() -> list[ModelPerformance]:
    """Build sample ModelPerformance objects."""
    return [
        ModelPerformance(
            model_name="gradient_boost",
            asset_class=AssetClass.EQUITY,
            total_predictions=50,
            wins=32,
            losses=18,
            win_rate=0.64,
            avg_confidence=0.68,
            avg_error=0.015,
        ),
        ModelPerformance(
            model_name="stats_forecast",
            asset_class=AssetClass.EQUITY,
            total_predictions=50,
            wins=27,
            losses=23,
            win_rate=0.54,
            avg_confidence=0.61,
            avg_error=0.022,
        ),
        ModelPerformance(
            model_name="neural_forecast",
            asset_class=AssetClass.EQUITY,
            total_predictions=40,
            wins=22,
            losses=18,
            win_rate=0.55,
            avg_confidence=0.59,
            avg_error=0.019,
        ),
        ModelPerformance(
            model_name="gradient_boost",
            asset_class=AssetClass.CRYPTO,
            total_predictions=30,
            wins=13,
            losses=17,
            win_rate=0.433,
            avg_confidence=0.62,
            avg_error=0.045,
        ),
        ModelPerformance(
            model_name="prophet",
            asset_class=AssetClass.EQUITY,
            total_predictions=45,
            wins=25,
            losses=20,
            win_rate=0.556,
            avg_confidence=0.57,
            avg_error=0.020,
        ),
    ]


def _sample_resolved_predictions() -> list[dict]:
    """Build resolved prediction dicts for scatter plot."""
    return [
        {"confidence": 0.8, "prediction_direction": "up", "resolved": 1},
        {"confidence": 0.7, "prediction_direction": "up", "resolved": 1},
        {"confidence": 0.3, "prediction_direction": "down", "resolved": 1},
        {"confidence": 0.5, "prediction_direction": "neutral", "resolved": 1},
        {"confidence": 0.9, "prediction_direction": "up", "resolved": 1},
        {"confidence": 0.2, "prediction_direction": "down", "resolved": 1},
    ]


# ── Win Rate Chart ───────────────────────────────────────────────────


class TestWinRateChart:
    """Tests for _build_win_rate_chart."""

    def test_returns_figure(self) -> None:
        perf = _sample_performance()
        fig = _build_win_rate_chart(perf)
        assert isinstance(fig, go.Figure)

    def test_returns_none_on_empty(self) -> None:
        fig = _build_win_rate_chart([])
        assert fig is None

    def test_returns_none_on_none(self) -> None:
        fig = _build_win_rate_chart(None)
        assert fig is None

    def test_bar_count_matches_models(self) -> None:
        perf = _sample_performance()
        fig = _build_win_rate_chart(perf)
        assert len(fig.data) == 1  # Single bar trace
        assert len(fig.data[0].x) == len(perf)

    def test_layout_has_title(self) -> None:
        perf = _sample_performance()
        fig = _build_win_rate_chart(perf)
        assert "Win Rate" in fig.layout.title.text

    def test_colors_reflect_threshold(self) -> None:
        """Models above 0.5 should be green, below should be red."""
        perf = _sample_performance()
        fig = _build_win_rate_chart(perf)
        colors = fig.data[0].marker.color
        # gradient_boost (equity) = 0.64 -> green
        # gradient_boost (crypto) = 0.433 -> red
        from quant_tools.dashboard.core.colors import ACCENT_GREEN, ACCENT_RED
        for i, p in enumerate(perf):
            expected = ACCENT_GREEN if p.win_rate > 0.5 else ACCENT_RED
            assert colors[i] == expected


# ── Heatmap ──────────────────────────────────────────────────────────


class TestHeatmap:
    """Tests for _build_heatmap."""

    def test_returns_figure(self) -> None:
        perf = _sample_performance()
        fig = _build_heatmap(perf)
        assert isinstance(fig, go.Figure)

    def test_returns_none_on_empty(self) -> None:
        fig = _build_heatmap([])
        assert fig is None

    def test_returns_none_on_none(self) -> None:
        fig = _build_heatmap(None)
        assert fig is None

    def test_heatmap_colorscale(self) -> None:
        perf = _sample_performance()
        fig = _build_heatmap(perf)
        assert fig.data[0].colorscale is not None

    def test_heatmap_z_range(self) -> None:
        perf = _sample_performance()
        fig = _build_heatmap(perf)
        assert fig.data[0].zmin == 0
        assert fig.data[0].zmax == 1

    def test_text_annotations_present(self) -> None:
        perf = _sample_performance()
        fig = _build_heatmap(perf)
        assert fig.data[0].text is not None
        assert len(fig.data[0].text) > 0


# ── Confidence vs Accuracy Scatter ───────────────────────────────────


class TestConfidenceAccuracyScatter:
    """Tests for _build_confidence_accuracy_scatter."""

    def test_returns_figure(self) -> None:
        preds = _sample_resolved_predictions()
        fig = _build_confidence_accuracy_scatter(preds)
        assert isinstance(fig, go.Figure)

    def test_returns_none_on_empty(self) -> None:
        fig = _build_confidence_accuracy_scatter([])
        assert fig is None

    def test_returns_none_on_none(self) -> None:
        fig = _build_confidence_accuracy_scatter(None)
        assert fig is None

    def test_scatter_mode_is_markers(self) -> None:
        preds = _sample_resolved_predictions()
        fig = _build_confidence_accuracy_scatter(preds)
        assert fig.data[0].mode == "markers"

    def test_point_count_matches_input(self) -> None:
        preds = _sample_resolved_predictions()
        fig = _build_confidence_accuracy_scatter(preds)
        assert len(fig.data[0].x) == len(preds)
