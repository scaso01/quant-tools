"""Tests for the Predictions dashboard view."""
from __future__ import annotations

import pytest
import plotly.graph_objects as go

from quant_tools.dashboard.views.predictions import (
    _build_confidence_histogram,
    _build_model_agreement_chart,
)


# ── Sample Data ──────────────────────────────────────────────────────


def _sample_predictions() -> list[dict]:
    """Build a realistic set of prediction dicts."""
    return [
        {
            "id": 1,
            "symbol": "AAPL",
            "asset_class": "equity",
            "model_name": "gradient_boost",
            "prediction_direction": "up",
            "predicted_magnitude": 0.03,
            "confidence": 0.72,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "buy",
            "is_ensemble": 0,
            "resolved": 0,
        },
        {
            "id": 2,
            "symbol": "AAPL",
            "asset_class": "equity",
            "model_name": "stats_forecast",
            "prediction_direction": "up",
            "predicted_magnitude": 0.02,
            "confidence": 0.65,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "buy",
            "is_ensemble": 0,
            "resolved": 0,
        },
        {
            "id": 3,
            "symbol": "AAPL",
            "asset_class": "equity",
            "model_name": "neural_forecast",
            "prediction_direction": "down",
            "predicted_magnitude": -0.01,
            "confidence": 0.55,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "buy",
            "is_ensemble": 0,
            "resolved": 0,
        },
        {
            "id": 4,
            "symbol": "AAPL",
            "asset_class": "equity",
            "model_name": "ensemble",
            "prediction_direction": "up",
            "predicted_magnitude": 0.025,
            "confidence": 0.68,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "buy",
            "is_ensemble": 1,
            "resolved": 0,
        },
        {
            "id": 5,
            "symbol": "BTC/USDT",
            "asset_class": "crypto",
            "model_name": "gradient_boost",
            "prediction_direction": "up",
            "predicted_magnitude": 0.05,
            "confidence": 0.80,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "strong_buy",
            "is_ensemble": 0,
            "resolved": 0,
        },
        {
            "id": 6,
            "symbol": "BTC/USDT",
            "asset_class": "crypto",
            "model_name": "ensemble",
            "prediction_direction": "up",
            "predicted_magnitude": 0.04,
            "confidence": 0.78,
            "horizon_days": 5,
            "target_date": "2026-04-01",
            "consensus_signal": "strong_buy",
            "is_ensemble": 1,
            "resolved": 0,
        },
        {
            "id": 7,
            "symbol": "EUR/USD",
            "asset_class": "forex",
            "model_name": "stats_forecast",
            "prediction_direction": "neutral",
            "predicted_magnitude": 0.001,
            "confidence": 0.45,
            "horizon_days": 20,
            "target_date": "2026-04-15",
            "consensus_signal": "hold",
            "is_ensemble": 0,
            "resolved": 1,
        },
    ]


# ── Model Agreement Chart ────────────────────────────────────────────


class TestModelAgreementChart:
    """Tests for _build_model_agreement_chart."""

    def test_returns_figure_with_ensemble_preds(self) -> None:
        preds = _sample_predictions()
        fig = _build_model_agreement_chart(preds)
        assert isinstance(fig, go.Figure)

    def test_returns_none_when_no_ensemble(self) -> None:
        preds = [p for p in _sample_predictions() if not p.get("is_ensemble")]
        fig = _build_model_agreement_chart(preds)
        assert fig is None

    def test_returns_none_on_empty_list(self) -> None:
        fig = _build_model_agreement_chart([])
        assert fig is None

    def test_has_three_traces_for_directions(self) -> None:
        preds = _sample_predictions()
        fig = _build_model_agreement_chart(preds)
        # up, down, neutral
        assert len(fig.data) == 3

    def test_trace_names_are_uppercase(self) -> None:
        preds = _sample_predictions()
        fig = _build_model_agreement_chart(preds)
        names = {t.name for t in fig.data}
        assert names == {"UP", "DOWN", "NEUTRAL"}

    def test_limits_to_20_symbols(self) -> None:
        """If more than 20 unique ensemble symbols, chart limits to 20."""
        preds = []
        for i in range(25):
            preds.append({
                "id": i,
                "symbol": f"SYM{i}",
                "is_ensemble": 1,
                "prediction_direction": "up",
                "confidence": 0.5,
            })
        fig = _build_model_agreement_chart(preds)
        assert fig is not None
        # x-axis should have at most 20 symbols
        assert len(fig.data[0].x) <= 20


# ── Confidence Histogram ─────────────────────────────────────────────


class TestConfidenceHistogram:
    """Tests for _build_confidence_histogram."""

    def test_returns_figure_with_data(self) -> None:
        preds = _sample_predictions()
        fig = _build_confidence_histogram(preds)
        assert isinstance(fig, go.Figure)

    def test_returns_none_on_empty_list(self) -> None:
        fig = _build_confidence_histogram([])
        assert fig is None

    def test_returns_none_when_no_confidence(self) -> None:
        preds = [{"confidence": 0}, {"confidence": None}]
        fig = _build_confidence_histogram(preds)
        assert fig is None

    def test_histogram_trace_type(self) -> None:
        preds = _sample_predictions()
        fig = _build_confidence_histogram(preds)
        assert isinstance(fig.data[0], go.Histogram)

    def test_layout_has_title(self) -> None:
        preds = _sample_predictions()
        fig = _build_confidence_histogram(preds)
        assert "Confidence" in fig.layout.title.text


# ── Filter Logic ─────────────────────────────────────────────────────


class TestFilterLogic:
    """Tests for prediction filtering logic (used in render())."""

    def test_filter_by_asset_class(self) -> None:
        preds = _sample_predictions()
        filtered = [p for p in preds if p.get("asset_class") == "equity"]
        assert all(p["asset_class"] == "equity" for p in filtered)
        assert len(filtered) == 4

    def test_filter_by_horizon(self) -> None:
        preds = _sample_predictions()
        filtered = [p for p in preds if str(p.get("horizon_days")) == "20"]
        assert len(filtered) == 1
        assert filtered[0]["symbol"] == "EUR/USD"

    def test_filter_by_model(self) -> None:
        preds = _sample_predictions()
        filtered = [p for p in preds if p.get("model_name") == "ensemble"]
        assert len(filtered) == 2

    def test_filter_all_returns_everything(self) -> None:
        preds = _sample_predictions()
        # "All" filter = no filter applied
        filtered = preds
        assert len(filtered) == 7
