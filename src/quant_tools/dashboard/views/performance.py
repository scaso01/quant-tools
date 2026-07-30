"""Performance tracking dashboard -- win rates, accuracy, model comparison."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from quant_tools.dashboard.core.page_help import render_page_help
from quant_tools.dashboard.core.colors import (
    ACCENT_GREEN, ACCENT_BLUE, ACCENT_AMBER, ACCENT_RED,
)
from quant_tools.dashboard.core.helpers import (
    directional_color,
    format_pct,
    PLOTLY_LAYOUT,
    render_metric_card,
)
from quant_tools.predict.horizons import horizon_display


def render() -> None:
    """Render the Performance Tracker page."""
    st.markdown("## Performance Tracker")

    render_page_help("performance")

    try:
        from quant_tools.predict.tracker import PredictionTracker
        tracker = PredictionTracker()
        performance = tracker.get_performance()
        all_predictions = tracker.get_predictions(limit=500)
    except Exception as exc:
        import logging
        logging.getLogger("quant_tools").exception("Failed to load performance data")
        st.warning(f"Could not load performance data: {exc}")
        performance = []
        all_predictions = []

    # -- KPI Row --
    cols = st.columns(4)
    total_preds = sum(p.total_predictions for p in performance) if performance else 0
    overall_wins = sum(p.wins for p in performance) if performance else 0
    overall_rate = overall_wins / total_preds if total_preds > 0 else 0

    best_model = "N/A"
    worst_model = "N/A"
    best_win_rate = 0.5
    worst_win_rate = 0.5
    if performance:
        valid = [p for p in performance if p.total_predictions >= 5]
        if valid:
            best = max(valid, key=lambda p: p.win_rate)
            worst = min(valid, key=lambda p: p.win_rate)
            best_model = f"{best.model_name} ({format_pct(best.win_rate)})"
            worst_model = f"{worst.model_name} ({format_pct(worst.win_rate)})"
            best_win_rate = best.win_rate
            worst_win_rate = worst.win_rate

    with cols[0]:
        render_metric_card("Total Predictions", str(total_preds), ACCENT_BLUE)
    with cols[1]:
        render_metric_card(
            "Overall Win Rate",
            format_pct(overall_rate),
            directional_color(overall_rate - 0.5),
            glossary_key="Win Rate",
        )
    with cols[2]:
        render_metric_card(
            "Best Model",
            best_model,
            directional_color(best_win_rate - 0.5),
        )
    with cols[3]:
        render_metric_card(
            "Worst Model",
            worst_model,
            directional_color(worst_win_rate - 0.5),
        )

    st.divider()

    # -- Model Comparison Table --
    if performance:
        st.markdown("### Model Comparison")
        perf_data = []
        for p in performance:
            perf_data.append({
                "Model": p.model_name,
                "Asset Class": p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class),
                "Total": p.total_predictions,
                "Wins": p.wins,
                "Losses": p.losses,
                "Win Rate": p.win_rate,
                "Avg Confidence": p.avg_confidence,
                "Avg Error": p.avg_error,
            })
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(
            perf_df.style.format({
                "Win Rate": "{:.1%}",
                "Avg Confidence": "{:.1%}",
                "Avg Error": "{:.2%}",
            }).background_gradient(subset=["Win Rate"], cmap="RdYlGn"),
            width="stretch",
        )

    # -- Charts --
    if performance:
        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.markdown("### Win Rate by Model")
            fig = _build_win_rate_chart(performance)
            if fig:
                st.plotly_chart(fig, width="stretch")

        with chart_cols[1]:
            st.markdown("### Win Rate Heatmap")
            fig = _build_heatmap(performance)
            if fig:
                st.plotly_chart(fig, width="stretch")

    # -- Confidence vs Accuracy Scatter --
    resolved = [p for p in all_predictions if p.get("resolved")]
    if resolved:
        st.markdown("### Confidence vs Accuracy")
        fig = _build_confidence_accuracy_scatter(resolved)
        if fig:
            st.plotly_chart(fig, width="stretch")

    # -- Recent Resolved Predictions --
    st.divider()
    st.markdown("### Recent Resolved Predictions")
    resolved_recent = [p for p in all_predictions if p.get("resolved")][:20]

    if resolved_recent:
        for pred in resolved_recent:
            direction = pred.get("prediction_direction", "?")
            confidence = pred.get("confidence", 0)
            symbol = pred.get("symbol", "?")
            h_display = horizon_display(
                horizon_days=pred.get("horizon_days"),
                horizon_hours=pred.get("horizon_hours"),
            )
            badge_color = ACCENT_GREEN if confidence and confidence > 0.6 else ACCENT_AMBER
            st.markdown(
                f'<span style="color:{badge_color}">&#9679;</span> '
                f'**{symbol}** ({h_display}) -- {direction.upper()} ({format_pct(confidence or 0)} conf) '
                f'-- Target: {pred.get("target_date", "?")}',
                unsafe_allow_html=True,
            )
    else:
        st.info("No resolved predictions yet. Predictions resolve when their target date passes.")

    # -- Resolve Button --
    st.divider()
    if st.button("Resolve Pending Predictions", type="primary", key="perf_resolve"):
        with st.spinner("Resolving..."):
            try:
                from quant_tools.predict.resolver import PredictionResolver
                from quant_tools.predict.tracker import PredictionTracker
                t = PredictionTracker()
                pending = t.get_pending()
                if not pending:
                    st.info("No pending predictions to resolve. Predictions resolve when their target date has passed.")
                else:
                    resolver = PredictionResolver(t)
                    outcomes = resolver.resolve_all_pending()
                    if outcomes:
                        wins = sum(1 for o in outcomes if o.correct)
                        st.success(
                            f"Resolved {len(outcomes)}/{len(pending)} predictions "
                            f"({wins} correct, {len(outcomes) - wins} incorrect)"
                        )
                        st.rerun()
                    else:
                        st.warning(
                            f"Found {len(pending)} pending predictions but could not resolve any. "
                            "This may happen if target dates haven't passed yet or price data is unavailable."
                        )
            except Exception as exc:
                st.error(f"Resolution failed: {exc}")


def _build_win_rate_chart(performance) -> go.Figure | None:
    """Bar chart of win rates by model."""
    if not performance:
        return None

    models = [p.model_name for p in performance]
    rates = [p.win_rate for p in performance]
    colors = [ACCENT_GREEN if r > 0.5 else ACCENT_RED for r in rates]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=rates, marker_color=colors))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="Win Rate by Model",
        xaxis_title="Model", yaxis_title="Win Rate",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def _build_heatmap(performance) -> go.Figure | None:
    """Heatmap of model x asset_class win rates."""
    if not performance:
        return None

    models = list({p.model_name for p in performance})
    asset_classes = list({
        p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class)
        for p in performance
    })

    z = []
    for model in models:
        row = []
        for ac in asset_classes:
            matching = [
                p for p in performance
                if p.model_name == model
                and (p.asset_class.value if hasattr(p.asset_class, "value") else str(p.asset_class)) == ac
            ]
            row.append(matching[0].win_rate if matching else 0)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=asset_classes, y=models,
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Win Rate: Model x Asset Class")
    return fig


def _build_confidence_accuracy_scatter(predictions: list[dict]) -> go.Figure | None:
    """Scatter plot: confidence vs actual correctness."""
    if not predictions:
        return None

    # Look up actual outcomes for resolved predictions
    outcome_map: dict[int, int] = {}
    pred_ids = [p["id"] for p in predictions if "id" in p]
    if pred_ids:
        try:
            from quant_tools.cache import PredictionOutcomeRow, _lock
            from quant_tools.predict.tracker import PredictionTracker
            tracker = PredictionTracker()
            with _lock:
                session = tracker._cache._session()
                try:
                    outcomes = session.query(PredictionOutcomeRow).filter(
                        PredictionOutcomeRow.prediction_id.in_(pred_ids)
                    ).all()
                    outcome_map = {o.prediction_id: o.correct for o in outcomes}
                finally:
                    session.close()
        except Exception:
            pass

    confidences = [p.get("confidence", 0) for p in predictions]
    correctness = [outcome_map.get(p.get("id"), 0) for p in predictions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=confidences, y=correctness,
        mode="markers", marker=dict(color=ACCENT_BLUE, opacity=0.5),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="Confidence vs Accuracy",
        xaxis_title="Confidence", yaxis_title="Correct (1/0)",
    )
    return fig
