"""Predictions dashboard page -- current predictions and model agreement."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from quant_tools.dashboard.core.page_help import render_page_help
from quant_tools.dashboard.core.colors import (
    ACCENT_GREEN, ACCENT_BLUE, ACCENT_AMBER, ACCENT_RED, ACCENT_PURPLE,
    BG_CARD, TEXT_MUTED,
)
from quant_tools.dashboard.core.helpers import (
    directional_color,
    format_pct,
    PLOTLY_LAYOUT,
    render_metric_card,
)
from quant_tools.dashboard.core.symbol_picker import render_symbol_picker
from quant_tools.dashboard.core.symbol_state import get_active_symbol
from quant_tools.predict.horizons import (
    get_horizons, get_horizon_labels, horizon_display, parse_horizon_label,
)


def render() -> None:
    """Render the Predictions page."""
    st.markdown("## Predictions")

    render_page_help("predictions")

    # Try to load predictions from tracker
    try:
        from quant_tools.predict.tracker import PredictionTracker
        tracker = PredictionTracker()
        predictions = tracker.get_predictions(limit=100)
        performance = tracker.get_performance()
    except Exception as exc:
        import logging
        logging.getLogger("quant_tools").exception("Failed to load predictions")
        st.warning(f"Could not load prediction data: {exc}")
        tracker = None
        predictions = []
        performance = []

    # -- KPI Row --
    cols = st.columns(4)
    active_count = sum(1 for p in predictions if not p.get("resolved"))
    avg_conf = (
        sum(p.get("confidence", 0) or 0 for p in predictions) / len(predictions)
        if predictions else 0
    )
    # Find top model
    top_model = "N/A"
    if performance:
        best = max(performance, key=lambda p: p.win_rate)
        top_model = f"{best.model_name} ({format_pct(best.win_rate)})"

    with cols[0]:
        render_metric_card(
            "Active Predictions",
            str(active_count),
            ACCENT_BLUE,
        )
    with cols[1]:
        render_metric_card(
            "Avg Confidence",
            format_pct(avg_conf),
            directional_color(avg_conf - 0.5),
            glossary_key="Confidence",
            delta=avg_conf * 100,
            delta_suffix="%",
        )
    with cols[2]:
        render_metric_card(
            "Top Model",
            top_model,
            ACCENT_PURPLE,
            glossary_key="Ensemble",
        )
    with cols[3]:
        buy_count = sum(
            1 for p in predictions
            if p.get("consensus_signal") in ("strong_buy", "buy")
        )
        render_metric_card(
            "Buy Signals",
            str(buy_count),
            ACCENT_AMBER,
            glossary_key="Consensus Signal",
        )

    # -- Guidance Summary --
    if tracker is not None:
        _build_guidance_summary(tracker)

    st.divider()

    # -- Filters --
    filter_cols = st.columns(3)
    with filter_cols[0]:
        asset_filter = st.selectbox(
            "Asset Class",
            ["All", "equity", "crypto", "forex", "commodity"],
            key="pred_asset_filter",
            help="Filter predictions by asset type",
        )
    with filter_cols[1]:
        # Dynamic horizon labels based on selected asset class
        if asset_filter != "All":
            from quant_tools.models import AssetClass
            _ac = AssetClass(asset_filter)
            _horizon_options = ["All"] + get_horizon_labels(_ac)
        else:
            # Show union of all horizon labels when "All" is selected
            _horizon_options = ["All", "1h", "4h", "8h", "24h", "1d", "5d", "20d", "60d"]
        horizon_filter = st.selectbox(
            "Horizon",
            _horizon_options,
            key="pred_horizon_filter",
            help="Filter by prediction timeframe (how far ahead)",
        )
    with filter_cols[2]:
        model_filter = st.selectbox(
            "Model",
            ["All", "ensemble", "gradient_boost", "stats_forecast",
             "neural_forecast", "prophet", "garch"],
            key="pred_model_filter",
            help="Show predictions from a specific model or all models",
        )

    # Apply filters
    filtered = predictions
    if asset_filter != "All":
        filtered = [p for p in filtered if p.get("asset_class") == asset_filter]
    if horizon_filter != "All":
        filtered = [p for p in filtered if _matches_horizon(p, horizon_filter)]
    if model_filter != "All":
        filtered = [p for p in filtered if p.get("model_name") == model_filter]

    # -- Predictions Table --
    if filtered:
        df = pd.DataFrame(filtered)
        display_cols = [
            "symbol", "asset_class", "model_name", "prediction_direction",
            "confidence", "predicted_magnitude", "target_date", "consensus_signal",
            "interval",
        ]
        available = [c for c in display_cols if c in df.columns]
        display_df = df[available].copy()
        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].apply(
                lambda x: f"{x:.2%}" if x is not None else "N/A"
            )
        if "predicted_magnitude" in display_df.columns:
            display_df["predicted_magnitude"] = display_df["predicted_magnitude"].apply(
                lambda x: f"{x:.2%}" if x is not None else "N/A"
            )
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info(
            "No predictions match your filters. Generate a new prediction below, "
            "or try one of these popular symbols."
        )

        st.markdown("#### Quick Start Predictions")
        _global_sym = get_active_symbol()
        _pred_examples = [
            ("AAPL", "Apple -- large-cap tech bellwether"),
            ("MSFT", "Microsoft -- enterprise software leader"),
            ("NVDA", "NVIDIA -- AI/GPU momentum play"),
        ]
        if _global_sym and _global_sym not in [t for t, _ in _pred_examples]:
            _pred_examples.insert(0, (_global_sym, f"{_global_sym} -- your active symbol"))
        _pred_ex_cols = st.columns(min(len(_pred_examples), 4))
        for i, (ex_sym, ex_desc) in enumerate(_pred_examples[:4]):
            with _pred_ex_cols[i]:
                st.markdown(f"**{ex_sym}**")
                st.caption(ex_desc)
                if st.button(
                    "Predict",
                    key=f"pred_example_{ex_sym}",
                    use_container_width=True,
                    type="primary",
                ):
                    from quant_tools.dashboard.core.symbol_state import set_active_symbol
                    set_active_symbol(ex_sym)
                    st.session_state["pred_symbol_picker"] = ex_sym
                    st.rerun()

    # -- Charts Row --
    if filtered:
        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.markdown("### Model Agreement")
            fig = _build_model_agreement_chart(filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with chart_cols[1]:
            st.markdown("### Confidence Distribution")
            fig = _build_confidence_histogram(filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # -- Clear Predictions --
    st.divider()
    clear_cols = st.columns([3, 1])
    with clear_cols[1]:
        if st.button("Clear All Predictions", type="secondary", key="pred_clear"):
            try:
                from quant_tools.predict.tracker import PredictionTracker
                t = PredictionTracker()
                count = t.clear_predictions()
                st.success(f"Cleared {count} predictions.")
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")

    # -- New Prediction --
    st.divider()
    st.markdown("### Generate New Prediction")

    pred_cols = st.columns([3, 1, 1])
    with pred_cols[0]:
        _pred_default = get_active_symbol()
        symbol_input = render_symbol_picker(
            label="Symbol",
            key="pred_symbol_picker",
            default=_pred_default,
            sidebar=False,
        )

    # Detect asset class from symbol to populate horizon dropdown
    _detected_ac = None
    if symbol_input:
        try:
            from quant_tools.data.providers.registry import ProviderRegistry as _PR
            _detected_ac = _PR().detect_asset_class(symbol_input)
        except Exception:
            pass

    with pred_cols[1]:
        if _detected_ac is not None:
            _preset_labels = get_horizon_labels(_detected_ac)
        else:
            # Default to equity presets when no symbol entered
            from quant_tools.models import AssetClass as _AC
            _preset_labels = get_horizon_labels(_AC.EQUITY)
        _default_idx = _preset_labels.index("5d") if "5d" in _preset_labels else 0
        horizon_label = st.selectbox(
            "Horizon", _preset_labels, index=_default_idx,
            key="pred_horizon_select",
            help="How far ahead to predict (e.g., 5d = 5 trading days)",
        )
    with pred_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        run_prediction = st.button("Predict", type="primary", key="pred_run")

    if run_prediction and symbol_input:
        parsed = parse_horizon_label(horizon_label)
        if parsed is None:
            st.error(f"Invalid horizon label: {horizon_label}")
        else:
            spinner_label = parsed.label
            with st.spinner(f"Running ensemble prediction for {symbol_input} ({spinner_label})..."):
                try:
                    from quant_tools.predict.ensemble import PredictionEnsemble
                    from quant_tools.data.providers.registry import ProviderRegistry

                    registry = ProviderRegistry()
                    det_asset_class = _detected_ac or registry.detect_asset_class(symbol_input)

                    if parsed.unit == "h":
                        ohlcv = registry.fetch_ohlcv(
                            symbol_input, period="60d", interval=parsed.interval,
                        )
                        ensemble = PredictionEnsemble()
                        result = ensemble.predict(
                            ohlcv, symbol_input, det_asset_class,
                            hours=parsed.value, interval=parsed.interval,
                        )
                    else:
                        ohlcv = registry.fetch_ohlcv(symbol_input, period="2y")
                        ensemble = PredictionEnsemble()
                        result = ensemble.predict(
                            ohlcv, symbol_input, det_asset_class, parsed.value,
                        )

                    # Store prediction
                    from quant_tools.predict.tracker import PredictionTracker
                    t = PredictionTracker()
                    t.store_prediction(result)

                    st.success(
                        f"**{result.consensus.value.upper()}** -- "
                        f"Confidence: {result.confidence:.1%}, "
                        f"Direction: {result.direction.value}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


def _build_guidance_summary(tracker) -> None:
    """Render tiered prediction guidance summary (banner + expandable cards)."""
    try:
        latest = tracker.get_latest_by_symbol(limit_per_symbol=6)
        performance = tracker.get_performance()
    except Exception:
        return

    if not latest:
        st.markdown("### Guidance Summary")
        st.info("No predictions yet. Generate a prediction below to see guidance.")
        return

    # Build win-rate lookup: model_name -> best win_rate
    win_rates: dict[str, float] = {}
    for p in performance:
        existing = win_rates.get(p.model_name, 0.0)
        if p.win_rate > existing:
            win_rates[p.model_name] = p.win_rate

    # Signal display config
    _SIGNAL_LABELS = {
        "strong_buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
    }
    _DIRECTION_ARROWS = {"up": "\u25b2", "down": "\u25bc", "neutral": "\u25b6"}
    _DIRECTION_COLORS = {
        "up": ACCENT_GREEN,
        "down": ACCENT_RED,
        "neutral": ACCENT_AMBER,
    }

    # -- Tier 1: Banner Row --
    st.markdown("### Guidance Summary")
    symbols = list(latest.keys())
    banner_cols = st.columns(min(len(symbols), 4))
    for i, sym in enumerate(symbols):
        rows = latest[sym]
        ensemble = next((r for r in rows if r.get("is_ensemble")), None)
        if not ensemble:
            continue
        models = [r for r in rows if not r.get("is_ensemble")]

        direction = ensemble.get("prediction_direction", "neutral")
        arrow = _DIRECTION_ARROWS.get(direction, "\u25b6")
        color = _DIRECTION_COLORS.get(direction, ACCENT_AMBER)
        signal = _SIGNAL_LABELS.get(
            ensemble.get("consensus_signal", "hold"), "Hold"
        )
        conf = ensemble.get("confidence") or 0
        agree_count = sum(
            1 for m in models
            if m.get("prediction_direction") == direction
        )
        total_models = len(models) or 1

        col = banner_cols[i % len(banner_cols)]
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border-left:3px solid {color};'
                f'padding:10px 14px;border-radius:6px;margin-bottom:4px;">'
                f'<span style="color:{color};font-size:1.3rem;font-weight:700;">'
                f'{arrow} {sym}</span> &nbsp;'
                f'<span style="color:{color};font-weight:600;">{signal}</span><br>'
                f'<span style="color:{TEXT_MUTED};font-size:0.85rem;">'
                f'{conf:.0%} conf &middot; {agree_count}/{total_models} agree</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # -- Tier 2: Guidance Card (expander per symbol) --
        with col:
            with st.expander(f"{sym} Details"):
                # Verdict
                st.markdown(
                    f'<div style="text-align:center;padding:8px 0;">'
                    f'<span style="font-size:2rem;color:{color};">{arrow}</span> '
                    f'<span style="font-size:1.4rem;font-weight:700;color:{color};">'
                    f"{signal}</span></div>",
                    unsafe_allow_html=True,
                )

                # Confidence gauge
                bar_pct = int(conf * 100)
                st.markdown(
                    f'<div style="margin:6px 0;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.85rem;color:{TEXT_MUTED};">'
                    f"<span>Confidence</span><span>{conf:.1%}</span></div>"
                    f'<div style="background:rgba(255,255,255,0.08);'
                    f'border-radius:4px;height:8px;overflow:hidden;">'
                    f'<div style="width:{bar_pct}%;height:100%;'
                    f'background:{color};border-radius:4px;"></div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                # Predicted magnitude
                mag = ensemble.get("predicted_magnitude")
                if mag is not None:
                    st.markdown(
                        f"**Predicted Move:** {mag:+.2%}"
                    )

                # Model agreement detail
                up_count = sum(
                    1 for m in models if m.get("prediction_direction") == "up"
                )
                down_count = sum(
                    1 for m in models if m.get("prediction_direction") == "down"
                )
                neutral_count = sum(
                    1 for m in models if m.get("prediction_direction") == "neutral"
                )
                st.markdown(
                    f"**Model Agreement:** {agree_count} of {total_models} "
                    f"predict {direction.upper()} "
                    f"({up_count} UP / {down_count} DOWN / {neutral_count} NEUTRAL)"
                )

                # GARCH volatility context
                garch_row = next(
                    (m for m in models if m.get("model_name") == "garch"), None
                )
                if garch_row:
                    garch_conf = garch_row.get("confidence") or 0
                    if garch_conf >= 0.7:
                        vol_label = "Low"
                    elif garch_conf >= 0.4:
                        vol_label = "Moderate"
                    else:
                        vol_label = "High"
                    st.markdown(f"**Volatility:** {vol_label} (GARCH conf {garch_conf:.0%})")

                # Win rate context
                if win_rates:
                    best_model = max(win_rates, key=win_rates.get)  # type: ignore[arg-type]
                    best_wr = win_rates[best_model]
                    best_agrees = any(
                        m for m in models
                        if m.get("model_name") == best_model
                        and m.get("prediction_direction") == direction
                    )
                    agrees_str = "agrees" if best_agrees else "disagrees"
                    st.markdown(
                        f"**Top Model:** {best_model} ({best_wr:.0%} win rate) "
                        f"{agrees_str} with consensus"
                    )

                # -- Tier 3: Model-by-Model Breakdown --
                with st.expander("Model Breakdown"):
                    if models:
                        breakdown_data = []
                        for m in models:
                            m_dir = m.get("prediction_direction", "")
                            breakdown_data.append({
                                "Model": m.get("model_name", ""),
                                "Direction": m_dir.upper(),
                                "Confidence": f"{(m.get('confidence') or 0):.0%}",
                                "Magnitude": (
                                    f"{m['predicted_magnitude']:+.2%}"
                                    if m.get("predicted_magnitude") is not None
                                    else "N/A"
                                ),
                                "Consensus": (
                                    "\u2705" if m_dir == direction else "\u274c"
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(breakdown_data),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No individual model data available.")


def _matches_horizon(pred: dict, label: str) -> bool:
    """Check if a prediction dict matches a horizon label like '4h' or '5d'."""
    parsed = parse_horizon_label(label)
    if parsed is None:
        return False
    if parsed.unit == "h":
        return pred.get("horizon_hours") == parsed.value
    # unit == "d"
    return pred.get("horizon_days") == parsed.value and pred.get("horizon_hours") is None


def _build_model_agreement_chart(predictions: list[dict]) -> go.Figure | None:
    """Stacked bar chart showing model agreement per symbol."""
    ensemble_preds = [p for p in predictions if p.get("is_ensemble")]
    if not ensemble_preds:
        return None

    symbols = list({p["symbol"] for p in ensemble_preds})[:20]

    fig = go.Figure()
    for direction, color in [("up", ACCENT_GREEN), ("down", ACCENT_RED), ("neutral", ACCENT_AMBER)]:
        counts = []
        for sym in symbols:
            sym_preds = [
                p for p in predictions
                if p["symbol"] == sym and not p.get("is_ensemble")
            ]
            counts.append(
                sum(1 for p in sym_preds if p.get("prediction_direction") == direction)
            )
        fig.add_trace(go.Bar(name=direction.upper(), x=symbols, y=counts, marker_color=color))

    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="stack", title="Model Agreement by Symbol",
        xaxis_title="Symbol", yaxis_title="Model Count",
    )
    return fig


def _build_confidence_histogram(predictions: list[dict]) -> go.Figure | None:
    """Histogram of prediction confidences."""
    confidences = [p.get("confidence", 0) for p in predictions if p.get("confidence")]
    if not confidences:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=confidences, nbinsx=20, marker_color=ACCENT_BLUE))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="Confidence Distribution",
        xaxis_title="Confidence", yaxis_title="Count",
    )
    return fig
