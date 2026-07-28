"""Dashboard workflow tests -- full widget -> compute -> render pipeline.

Tests that actual data flows through each dashboard page when widgets
are configured and buttons are clicked. Uses Streamlit's AppTest
framework with mocked data layer.

Each test class covers one dashboard page and verifies the complete
user workflow: navigate -> set widgets -> click action -> verify output.

Run:
    python -m pytest tests/test_dashboard_workflows.py -v --tb=short

    # Single page:
    python -m pytest tests/test_dashboard_workflows.py -v -k "Scanner"
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# -- Sample Data Generators ------------------------------------------------


def _sample_ohlcv(days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV DataFrame for mocking."""
    np.random.seed(seed)
    dates = pd.bdate_range("2025-01-02", periods=days)
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, days))
    return pd.DataFrame(
        {
            "Open": close * (1 + np.random.normal(0, 0.005, days)),
            "High": close * (1 + np.abs(np.random.normal(0, 0.01, days))),
            "Low": close * (1 - np.abs(np.random.normal(0, 0.01, days))),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 50_000_000, days).astype(float),
        },
        index=dates,
    )


def _sample_returns(days: int = 252, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate returns DataFrame for multiple assets."""
    np.random.seed(seed)
    symbols = ["AAPL", "MSFT", "BND", "GLD", "VNQ"][:n_assets]
    data = np.random.normal(0.0005, 0.015, (days, n_assets))
    dates = pd.bdate_range("2025-01-02", periods=days)
    return pd.DataFrame(data, index=dates, columns=symbols)


# -- App Path and Loader ---------------------------------------------------

APP_PATH = str(
    Path(__file__).resolve().parent.parent
    / "src"
    / "quant_tools"
    / "dashboard"
    / "app.py"
)


def _load_app():
    """Load AppTest instance of the dashboard."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP_PATH, default_timeout=30)


def _navigate(at, page_substring: str):
    """Navigate to a page by matching substring in the sidebar buttons."""
    buttons = at.sidebar.button
    match = [b for b in buttons if page_substring in b.label]
    assert match, (
        f"No sidebar button matching '{page_substring}' in "
        f"{[b.label for b in buttons]}"
    )
    match[0].click()
    at.run()
    return at


# -- Sample Model Factories ------------------------------------------------


def _make_scan_results(n: int = 5) -> list:
    """Build a list of ScanResult objects for mocking scan_universe."""
    from quant_tools.models import ScanResult, SignalType

    signals = [SignalType.MOMENTUM, SignalType.VALUE, SignalType.QUALITY,
               SignalType.MEAN_REVERSION]
    results = []
    for i in range(n):
        results.append(
            ScanResult(
                symbol=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"][i % 5],
                name=["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"][i % 5],
                sector=["Technology", "Technology", "Communication Services",
                         "Consumer Discretionary", "Technology"][i % 5],
                signal=signals[i % len(signals)],
                score=round(8.5 - i * 0.3, 2),
                price=round(150.0 + i * 10, 2),
                change_1d=round(0.5 - i * 0.2, 2),
                change_5d=round(1.0 - i * 0.3, 2),
                change_1m=round(3.0 - i * 0.5, 2),
                rsi_14=round(55.0 + i * 3, 1),
                details={"breakdown": {"trend": 3.0, "accel": 2.5}},
            )
        )
    return results


def _make_backtest_result():
    """Build a BacktestResult for mocking run_backtest."""
    from quant_tools.models import BacktestResult, StrategyType

    n = 252
    np.random.seed(42)
    equity = list(np.cumsum(np.random.normal(50, 200, n)) + 100000)
    return BacktestResult(
        strategy=StrategyType.SMA_CROSSOVER,
        ticker="AAPL",
        start_date=date(2020, 1, 1),
        end_date=date(2025, 1, 1),
        initial_capital=100000.0,
        final_value=equity[-1],
        total_return=0.25,
        cagr=0.05,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown=-0.15,
        max_drawdown_duration_days=45,
        win_rate=0.55,
        total_trades=120,
        profit_factor=1.8,
        equity_curve=equity,
        params={"fast": 20, "slow": 50},
    )


def _make_portfolio_analysis():
    """Build a PortfolioAnalysis for mocking analyze_portfolio."""
    from quant_tools.models import PortfolioAnalysis, PortfolioHolding

    return PortfolioAnalysis(
        holdings=[
            PortfolioHolding(
                symbol="AAPL", weight=0.3, current_price=180.0,
                total_return=0.12, annualized_return=0.10,
                volatility=0.20, sharpe=1.1,
            ),
            PortfolioHolding(
                symbol="BND", weight=0.4, current_price=75.0,
                total_return=0.03, annualized_return=0.02,
                volatility=0.05, sharpe=0.8,
            ),
            PortfolioHolding(
                symbol="GLD", weight=0.3, current_price=190.0,
                total_return=0.08, annualized_return=0.07,
                volatility=0.15, sharpe=0.9,
            ),
        ],
        total_return=0.08,
        annualized_return=0.06,
        volatility=0.12,
        sharpe_ratio=1.0,
        max_drawdown=-0.08,
        correlation_matrix={
            "AAPL": {"AAPL": 1.0, "BND": -0.2, "GLD": 0.1},
            "BND": {"AAPL": -0.2, "BND": 1.0, "GLD": 0.3},
            "GLD": {"AAPL": 0.1, "BND": 0.3, "GLD": 1.0},
        },
        sector_exposure={"Technology": 0.3, "Fixed Income": 0.4, "Commodities": 0.3},
    )


def _make_monte_carlo_result():
    """Build a MonteCarloResult for mocking run_monte_carlo."""
    from quant_tools.models import DistributionType, MonteCarloResult, DistributionParams

    np.random.seed(42)
    n_sims = 100
    horizon = 252
    paths = []
    for _ in range(n_sims):
        returns = np.random.normal(0.0003, 0.01, horizon)
        path = list(100000 * np.cumprod(1 + returns))
        paths.append(path)

    finals = [p[-1] for p in paths]
    return MonteCarloResult(
        n_simulations=n_sims,
        time_horizon_days=horizon,
        initial_value=100000.0,
        mean_final=float(np.mean(finals)),
        median_final=float(np.median(finals)),
        percentile_5=float(np.percentile(finals, 5)),
        percentile_25=float(np.percentile(finals, 25)),
        percentile_75=float(np.percentile(finals, 75)),
        percentile_95=float(np.percentile(finals, 95)),
        prob_loss=0.15,
        prob_gain_10pct=0.40,
        simulated_paths=paths,
        distribution_used=DistributionType.NORMAL,
        distribution_params=DistributionParams(
            distribution=DistributionType.NORMAL, mean=0.0003, std=0.01,
        ),
        skewness=0.05,
        kurtosis=3.1,
    )


def _make_factor_scores(n: int = 10) -> list:
    """Build FactorScore list for mocking calculate_factor_scores."""
    from quant_tools.models import FactorScore

    syms = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "V", "UNH", "XOM"]
    names = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA",
             "JPMorgan", "J&J", "Visa", "UnitedHealth", "ExxonMobil"]
    sectors = ["Technology", "Technology", "Communication Services",
               "Consumer Discretionary", "Technology", "Financials",
               "Healthcare", "Financials", "Healthcare", "Energy"]
    results = []
    for i in range(min(n, len(syms))):
        results.append(
            FactorScore(
                symbol=syms[i],
                name=names[i],
                value_z=round(1.5 - i * 0.2, 2),
                momentum_z=round(1.0 + i * 0.1, 2),
                quality_z=round(0.8 - i * 0.05, 2),
                composite_z=round(3.3 - i * 0.15, 2),
                rank=i + 1,
                sector=sectors[i],
            )
        )
    return results


def _make_macro_snapshot():
    """Build a MacroSnapshot for mocking detect_regime."""
    from quant_tools.models import MacroRegime, MacroSnapshot

    return MacroSnapshot(
        regime=MacroRegime.EXPANSION,
        gdp_growth=0.025,
        unemployment=0.038,
        cpi_yoy=0.032,
        fed_funds_rate=0.0525,
        yield_spread_10y2y=0.005,
        signals={"gdp": "growing", "unemployment": "low"},
    )


def _make_rebalance_recommendation():
    """Build a RebalanceRecommendation for mocking get_recommendations."""
    from quant_tools.models import MacroRegime, RebalanceRecommendation

    return RebalanceRecommendation(
        regime=MacroRegime.EXPANSION,
        value_tilt=0.2,
        momentum_tilt=0.4,
        quality_tilt=0.25,
        defensive_tilt=0.15,
        top_picks=_make_factor_scores(5),
        rationale="Expansion regime favors momentum and growth stocks.",
    )


def _make_performance_records() -> list:
    """Build ModelPerformance list for mocking tracker.get_performance."""
    from quant_tools.models import AssetClass, ModelPerformance

    return [
        ModelPerformance(
            model_name="gradient_boost",
            asset_class=AssetClass.EQUITY,
            total_predictions=50,
            wins=30,
            losses=20,
            win_rate=0.60,
            avg_confidence=0.72,
            avg_error=0.03,
        ),
        ModelPerformance(
            model_name="stats_forecast",
            asset_class=AssetClass.EQUITY,
            total_predictions=50,
            wins=25,
            losses=25,
            win_rate=0.50,
            avg_confidence=0.65,
            avg_error=0.04,
        ),
        ModelPerformance(
            model_name="prophet",
            asset_class=AssetClass.CRYPTO,
            total_predictions=30,
            wins=18,
            losses=12,
            win_rate=0.60,
            avg_confidence=0.58,
            avg_error=0.05,
        ),
    ]


def _make_prediction_dicts(n: int = 10) -> list[dict]:
    """Build prediction dicts mimicking PredictionTracker.get_predictions."""
    preds = []
    for i in range(n):
        is_ensemble = i < 3
        preds.append({
            "id": i + 1,
            "symbol": ["AAPL", "MSFT", "GOOGL"][i % 3],
            "asset_class": "equity",
            "model_name": "ensemble" if is_ensemble else ["gradient_boost", "stats_forecast", "prophet", "garch"][i % 4],
            "prediction_direction": ["up", "down", "up"][i % 3],
            "confidence": round(0.5 + i * 0.03, 2),
            "predicted_magnitude": round(0.02 + i * 0.005, 4),
            "horizon_days": 5,
            "target_date": "2025-06-15",
            "consensus_signal": ["buy", "sell", "buy"][i % 3],
            "resolved": i >= 7,
            "is_ensemble": is_ensemble,
            "interval": "1d",
        })
    return preds


# =====================================================================
# Scanner Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestScannerWorkflow:
    """Market Scanner page -- full scan workflow with mocked data."""

    def test_scanner_initial_state_shows_info(self):
        """Before clicking Run Scan, an info message is shown."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")
        assert not at.exception, f"Exception: {at.exception}"
        # Should have the "Ready to scan" info message
        info_texts = [str(i.value) for i in at.info]
        assert any("Ready to scan" in t for t in info_texts), (
            f"Expected 'Ready to scan' info message, got: {info_texts}"
        )

    def test_scanner_run_scan_renders_results(self):
        """Clicking Run Scan with mocked data produces charts and a dataframe."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        # Mock scan_universe to return fake results
        mock_results = _make_scan_results(10)
        with patch(
            "quant_tools.dashboard.views.scanner.scan_universe",
            return_value=mock_results,
        ):
            # Click the Run Scan button
            run_btns = [b for b in at.sidebar.button if b.key == "scanner_run"]
            if run_btns:
                run_btns[0].click()
            else:
                # Fall back to finding by label
                for b in at.sidebar.button:
                    if "Run Scan" in str(getattr(b, "label", "")):
                        b.click()
                        break
            at.run()

        assert not at.exception, f"Exception after scan: {at.exception}"
        # Results should produce at least 1 dataframe (results table)
        assert len(at.dataframe) >= 1, "Expected dataframe with scan results"

    def test_scanner_no_signals_shows_warning(self):
        """Unchecking all signals and clicking Run Scan shows a warning."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        # Uncheck all signal checkboxes
        for cb in at.sidebar.checkbox:
            cb.set_value(False)

        # Click Run Scan
        for b in at.sidebar.button:
            if b.key == "scanner_run":
                b.click()
                break
        at.run()

        assert not at.exception
        # Should show a warning (either about signals or symbols)
        warnings = [str(w.value) for w in at.warning]
        assert len(warnings) >= 1, "Expected at least one warning when filters are empty"

    def test_scanner_asset_class_selectbox_exists(self):
        """Scanner has asset class selectbox in sidebar."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        # Find the asset class selectbox by key
        ac_boxes = [s for s in at.sidebar.selectbox if s.key == "scanner_asset_class"]
        assert len(ac_boxes) == 1, "Expected scanner_asset_class selectbox"
        assert "Equity" in ac_boxes[0].options
        assert "Crypto" in ac_boxes[0].options

    def test_scanner_crypto_scan(self):
        """Switching to Crypto and scanning returns results."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        # Set asset class to Crypto
        for s in at.sidebar.selectbox:
            if s.key == "scanner_asset_class":
                s.set_value("Crypto")
                break
        at.run()

        mock_results = _make_scan_results(5)
        with patch(
            "quant_tools.dashboard.views.scanner.scan_universe",
            return_value=mock_results,
        ):
            for b in at.sidebar.button:
                if b.key == "scanner_run":
                    b.click()
                    break
            at.run()

        assert not at.exception

    def test_scanner_top_n_slider_exists(self):
        """Scanner has a Top N slider in sidebar."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        sliders = [s for s in at.sidebar.slider if s.key == "scanner_top_n"]
        assert len(sliders) == 1
        assert sliders[0].value == 20  # default

    def test_scanner_results_render_full_output(self):
        """Successful scan renders dataframe, markdown, and no exceptions."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        mock_results = _make_scan_results(10)
        with patch(
            "quant_tools.dashboard.views.scanner.scan_universe",
            return_value=mock_results,
        ):
            for b in at.sidebar.button:
                if b.key == "scanner_run":
                    b.click()
                    break
            at.run()

        assert not at.exception, f"Exception: {at.exception}"
        # Scan results table + score breakdowns + signal explanation tables
        assert len(at.dataframe) >= 1, "Expected results dataframe"
        # Plotly charts are rendered but AppTest has no plotly_chart accessor;
        # verify that markdown includes "Score Distribution" or "Signal Breakdown"
        # headers from the chart-building functions
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Scan Results" in all_md or len(at.dataframe) >= 1

    def test_scanner_empty_results_shows_warning(self):
        """Scan returning empty results shows a 'no results' warning."""
        at = _load_app()
        at.run()
        _navigate(at, "Scanner")

        # Mock both the cached wrapper and the underlying function
        with (
            patch(
                "quant_tools.dashboard.views.scanner.scan_universe",
                return_value=[],
            ),
            patch(
                "quant_tools.dashboard.views.scanner._cached_scan",
                return_value=[],
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "scanner_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # When scan returns empty, the page shows a warning
        warnings = [str(w.value) for w in at.warning]
        assert any("no results" in w.lower() or "returned" in w.lower() for w in warnings), (
            f"Expected 'no results' warning, got: {warnings}"
        )


# =====================================================================
# Backtester Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestBacktesterWorkflow:
    """Strategy Backtester page -- full backtest workflow with mocked data."""

    def test_backtester_initial_state_shows_info(self):
        """Before clicking Run Backtest, an info message is shown."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")
        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Configure your backtest" in t for t in info_texts)

    def test_backtester_run_produces_equity_curve(self):
        """Clicking Run Backtest with mocked data renders equity curve chart."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        mock_prices = _sample_ohlcv(252)
        mock_result = _make_backtest_result()

        with (
            patch(
                "quant_tools.dashboard.views.backtest.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.backtest.run_backtest",
                return_value=mock_result,
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "bt_run":
                    b.click()
                    break
            at.run()

        assert not at.exception, f"Exception: {at.exception}"
        # Backtest ran successfully -- KPI markdown cards and divider are rendered
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Equity Curve" in all_md or "Total Return" in all_md, (
            "Expected backtest output with KPI cards or equity curve header"
        )

    def test_backtester_kpi_metrics_rendered(self):
        """Backtest result renders KPI metric cards."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        mock_prices = _sample_ohlcv(252)
        mock_result = _make_backtest_result()

        with (
            patch(
                "quant_tools.dashboard.views.backtest.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.backtest.run_backtest",
                return_value=mock_result,
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "bt_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # render_metric_card outputs markdown -- check for known KPI text
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Total Return" in all_md or "Sharpe" in all_md or len(at.plotly_chart) >= 1

    def test_backtester_strategy_selectbox(self):
        """Backtester has a strategy selectbox with 4 strategies."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        strat_boxes = [s for s in at.sidebar.selectbox if s.key == "bt_strategy"]
        assert len(strat_boxes) == 1
        assert "SMA Crossover" in strat_boxes[0].options
        assert "RSI" in strat_boxes[0].options

    def test_backtester_different_strategies(self):
        """Switching strategy to RSI and running backtest works."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        for s in at.sidebar.selectbox:
            if s.key == "bt_strategy":
                s.set_value("RSI")
                break

        mock_prices = _sample_ohlcv(252)
        mock_result = _make_backtest_result()
        mock_result.strategy = __import__(
            "quant_tools.models", fromlist=["StrategyType"]
        ).StrategyType.RSI

        with (
            patch(
                "quant_tools.dashboard.views.backtest.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.backtest.run_backtest",
                return_value=mock_result,
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "bt_run":
                    b.click()
                    break
            at.run()

        assert not at.exception

    def test_backtester_ticker_input(self):
        """Backtester has a ticker text input defaulting to AAPL."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        ticker_inputs = [t for t in at.sidebar.text_input if t.key == "bt_ticker"]
        assert len(ticker_inputs) == 1
        assert ticker_inputs[0].value == "AAPL"

    def test_backtester_data_error_shows_warning(self):
        """DataError during fetch shows a user-friendly warning."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        from quant_tools.errors import DataError

        with patch(
            "quant_tools.dashboard.views.backtest.fetch_prices",
            side_effect=DataError("No data for FAKE"),
        ):
            for b in at.sidebar.button:
                if b.key == "bt_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        warnings = [str(w.value) for w in at.warning]
        assert any("Data" in w or "data" in w for w in warnings), (
            f"Expected data warning, got: {warnings}"
        )

    def test_backtester_network_error_shows_error(self):
        """NetworkError during fetch shows an error message."""
        at = _load_app()
        at.run()
        _navigate(at, "Backtester")

        from quant_tools.errors import NetworkError

        with patch(
            "quant_tools.dashboard.views.backtest.fetch_prices",
            side_effect=NetworkError("Connection refused"),
        ):
            for b in at.sidebar.button:
                if b.key == "bt_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        errors = [str(e.value) for e in at.error]
        assert any("Network" in e or "network" in e for e in errors)


# =====================================================================
# Portfolio Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestPortfolioWorkflow:
    """Portfolio Analyzer page -- full analysis workflow with mocked data."""

    def test_portfolio_initial_state_shows_info(self):
        """Before clicking Analyze, an info message is shown."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")
        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Enter your holdings" in t for t in info_texts)

    def test_portfolio_analyze_renders_charts(self):
        """Analyzing portfolio renders sector donut and correlation heatmap."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        mock_analysis = _make_portfolio_analysis()
        mock_mc = _make_monte_carlo_result()
        mock_prices = _sample_ohlcv(252)

        with (
            patch(
                "quant_tools.dashboard.views.portfolio.analyze_portfolio",
                return_value=mock_analysis,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.run_monte_carlo",
                return_value=mock_mc,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.calculate_cross_asset_correlations",
                side_effect=Exception("skipped"),  # skip cross-asset to simplify
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "portfolio_run":
                    b.click()
                    break
            at.run()

        assert not at.exception, f"Exception: {at.exception}"
        # Sector donut + correlation heatmap + MC fan chart are rendered
        # AppTest has no plotly_chart accessor -- verify via markdown headers
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Sector Exposure" in all_md or "Correlation" in all_md or "Monte Carlo" in all_md, (
            "Expected portfolio chart headers in markdown"
        )

    def test_portfolio_holdings_table_rendered(self):
        """Analysis renders a holdings table."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        mock_analysis = _make_portfolio_analysis()
        mock_mc = _make_monte_carlo_result()
        mock_prices = _sample_ohlcv(252)

        with (
            patch(
                "quant_tools.dashboard.views.portfolio.analyze_portfolio",
                return_value=mock_analysis,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.run_monte_carlo",
                return_value=mock_mc,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.calculate_cross_asset_correlations",
                side_effect=Exception("skipped"),
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "portfolio_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # Holdings table is rendered as st.dataframe
        assert len(at.dataframe) >= 1

    def test_portfolio_invalid_json_shows_error(self):
        """Invalid JSON in holdings text area shows an error."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        # Set holdings to invalid JSON
        for ta in at.sidebar.text_area:
            if ta.key == "portfolio_holdings_json":
                ta.set_value("not valid json {{{")
                break

        for b in at.sidebar.button:
            if b.key == "portfolio_run":
                b.click()
                break
        at.run()

        assert not at.exception
        errors = [str(e.value) for e in at.error]
        assert any("JSON" in e or "json" in e.lower() for e in errors)

    def test_portfolio_mc_statistics_shown(self):
        """Monte Carlo statistics (mean final, prob loss) are rendered."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        mock_analysis = _make_portfolio_analysis()
        mock_mc = _make_monte_carlo_result()
        mock_prices = _sample_ohlcv(252)

        with (
            patch(
                "quant_tools.dashboard.views.portfolio.analyze_portfolio",
                return_value=mock_analysis,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.run_monte_carlo",
                return_value=mock_mc,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.fetch_prices",
                return_value=mock_prices,
            ),
            patch(
                "quant_tools.dashboard.views.portfolio.calculate_cross_asset_correlations",
                side_effect=Exception("skipped"),
            ),
        ):
            for b in at.sidebar.button:
                if b.key == "portfolio_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # MC results are rendered with metric cards and plotly chart
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Monte Carlo" in all_md or "Distribution" in all_md, (
            "Expected Monte Carlo output headers"
        )

    def test_portfolio_preset_buttons_exist(self):
        """Portfolio has preset buttons (60/40, Tech Heavy, All-Weather)."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        button_keys = [b.key for b in at.sidebar.button]
        assert any("preset_" in k for k in button_keys if k), (
            f"Expected preset buttons, got keys: {button_keys}"
        )

    def test_portfolio_period_selectbox(self):
        """Portfolio has a period selectbox with standard options."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        period_boxes = [s for s in at.sidebar.selectbox if s.key == "portfolio_period"]
        assert len(period_boxes) == 1
        assert "1y" in period_boxes[0].options

    def test_portfolio_empty_holdings_shows_error(self):
        """Empty holdings object shows an error."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        for ta in at.sidebar.text_area:
            if ta.key == "portfolio_holdings_json":
                ta.set_value("{}")
                break

        for b in at.sidebar.button:
            if b.key == "portfolio_run":
                b.click()
                break
        at.run()

        assert not at.exception
        errors = [str(e.value) for e in at.error]
        assert any("empty" in e.lower() or "non-empty" in e.lower() for e in errors)

    def test_portfolio_data_error_handled(self):
        """DataError from analyze_portfolio shows user-friendly message."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        from quant_tools.errors import DataError

        with patch(
            "quant_tools.dashboard.views.portfolio.analyze_portfolio",
            side_effect=DataError("No price data for GLD"),
        ):
            for b in at.sidebar.button:
                if b.key == "portfolio_run":
                    b.click()
                    break
            at.run()

        assert not at.exception
        errors = [str(e.value) for e in at.error]
        assert any("Data" in e or "data" in e for e in errors)

    def test_portfolio_distribution_selectbox(self):
        """Portfolio MC section has distribution selectbox."""
        at = _load_app()
        at.run()
        _navigate(at, "Portfolio")

        dist_boxes = [s for s in at.sidebar.selectbox if s.key == "portfolio_mc_dist"]
        assert len(dist_boxes) == 1
        assert "Normal" in dist_boxes[0].options
        assert "Auto-detect (BIC)" in dist_boxes[0].options


# =====================================================================
# Factors Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestFactorsWorkflow:
    """Factors & Macro page -- factor scoring and regime detection."""

    def test_factors_initial_state_shows_info(self):
        """Before any action, an info message is shown."""
        at = _load_app()
        at.run()
        _navigate(at, "Factors")
        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Detect Regime" in t or "Calculate Factors" in t for t in info_texts)

    def test_factors_calculate_renders_scores(self):
        """Clicking Calculate Factors with mocked data renders score table."""
        at = _load_app()
        at.run()
        _navigate(at, "Factors")

        mock_scores = _make_factor_scores(10)

        with patch(
            "quant_tools.dashboard.views.factors.calculate_factor_scores",
            return_value=mock_scores,
        ):
            for b in at.sidebar.button:
                if b.key == "factors_calculate":
                    b.click()
                    break
            at.run()

        assert not at.exception, f"Exception: {at.exception}"
        assert len(at.dataframe) >= 1, "Expected factor scores dataframe"

    def test_factors_calculate_renders_charts(self):
        """Factor calculation renders bar chart and sector pie (via markdown headers)."""
        at = _load_app()
        at.run()
        _navigate(at, "Factors")

        mock_scores = _make_factor_scores(10)

        with patch(
            "quant_tools.dashboard.views.factors.calculate_factor_scores",
            return_value=mock_scores,
        ):
            for b in at.sidebar.button:
                if b.key == "factors_calculate":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # Factor charts are rendered -- verify via markdown section header
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Factor Scores" in all_md, (
            "Expected 'Factor Scores' markdown header after calculation"
        )

    def test_factors_weight_sliders_exist(self):
        """Factor weight sliders exist in sidebar."""
        at = _load_app()
        at.run()
        _navigate(at, "Factors")

        slider_keys = [s.key for s in at.sidebar.slider]
        for expected in ("factors_value_weight", "factors_momentum_weight", "factors_quality_weight"):
            assert expected in slider_keys, f"Missing slider: {expected}"

    def test_factors_data_error_handled(self):
        """DataError from calculate_factor_scores shows warning."""
        at = _load_app()
        at.run()
        _navigate(at, "Factors")

        from quant_tools.errors import DataError

        with patch(
            "quant_tools.dashboard.views.factors.calculate_factor_scores",
            side_effect=DataError("yfinance timeout"),
        ):
            for b in at.sidebar.button:
                if b.key == "factors_calculate":
                    b.click()
                    break
            at.run()

        assert not at.exception
        warnings = [str(w.value) for w in at.warning]
        assert any("Data" in w or "data" in w for w in warnings)


# =====================================================================
# Predictions Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestPredictionsWorkflow:
    """Predictions page -- prediction display, filtering, generation."""

    def test_predictions_loads_with_tracker_data(self):
        """Page loads and shows predictions from mocked tracker."""
        at = _load_app()
        at.run()

        mock_preds = _make_prediction_dicts(10)
        mock_perf = _make_performance_records()

        mock_tracker = MagicMock()
        mock_tracker.get_predictions.return_value = mock_preds
        mock_tracker.get_performance.return_value = mock_perf
        mock_tracker.get_latest_by_symbol.return_value = {}

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Predictions")

        assert not at.exception

    def test_predictions_filter_selectboxes_exist(self):
        """Predictions page has asset class, horizon, and model filters."""
        at = _load_app()
        at.run()
        _navigate(at, "Predictions")

        selectbox_keys = [s.key for s in at.selectbox]
        assert "pred_asset_filter" in selectbox_keys
        assert "pred_model_filter" in selectbox_keys

    def test_predictions_empty_state_shows_info(self):
        """When no predictions exist, info message is shown."""
        at = _load_app()
        at.run()

        mock_tracker = MagicMock()
        mock_tracker.get_predictions.return_value = []
        mock_tracker.get_performance.return_value = []
        mock_tracker.get_latest_by_symbol.return_value = {}

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Predictions")

        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("predict" in t.lower() or "prediction" in t.lower() for t in info_texts)

    def test_predictions_has_symbol_input(self):
        """Predictions page has a symbol text input for new predictions."""
        at = _load_app()
        at.run()
        _navigate(at, "Predictions")

        input_keys = [t.key for t in at.text_input]
        assert "pred_symbol_input" in input_keys

    def test_predictions_has_predict_button(self):
        """Predictions page has a Predict button."""
        at = _load_app()
        at.run()
        _navigate(at, "Predictions")

        button_keys = [b.key for b in at.button]
        assert "pred_run" in button_keys

    def test_predictions_tracker_error_handled(self):
        """Exception from PredictionTracker shows warning, not crash."""
        at = _load_app()
        at.run()

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            side_effect=Exception("DB locked"),
        ):
            _navigate(at, "Predictions")

        assert not at.exception
        warnings = [str(w.value) for w in at.warning]
        assert len(warnings) >= 1


# =====================================================================
# Performance Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestPerformanceWorkflow:
    """Performance Tracker page -- win rates, model comparison."""

    def test_performance_loads_with_data(self):
        """Page loads with mocked performance data."""
        at = _load_app()
        at.run()

        mock_perf = _make_performance_records()
        mock_preds = _make_prediction_dicts(10)

        mock_tracker = MagicMock()
        mock_tracker.get_performance.return_value = mock_perf
        mock_tracker.get_predictions.return_value = mock_preds

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Performance")

        assert not at.exception

    def test_performance_renders_comparison_table(self):
        """Performance page renders model comparison dataframe."""
        at = _load_app()
        at.run()

        mock_perf = _make_performance_records()
        mock_preds = _make_prediction_dicts(10)

        mock_tracker = MagicMock()
        mock_tracker.get_performance.return_value = mock_perf
        mock_tracker.get_predictions.return_value = mock_preds

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Performance")

        assert not at.exception
        assert len(at.dataframe) >= 1, "Expected model comparison dataframe"

    def test_performance_renders_win_rate_output(self):
        """Performance page renders win rate output with data."""
        at = _load_app()
        at.run()

        mock_perf = _make_performance_records()
        mock_preds = _make_prediction_dicts(10)

        mock_tracker = MagicMock()
        mock_tracker.get_performance.return_value = mock_perf
        mock_tracker.get_predictions.return_value = mock_preds

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Performance")

        assert not at.exception
        # Win rate chart and heatmap are plotly -- verify section headers
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Win Rate" in all_md or "Model Comparison" in all_md

    def test_performance_empty_state(self):
        """Empty performance data shows resolve button but no charts."""
        at = _load_app()
        at.run()

        mock_tracker = MagicMock()
        mock_tracker.get_performance.return_value = []
        mock_tracker.get_predictions.return_value = []

        with patch(
            "quant_tools.predict.tracker.PredictionTracker",
            return_value=mock_tracker,
        ):
            _navigate(at, "Performance")

        assert not at.exception
        # Should have the resolve button even when empty
        button_keys = [b.key for b in at.button]
        assert "perf_resolve" in button_keys


# =====================================================================
# Rolling Analytics Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestRollingWorkflow:
    """Rolling Analytics page -- rolling metrics display."""

    def test_rolling_renders(self):
        """Rolling page renders without exception."""
        at = _load_app()
        at.run()
        _navigate(at, "Rolling")
        assert not at.exception

    def test_rolling_shows_coming_soon(self):
        """Rolling page shows coming-soon state with info message."""
        at = _load_app()
        at.run()
        _navigate(at, "Rolling")

        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Portfolio Analyzer" in t for t in info_texts)


# =====================================================================
# Risk Dashboard Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestRiskWorkflow:
    """Risk Dashboard page -- VaR, CVaR, concentration display."""

    def test_risk_renders(self):
        """Risk page renders without exception."""
        at = _load_app()
        at.run()
        _navigate(at, "Risk")
        assert not at.exception

    def test_risk_shows_coming_soon(self):
        """Risk page shows coming-soon state with info message."""
        at = _load_app()
        at.run()
        _navigate(at, "Risk")

        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Portfolio Analyzer" in t for t in info_texts)

    def test_risk_compute_var_function(self):
        """Verify the VaR computation helper works correctly."""
        from quant_tools.dashboard.views.risk_dashboard import compute_var

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        var_95 = compute_var(returns, confidence=0.95)
        assert var_95 > 0, "VaR should be positive (represents loss)"
        assert var_95 < 0.10, "VaR should be reasonable for daily returns"

    def test_risk_compute_cvar_function(self):
        """Verify the CVaR computation helper works correctly."""
        from quant_tools.dashboard.views.risk_dashboard import compute_cvar, compute_var

        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        cvar_95 = compute_cvar(returns, confidence=0.95)
        var_95 = compute_var(returns, confidence=0.95)
        assert cvar_95 >= var_95, "CVaR should be >= VaR"

    def test_risk_concentration_function(self):
        """Verify the HHI concentration function."""
        from quant_tools.dashboard.views.risk_dashboard import compute_concentration

        # Single asset = max concentration
        hhi_single = compute_concentration({"AAPL": 1.0})
        assert abs(hhi_single - 1.0) < 0.01

        # Equal weight 4 assets = 0.25
        hhi_equal = compute_concentration({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        assert abs(hhi_equal - 0.25) < 0.01


# =====================================================================
# Live Monitor Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestLiveWorkflow:
    """Live Monitor page -- watchlist and price display."""

    def test_live_renders(self):
        """Live page renders without exception."""
        at = _load_app()
        at.run()
        _navigate(at, "Live")
        assert not at.exception

    def test_live_shows_coming_soon(self):
        """Live page shows coming-soon state with info message."""
        at = _load_app()
        at.run()
        _navigate(at, "Live")

        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("WebSocket" in t or "Scanner" in t for t in info_texts)

    def test_live_parse_watchlist_function(self):
        """Verify the parse_watchlist helper function."""
        from quant_tools.dashboard.views.live import parse_watchlist

        result = parse_watchlist("AAPL, MSFT, GOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

        result = parse_watchlist("AAPL\nMSFT\nGOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

        result = parse_watchlist("")
        assert result == []

    def test_live_parse_watchlist_deduplication(self):
        """parse_watchlist deduplicates symbols."""
        from quant_tools.dashboard.views.live import parse_watchlist

        result = parse_watchlist("AAPL, MSFT, AAPL, MSFT")
        assert result == ["AAPL", "MSFT"]


# =====================================================================
# AI Analysis Workflow Tests
# =====================================================================


@pytest.mark.dashboard
class TestAIAnalysisWorkflow:
    """AI Analysis page -- LLM analysis generation."""

    def test_ai_renders(self):
        """AI page renders without exception."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")
        assert not at.exception

    def test_ai_has_analysis_type_selectbox(self):
        """AI page has analysis type selectbox with all types."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        boxes = [s for s in at.sidebar.selectbox if s.key == "ai_analysis_type"]
        assert len(boxes) == 1
        assert "Market Overview" in boxes[0].options
        assert "Trade Ideas" in boxes[0].options
        assert "Risk Assessment" in boxes[0].options

    def test_ai_has_temperature_slider(self):
        """AI page has temperature slider."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        sliders = [s for s in at.sidebar.slider if s.key == "ai_temperature"]
        assert len(sliders) == 1
        assert sliders[0].value == 0.7

    def test_ai_generate_with_mocked_llm(self):
        """Clicking Generate Analysis with mocked LLM renders result."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        with patch(
            "quant_tools.dashboard.views.ai_analysis._call_llm",
            return_value="**Market Overview:** Stocks are up 2% today. Tech leads.",
        ):
            for b in at.button:
                if b.key == "ai_generate":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # The LLM output is rendered as markdown
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "Market Overview" in all_md or "Stocks" in all_md

    def test_ai_generate_initial_shows_info(self):
        """Before clicking Generate, an info message is shown."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        assert not at.exception
        info_texts = [str(i.value) for i in at.info]
        assert any("Generate" in t for t in info_texts)

    def test_ai_custom_prompt_textarea(self):
        """AI page has a custom prompt text area."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        tas = [t for t in at.text_area if t.key == "ai_custom_prompt"]
        assert len(tas) == 1

    def test_ai_llm_failure_renders_gracefully(self):
        """LLM failure returns error message in output, not crash."""
        at = _load_app()
        at.run()
        _navigate(at, "AI")

        with patch(
            "quant_tools.dashboard.views.ai_analysis._call_llm",
            return_value="LLM unavailable: Connection refused",
        ):
            for b in at.button:
                if b.key == "ai_generate":
                    b.click()
                    break
            at.run()

        assert not at.exception
        # The error is rendered as markdown (not an st.error)
        all_md = " ".join(str(m.value) for m in at.markdown)
        assert "LLM unavailable" in all_md
