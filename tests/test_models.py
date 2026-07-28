"""Tests for models.py."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_tools.models import (
    AssetClass,
    BacktestResult,
    ConsensusSignal,
    DashboardState,
    DashboardTab,
    EnsemblePrediction,
    FactorScore,
    FeatureSet,
    MacroRegime,
    MacroSnapshot,
    ModelPerformance,
    ModelType,
    MonteCarloResult,
    OHLCV,
    PortfolioAnalysis,
    PortfolioHolding,
    PortfolioOptimization,
    PredictionDirection,
    PredictionHorizon,
    PredictionOutcome,
    PredictionResult,
    PriceBar,
    RebalanceRecommendation,
    RiskMetrics,
    ScanResult,
    Sector,
    SignalType,
    StrategyType,
    TickerInfo,
)


class TestEnums:
    """Test enum definitions."""

    def test_sector_values(self):
        assert Sector.TECHNOLOGY.value == "Technology"
        assert Sector.FINANCIALS.value == "Financials"
        assert len(Sector) == 11

    def test_signal_type_values(self):
        assert SignalType.MOMENTUM.value == "momentum"
        assert len(SignalType) == 4

    def test_strategy_type_values(self):
        assert StrategyType.SMA_CROSSOVER.value == "sma_crossover"
        assert len(StrategyType) == 13  # 5 original + 8 new (WU8 integration)

    def test_macro_regime_values(self):
        assert MacroRegime.EXPANSION.value == "expansion"
        assert len(MacroRegime) == 4

    def test_dashboard_tab_values(self):
        assert DashboardTab.SCANNER.value == "scanner"
        assert len(DashboardTab) == 4


class TestPriceBar:
    """Test PriceBar dataclass."""

    def test_create_price_bar(self):
        bar = PriceBar(
            date=date(2024, 1, 2),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1_000_000,
        )
        assert bar.close == 154.0
        assert bar.adj_close is None

    def test_price_bar_with_adj_close(self):
        bar = PriceBar(
            date=date(2024, 1, 2),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1_000_000,
            adj_close=153.5,
        )
        assert bar.adj_close == 153.5


class TestScanResult:
    """Test ScanResult dataclass."""

    def test_create_scan_result(self):
        r = ScanResult(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            signal=SignalType.MOMENTUM,
            score=85.0,
            price=190.0,
            change_1d=1.5,
        )
        assert r.symbol == "AAPL"
        assert r.signal == SignalType.MOMENTUM
        assert r.rsi_14 is None

    def test_scan_result_defaults(self):
        r = ScanResult(
            symbol="TEST", name="Test", sector="Tech",
            signal=SignalType.VALUE, score=50.0, price=100.0,
        )
        assert r.change_1d == 0.0
        assert r.details == {}


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_create_backtest_result(self):
        r = BacktestResult(
            strategy=StrategyType.SMA_CROSSOVER,
            ticker="AAPL",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 1, 1),
            initial_capital=100_000,
            final_value=150_000,
            total_return=0.5,
            cagr=0.107,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.15,
            max_drawdown_duration_days=60,
            win_rate=0.55,
            total_trades=50,
            profit_factor=1.8,
        )
        assert r.total_return == 0.5
        assert r.equity_curve == []


class TestPortfolioModels:
    """Test portfolio dataclasses."""

    def test_portfolio_holding(self):
        h = PortfolioHolding(symbol="AAPL", weight=0.3, current_price=190.0)
        assert h.weight == 0.3
        assert h.sharpe == 0.0

    def test_portfolio_analysis(self):
        a = PortfolioAnalysis(
            holdings=[PortfolioHolding(symbol="AAPL", weight=1.0)],
            total_return=0.15,
            annualized_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            max_drawdown=-0.10,
        )
        assert len(a.holdings) == 1
        assert a.correlation_matrix == {}


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""

    def test_create_monte_carlo(self):
        r = MonteCarloResult(
            n_simulations=1000,
            time_horizon_days=252,
            initial_value=100_000,
            mean_final=112_000,
            median_final=110_000,
            percentile_5=85_000,
            percentile_25=98_000,
            percentile_75=125_000,
            percentile_95=145_000,
            prob_loss=0.15,
            prob_gain_10pct=0.60,
        )
        assert r.n_simulations == 1000
        assert r.simulated_paths == []


class TestFactorModels:
    """Test factor dataclasses."""

    def test_factor_score(self):
        f = FactorScore(
            symbol="AAPL", name="Apple", value_z=-0.5,
            momentum_z=1.2, quality_z=0.8, composite_z=0.5, rank=1,
        )
        assert f.composite_z == 0.5

    def test_macro_snapshot(self):
        s = MacroSnapshot(regime=MacroRegime.EXPANSION, gdp_growth=2.5)
        assert s.regime == MacroRegime.EXPANSION
        assert s.signals == {}

    def test_rebalance_recommendation(self):
        r = RebalanceRecommendation(
            regime=MacroRegime.EXPANSION,
            value_tilt=0.2,
            momentum_tilt=0.3,
            quality_tilt=0.3,
            defensive_tilt=0.2,
            rationale="Expansion favors momentum and quality",
        )
        assert r.value_tilt + r.momentum_tilt + r.quality_tilt + r.defensive_tilt == pytest.approx(1.0)


class TestDashboardState:
    """Test DashboardState dataclass."""

    def test_default_state(self):
        s = DashboardState()
        assert s.active_tab == DashboardTab.SCANNER
        assert s.scan_results == []
        assert s.backtest_result is None
        assert s.is_loading is False
        assert s.status_message == "Ready"


class TestAssetClass:
    """Test AssetClass enum."""

    def test_values(self):
        assert AssetClass.EQUITY.value == "equity"
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.FOREX.value == "forex"
        assert AssetClass.COMMODITY.value == "commodity"
        assert len(AssetClass) == 4

    def test_from_string(self):
        assert AssetClass("equity") == AssetClass.EQUITY
        assert AssetClass("crypto") == AssetClass.CRYPTO


class TestPredictionDirection:
    """Test PredictionDirection enum."""

    def test_values(self):
        assert PredictionDirection.UP.value == "up"
        assert PredictionDirection.DOWN.value == "down"
        assert PredictionDirection.NEUTRAL.value == "neutral"
        assert len(PredictionDirection) == 3


class TestConsensusSignal:
    """Test ConsensusSignal enum."""

    def test_values(self):
        assert ConsensusSignal.STRONG_BUY.value == "strong_buy"
        assert ConsensusSignal.HOLD.value == "hold"
        assert ConsensusSignal.STRONG_SELL.value == "strong_sell"
        assert len(ConsensusSignal) == 5


class TestPredictionHorizon:
    """Test PredictionHorizon enum."""

    def test_values(self):
        assert PredictionHorizon.DAY_1.value == 1
        assert PredictionHorizon.DAY_5.value == 5
        assert PredictionHorizon.DAY_20.value == 20
        assert PredictionHorizon.DAY_60.value == 60
        assert len(PredictionHorizon) == 4


class TestModelType:
    """Test ModelType enum."""

    def test_values(self):
        assert ModelType.GRADIENT_BOOST.value == "gradient_boost"
        assert ModelType.GARCH.value == "garch"
        assert len(ModelType) == 5


class TestPredictionResult:
    """Test PredictionResult dataclass."""

    def test_create(self):
        r = PredictionResult(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            model_name="gradient_boost",
            direction=PredictionDirection.UP,
            magnitude=0.023,
            confidence=0.65,
            horizon_days=5,
            target_date="2024-01-10",
        )
        assert r.symbol == "AAPL"
        assert r.direction == PredictionDirection.UP
        assert r.confidence == 0.65
        assert r.magnitude == 0.023

    def test_defaults(self):
        r = PredictionResult(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            model_name="prophet",
            direction=PredictionDirection.NEUTRAL,
        )
        assert r.magnitude is None
        assert r.confidence == 0.5
        assert r.horizon_days == 5
        assert r.target_date == ""


class TestEnsemblePrediction:
    """Test EnsemblePrediction dataclass."""

    def test_create(self):
        p = EnsemblePrediction(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            consensus=ConsensusSignal.BUY,
            confidence=0.62,
            direction=PredictionDirection.UP,
            magnitude=0.02,
            horizon_days=5,
            target_date="2024-01-10",
        )
        assert p.consensus == ConsensusSignal.BUY
        assert p.model_predictions == []

    def test_with_model_predictions(self):
        model_pred = PredictionResult(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            model_name="lightgbm",
            direction=PredictionDirection.UP,
            confidence=0.7,
        )
        p = EnsemblePrediction(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            consensus=ConsensusSignal.BUY,
            confidence=0.62,
            direction=PredictionDirection.UP,
            model_predictions=[model_pred],
        )
        assert len(p.model_predictions) == 1
        assert p.model_predictions[0].model_name == "lightgbm"


class TestOHLCV:
    """Test OHLCV dataclass."""

    def test_create(self):
        bar = OHLCV(
            date=date(2024, 1, 2),
            open=150.0, high=155.0, low=149.0,
            close=154.0, volume=1_000_000.0,
        )
        assert bar.close == 154.0


class TestFeatureSet:
    """Test FeatureSet dataclass."""

    def test_create(self):
        fs = FeatureSet(symbol="AAPL", feature_names=["rsi_14", "macd"])
        assert fs.symbol == "AAPL"
        assert len(fs.feature_names) == 2
        assert fs.features_df is None
        assert fs.target is None


class TestModelPerformance:
    """Test ModelPerformance dataclass."""

    def test_create(self):
        mp = ModelPerformance(
            model_name="gradient_boost",
            asset_class=AssetClass.EQUITY,
            total_predictions=100,
            wins=62,
            losses=38,
            win_rate=0.62,
        )
        assert mp.win_rate == 0.62
        assert mp.avg_error == 0.0


class TestPredictionOutcome:
    """Test PredictionOutcome dataclass."""

    def test_create(self):
        o = PredictionOutcome(
            prediction_id=42,
            actual_direction=PredictionDirection.UP,
            actual_magnitude=0.015,
            correct=True,
            error_pct=0.008,
        )
        assert o.correct is True
        assert o.prediction_id == 42


class TestPortfolioOptimization:
    """Test PortfolioOptimization dataclass."""

    def test_create(self):
        po = PortfolioOptimization(
            weights={"AAPL": 0.5, "BND": 0.5},
            expected_return=0.08,
            volatility=0.12,
            sharpe_ratio=0.67,
            method="markowitz",
        )
        assert po.method == "markowitz"
        assert sum(po.weights.values()) == pytest.approx(1.0)

    def test_defaults(self):
        po = PortfolioOptimization()
        assert po.weights == {}
        assert po.method == "markowitz"


class TestRiskMetrics:
    """Test RiskMetrics dataclass."""

    def test_create(self):
        rm = RiskMetrics(var_95=-0.02, cvar_95=-0.035, kelly_fraction=0.15)
        assert rm.var_95 == -0.02
        assert rm.max_position_size == 0.0

    def test_defaults(self):
        rm = RiskMetrics()
        assert rm.var_95 == 0.0
        assert rm.kelly_fraction == 0.0
