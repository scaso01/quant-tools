"""All dataclasses and enums for quant-tools (~250 lines)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────


class Sector(Enum):
    """GICS sector classifications."""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    MATERIALS = "Materials"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    COMMUNICATION = "Communication Services"


class SignalType(Enum):
    """Scanner signal types."""
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    MEAN_REVERSION = "mean_reversion"


class StrategyType(Enum):
    """Backtest strategy types."""
    SMA_CROSSOVER = "sma_crossover"
    BOLLINGER = "bollinger"
    RSI = "rsi"
    MOMENTUM = "momentum"
    FACTOR = "factor"
    MEAN_REVERSION = "mean_reversion"
    PAIRS = "pairs"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    FACTOR_ROTATION = "factor_rotation"
    RISK_PARITY_DYNAMIC = "risk_parity_dynamic"
    CARRY = "carry"
    VOL_TARGET = "vol_target"


class MacroRegime(Enum):
    """FRED-based macro regime classification."""
    EXPANSION = "expansion"
    SLOWDOWN = "slowdown"
    CONTRACTION = "contraction"
    RECOVERY = "recovery"


class DashboardTab(Enum):
    """TUI dashboard tabs."""
    SCANNER = "scanner"
    BACKTEST = "backtest"
    PORTFOLIO = "portfolio"
    FACTORS = "factors"


class AssetClass(Enum):
    """Supported asset classes."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


class PredictionDirection(Enum):
    """Predicted price direction."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class ConsensusSignal(Enum):
    """Ensemble consensus signal strength."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class PredictionHorizon(Enum):
    """Standard prediction horizons."""
    DAY_1 = 1
    DAY_5 = 5
    DAY_20 = 20
    DAY_60 = 60


class ModelType(Enum):
    """ML model types in the prediction ensemble."""
    GRADIENT_BOOST = "gradient_boost"
    STATS_FORECAST = "stats_forecast"
    NEURAL_FORECAST = "neural_forecast"
    PROPHET = "prophet"
    GARCH = "garch"


class DistributionType(Enum):
    """Monte Carlo return distribution types."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    SKEW_NORMAL = "skew_normal"
    JOHNSON_SU = "johnson_su"
    GED = "ged"


class RollMethod(Enum):
    """Futures contract roll methods."""
    CALENDAR = "calendar"
    VOLUME = "volume"


class RollAdjustment(Enum):
    """Futures continuous series adjustment methods."""
    PANAMA = "panama"
    RATIO = "ratio"
    NONE = "none"


class Region(Enum):
    """Geographic region for universe filtering."""
    US = "US"
    EUROPE = "Europe"
    ASIA_PACIFIC = "Asia Pacific"
    EMERGING = "Emerging"


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class PriceBar:
    """Single OHLCV price bar."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None


@dataclass
class TickerInfo:
    """Basic ticker metadata."""
    symbol: str
    name: str = ""
    sector: Sector | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None


@dataclass
class ScanResult:
    """Result from the stock scanner."""
    symbol: str
    name: str
    sector: str
    signal: SignalType
    score: float
    price: float
    change_1d: float = 0.0
    change_5d: float = 0.0
    change_1m: float = 0.0
    rsi_14: float | None = None
    volume_ratio: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy: StrategyType
    ticker: str
    # None when the caller ran the engine on a bare price array and did not say
    # what window it covers. The engine cannot infer it, and inventing a date
    # here once had every CLI backtest reporting a window it had not run.
    start_date: date | None
    end_date: date | None
    initial_capital: float
    final_value: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    total_trades: int
    profit_factor: float
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioHolding:
    """Single portfolio holding."""
    symbol: str
    weight: float
    current_price: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0


@dataclass
class PortfolioAnalysis:
    """Full portfolio analysis result."""
    holdings: list[PortfolioHolding]
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    sector_exposure: dict[str, float] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_simulations: int
    time_horizon_days: int
    initial_value: float
    mean_final: float
    median_final: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_loss: float
    prob_gain_10pct: float
    simulated_paths: list[list[float]] = field(default_factory=list)
    distribution_used: DistributionType = DistributionType.NORMAL
    distribution_params: DistributionParams | None = None
    skewness: float = 0.0
    kurtosis: float = 3.0


@dataclass
class FactorScore:
    """Factor-based stock score."""
    symbol: str
    name: str
    value_z: float = 0.0
    momentum_z: float = 0.0
    quality_z: float = 0.0
    composite_z: float = 0.0
    rank: int = 0
    sector: str = ""


@dataclass
class MacroSnapshot:
    """Current macro regime snapshot from FRED data."""
    regime: MacroRegime
    gdp_growth: float | None = None
    unemployment: float | None = None
    cpi_yoy: float | None = None
    fed_funds_rate: float | None = None
    yield_spread_10y2y: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    signals: dict[str, str] = field(default_factory=dict)


@dataclass
class RebalanceRecommendation:
    """Factor-tilt rebalance recommendation."""
    regime: MacroRegime
    value_tilt: float
    momentum_tilt: float
    quality_tilt: float
    defensive_tilt: float
    top_picks: list[FactorScore] = field(default_factory=list)
    rationale: str = ""


@dataclass
class DashboardState:
    """Reactive state for the TUI dashboard."""
    active_tab: DashboardTab = DashboardTab.SCANNER
    scan_results: list[ScanResult] = field(default_factory=list)
    backtest_result: BacktestResult | None = None
    portfolio_analysis: PortfolioAnalysis | None = None
    factor_scores: list[FactorScore] = field(default_factory=list)
    macro_snapshot: MacroSnapshot | None = None
    last_update: datetime = field(default_factory=datetime.now)
    status_message: str = "Ready"
    is_loading: bool = False


# ── Constraint & Optimization Models ────────────────────────────────


@dataclass
class AssetConstraint:
    """Per-asset weight constraint for portfolio optimization."""
    symbol: str
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class GroupConstraint:
    """Group/sector weight constraint for portfolio optimization."""
    name: str
    symbols: list[str] = field(default_factory=list)
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class OptimizationConstraints:
    """Full constraint specification for institutional-grade optimization."""
    asset_constraints: list[AssetConstraint] = field(default_factory=list)
    group_constraints: list[GroupConstraint] = field(default_factory=list)
    max_turnover: float | None = None
    current_weights: dict[str, float] = field(default_factory=dict)
    tracking_error_target: float | None = None
    benchmark_returns: Any = None  # NDArray at runtime
    correlation_override: Any = None  # NDArray at runtime


@dataclass
class EfficientFrontierPoint:
    """Single point on the efficient frontier."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class EfficientFrontier:
    """Full efficient frontier from parametric sweep."""
    points: list[EfficientFrontierPoint] = field(default_factory=list)
    max_sharpe_point: EfficientFrontierPoint | None = None
    min_variance_point: EfficientFrontierPoint | None = None


# ── Benchmark Models ────────────────────────────────────────────────


@dataclass
class BenchmarkDefinition:
    """Definition of a benchmark for relative performance measurement."""
    name: str = ""
    symbol: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    returns: Any = None  # NDArray at runtime (for custom upload)


@dataclass
class BenchmarkMetrics:
    """Benchmark-relative performance metrics."""
    benchmark_name: str = ""
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    correlation: float = 0.0
    active_return: float = 0.0


# ── Distribution & Monte Carlo Models ───────────────────────────────


@dataclass
class DistributionParams:
    """Fitted distribution parameters for Monte Carlo simulation."""
    distribution: DistributionType = DistributionType.NORMAL
    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 3.0
    df: float | None = None  # Student-t degrees of freedom
    skew_param: float | None = None  # Skew-normal alpha
    johnson_a: float | None = None
    johnson_b: float | None = None
    johnson_loc: float | None = None
    johnson_scale: float | None = None


# ── Futures Models ──────────────────────────────────────────────────


@dataclass
class FuturesContract:
    """Futures contract identifier."""
    root: str
    month_code: str
    year: int
    exchange: str = ""

    @property
    def ticker(self) -> str:
        """yfinance-style ticker (e.g., 'GCZ25.CMX')."""
        yr = self.year % 100
        base = f"{self.root}{self.month_code}{yr:02d}"
        return f"{base}.{self.exchange}" if self.exchange else base


@dataclass
class CommoditySpec:
    """Specification for a tradable commodity futures product."""
    root: str
    name: str
    category: str
    contract_months: str  # e.g., "GJMQVZ"
    tick_size: float = 0.01
    point_value: float = 1.0
    roll_days_before_expiry: int = 5


# ── Correlation Models ──────────────────────────────────────────────


@dataclass
class RollingCorrelation:
    """Rolling pairwise correlation between two assets."""
    symbol_a: str
    symbol_b: str
    window_days: int
    dates: list[date] = field(default_factory=list)
    correlations: list[float] = field(default_factory=list)


@dataclass
class CrossAssetCorrelation:
    """Full cross-asset correlation matrix."""
    symbols: list[str] = field(default_factory=list)
    asset_classes: list[AssetClass] = field(default_factory=list)
    correlation_matrix: Any = None  # NDArray at runtime
    rolling_correlations: list[RollingCorrelation] = field(default_factory=list)


# ── Universe Models ─────────────────────────────────────────────────


@dataclass
class TickerMeta:
    """Extended ticker metadata for universe files."""
    name: str
    sector: str = ""
    sub_industry: str = ""
    region: str = "US"
    index_membership: list[str] = field(default_factory=list)


# ── Prediction Models ────────────────────────────────────────────────


@dataclass
class PredictionResult:
    """Result from a single ML model prediction."""
    symbol: str
    asset_class: AssetClass
    model_name: str
    direction: PredictionDirection
    magnitude: float | None = None
    confidence: float = 0.5
    horizon_days: int = 5
    horizon_hours: int | None = None
    target_date: str = ""
    interval: str = "1d"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple models."""
    symbol: str
    asset_class: AssetClass
    consensus: ConsensusSignal
    confidence: float
    direction: PredictionDirection
    magnitude: float | None = None
    horizon_days: int = 5
    horizon_hours: int | None = None
    target_date: str = ""
    interval: str = "1d"
    model_predictions: list[PredictionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OHLCV:
    """Single OHLCV bar (provider-agnostic)."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FeatureSet:
    """Feature matrix for ML model input."""
    symbol: str
    features_df: Any = None  # pd.DataFrame at runtime
    target: Any = None  # np.ndarray at runtime
    feature_names: list[str] = field(default_factory=list)
    close_prices: Any = None  # pd.Series at runtime -- raw close for time-series models


@dataclass
class QuantilesResult:
    """Quantile summary for prediction distributions."""
    q10: float = 0.0
    q25: float = 0.0
    q50: float = 0.0
    q75: float = 0.0
    q90: float = 0.0


@dataclass
class ModelPerformance:
    """Aggregate model performance metrics."""
    model_name: str
    asset_class: AssetClass
    total_predictions: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_error: float = 0.0
    sharpe_of_predictions: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionOutcome:
    """Resolved outcome for a prediction."""
    prediction_id: int
    actual_direction: PredictionDirection
    actual_magnitude: float | None = None
    correct: bool = False
    error_pct: float = 0.0
    resolved_at: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioOptimization:
    """Result from portfolio optimization."""
    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = "markowitz"
    constraints_used: OptimizationConstraints | None = None
    efficient_frontier: EfficientFrontier | None = None
    tracking_error: float | None = None


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    var_95: float = 0.0
    cvar_95: float = 0.0
    kelly_fraction: float = 0.0
    max_position_size: float = 0.0
