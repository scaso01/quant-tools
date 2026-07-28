# quant-tools

Multi-asset quantitative finance toolkit for retail market analysis, backtesting, portfolio optimization, factor-based investing, and ML-powered predictions with win-rate tracking.

**v0.1.0** | Python 3.11+ | 10-page dashboard | Zero cost (all free data providers)

## Quick Start

```bash
# Install
git clone https://github.com/scaso01/quant-tools.git
cd quant-tools
pip install -e ".[dev]"

# Optional: ML prediction engine (requires Python 3.11-3.13; torch-based deps
# do not yet support 3.14). Core features work without it.
pip install -e ".[ml]"

# Configure API keys (all free) — only needed for the features you use
cp .env.example .env   # then fill in FRED, Reddit, SEC EDGAR email, etc.

# Scan the S&P 500 for top signals
quant-tools scan --universe sp500 --top-n 20

# Backtest a strategy
quant-tools backtest --ticker AAPL --strategy sma_crossover --start 2020-01-01

# ML prediction
quant-tools predict AAPL --horizon 5

# Launch the web dashboard
quant-tools web
```

## Asset Classes

4 asset classes with unified data providers:

| Class | Provider | Source |
|-------|----------|--------|
| Equities | `EquityProvider` | yfinance + stooq |
| Crypto | `CryptoProvider` | CCXT |
| Forex | `ForexProvider` | yfinance + CCXT |
| Commodities | `CommodityProvider` | COT reports + ETF proxies |

Symbol auto-detection routes queries to the correct provider -- `BTC/USDT` hits crypto, `EUR/USD` hits forex, `AAPL` hits equities.

## CLI Reference

```bash
# Scanner
quant-tools scan [--universe sp500|russell1000|international|emerging|global]
                 [--sector Tech] [--top-n 20]
                 [--asset-class equity|crypto|forex|commodity|all]

# Backtesting
quant-tools backtest --ticker AAPL --strategy sma_crossover --start 2020-01-01

# Portfolio Analysis
quant-tools portfolio --holdings '{"AAPL":0.3,"BND":0.4,"GLD":0.3}'

# Factor Investing
quant-tools factors [--universe sp500] [--regime]

# ML Predictions
quant-tools predict AAPL [--horizon 5] [--asset-class equity]
quant-tools predict BTC/USDT --asset-class crypto
quant-tools track [--symbol AAPL] [--model gradient_boost] [--limit 20]
quant-tools resolve                    # Resolve pending predictions against outcomes

# Data Tools
quant-tools filings AAPL --type 10-K --count 5     # SEC EDGAR filings
quant-tools sentiment AAPL                           # Reddit sentiment analysis
quant-tools macro US --indicator gdp --range 2010:2025  # World Bank macro data

# Dashboard
quant-tools web                        # Streamlit web dashboard (10 pages)
quant-tools dashboard                  # TUI dashboard
```

## Prediction Engine

Requires the `[ml]` extra (`pip install -e ".[ml]"`, Python 3.11-3.13). If no ML packages are installed, `quant-tools predict` raises a clear error instead of returning a fake neutral forecast.

12 ML models combined via weighted ensemble voting (core seven below, plus TFT, PatchTST, N-HiTS, TimesNet, and foundation models via the `[foundation]` extra):

| Model | Type | Predicts |
|-------|------|----------|
| LightGBM | Gradient boosting | Direction + magnitude |
| AutoARIMA | Statistical | Trend |
| ETS | Exponential smoothing | Trend |
| LSTM | Neural network | Temporal patterns |
| N-BEATS | Neural network | Temporal patterns |
| Prophet | Additive model | Seasonality |
| GARCH | Volatility model | Volatility (confidence adjustment) |

All models use walk-forward validation (`TimeSeriesSplit`). Ensemble weights auto-adjust based on tracked win rates per model per asset class (5% floor). The prediction tracker stores forecasts in SQLite and resolves them against actual outcomes.

## Web Dashboard

10-page Streamlit dashboard with dark theme (`quant-tools web`):

| Page | Description |
|------|-------------|
| Market Scanner | Multi-asset scan with universe presets, signal breakdowns |
| Strategy Backtester | Equity curve with drawdown overlay, strategy comparison |
| Portfolio Analyzer | Sector donut, correlation heatmap, Monte Carlo fan chart (5 distributions), optimization (Markowitz/risk parity/min variance), benchmark metrics |
| Factors & Macro | FRED regime detection, factor z-scores, rebalance recommendations |
| Predictions | Current predictions, model agreement, confidence distribution |
| Performance | Win rate tracker, model comparison, confidence vs accuracy scatter |

Monte Carlo simulations fit 5 distributions (Normal, Student-t, Skew-Normal, Johnson SU, GED) + GARCH(1,1), auto-selecting by BIC. Portfolio optimization uses cvxpy for convex problems (Markowitz, min-variance) and scipy for non-convex (risk parity).

## Architecture

```
src/quant_tools/
├── data/
│   ├── providers/       # Unified data layer: base ABC, equity, crypto, forex, commodity, registry
│   ├── sentiment/       # Reddit sentiment via praw
│   ├── universes/       # Static JSON: S&P 500 (518), Russell 1000 (922), International (81)
│   ├── futures.py       # 21 commodity products, calendar + volume roll, Panama back-adjustment
│   └── indicators_enhanced.py  # pandas-ta wrapper (200+ indicators)
├── predict/
│   ├── models/          # 5 ML model modules (gradient_boost, stats_forecast, neural, prophet, garch)
│   ├── features.py      # Feature engineering (51 features from OHLCV)
│   ├── ensemble.py      # Weighted voting combiner
│   ├── tracker.py       # Prediction storage + retrieval (SQLite)
│   └── resolver.py      # Outcome resolution + weight updates
├── scanner/             # Multi-asset concurrent scanner + signal explanations
├── backtest/            # Vectorized numpy backtest engine + benchmark metrics
├── portfolio/           # Monte Carlo (5 dist + GARCH), cvxpy optimizer, cross-correlation
├── factors/             # Factor scoring + macro regime detection
├── tui/                 # Textual TUI dashboard
└── dashboard/           # Streamlit web dashboard (10 pages)
```

## Docker Deployment

```bash
# Set your FRED key in the environment (or a .env file next to the compose file)
export FRED_API_KEY=your_key

docker compose -f docker/docker-compose.yml up -d
# Dashboard at http://localhost:8503
```

Multi-stage Dockerfile (builder + runtime). Or skip Docker entirely: `quant-tools web` runs Streamlit directly on port 8501.

## Testing

```bash
# Unit tests
python -m pytest tests/ --ignore=tests/e2e -v

# Everything, including E2E Playwright tests (need a running dashboard)
python -m pytest tests/ -v
```

Network guards block all real HTTP in unit tests. E2E tests use Playwright headless Chrome against a running dashboard.

## Dependencies

No `ta-lib` or `vectorbt` -- indicators hand-rolled in numpy to avoid C-extension issues on newer Python versions. Key packages: `streamlit`, `plotly`, `ccxt`, `lightgbm`, `statsforecast`, `neuralforecast`, `prophet`, `arch`, `cvxpy`, `sec-edgar-downloader`, `praw`. All free and unlimited.

## License

MIT
