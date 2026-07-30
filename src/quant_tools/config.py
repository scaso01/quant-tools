"""Configuration loaded from .env file (deco-monitor pattern)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from quant_tools.errors import ConfigError


def _data_dir() -> Path:
    """Where the cache and the log live unless the user says otherwise.

    Both used to default to a bare filename, which resolves against the working
    directory. Launched from a directory you cannot write to, the app died with
    a PermissionError before the dashboard came up; launched from anywhere else
    it dropped a log file and a private copy of the cache into whatever folder
    you happened to be standing in.

    Does not create the directory. The two writers do that when they open.
    """
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "quant-tools"


def default_cache_db_path() -> str:
    """Cache location: CACHE_DB_PATH if set, otherwise the data directory."""
    return os.getenv("CACHE_DB_PATH") or str(_data_dir() / "quant_tools_cache.db")


def default_log_file() -> str:
    """Log location: LOG_FILE if set, otherwise the data directory."""
    return os.getenv("LOG_FILE") or str(_data_dir() / "quant_tools.log")


def _find_env() -> Path:
    """Walk up from CWD or package dir to find .env."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        env = base / ".env"
        if env.exists():
            return env
    return Path.cwd() / ".env"


load_dotenv(_find_env())


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


@dataclass(frozen=True)
class Config:
    """Application configuration. Frozen for thread safety.

    Secrets carry repr=False. The dataclass repr is printed by any pytest
    assertion that touches a Config, which put a live API key straight into
    the test output, and would put it into a public CI log.
    """

    # API keys
    fred_api_key: str = field(default=os.getenv("FRED_API_KEY", ""), repr=False)

    # Cache settings
    cache_db_path: str = field(default_factory=default_cache_db_path)
    cache_ttl_hours: int = _int("CACHE_TTL_HOURS", 24)

    # Scanner settings
    scanner_max_workers: int = _int("SCANNER_MAX_WORKERS", 8)
    scanner_top_n: int = _int("SCANNER_TOP_N", 20)

    # Backtest defaults
    backtest_start_date: str = os.getenv("BACKTEST_START_DATE", "2020-01-01")
    backtest_initial_capital: float = _float("BACKTEST_INITIAL_CAPITAL", 100000)

    # Monte Carlo
    monte_carlo_simulations: int = _int("MONTE_CARLO_SIMULATIONS", 1000)

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = field(default_factory=default_log_file)

    # Prediction settings
    prediction_horizon_days: int = _int("PREDICTION_HORIZON_DAYS", 5)
    prediction_min_history_days: int = _int("PREDICTION_MIN_HISTORY_DAYS", 252)
    ensemble_min_confidence: float = _float("ENSEMBLE_MIN_CONFIDENCE", 0.3)

    # Crypto settings
    crypto_default_exchange: str = os.getenv("CRYPTO_DEFAULT_EXCHANGE", "kraken")

    # Finnhub API key (free at https://finnhub.io/)
    finnhub_api_key: str = field(default=os.getenv("FINNHUB_API_KEY", ""), repr=False)

    # SEC EDGAR (email required by SEC fair-use policy)
    sec_edgar_email: str = os.getenv("SEC_EDGAR_EMAIL", "")

    # Reddit API (free at https://www.reddit.com/prefs/apps)
    reddit_client_id: str = field(default=os.getenv("REDDIT_CLIENT_ID", ""), repr=False)
    reddit_client_secret: str = field(
        default=os.getenv("REDDIT_CLIENT_SECRET", ""), repr=False
    )

    # Commodity symbol map (futures tickers preferred, ETF fallback)
    commodity_etf_proxies: dict[str, str] = field(default_factory=lambda: {
        "GOLD": "GC=F", "OIL": "CL=F", "SILVER": "SI=F",
        "NATGAS": "NG=F", "COPPER": "HG=F", "PLATINUM": "PL=F",
        "WHEAT": "ZW=F", "CORN": "ZC=F", "SOYBEANS": "ZS=F",
        "AGRICULTURE": "DBA",
    })

    # Portfolio optimization
    optimization_method: str = os.getenv("OPTIMIZATION_METHOD", "markowitz")
    risk_free_rate: float = _float("RISK_FREE_RATE", 0.04)

    # LLM settings, any OpenAI-compatible server (llama.cpp, Ollama, vLLM)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8080")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout: int = _int("LLM_TIMEOUT", 60)

    # Alpaca paper trading
    alpaca_api_key: str = field(default=os.getenv("ALPACA_API_KEY", ""), repr=False)
    alpaca_secret_key: str = field(
        default=os.getenv("ALPACA_SECRET_KEY", ""), repr=False
    )

    # Backtest cost defaults
    default_cost_bps: float = _float("DEFAULT_COST_BPS", 10.0)


def validate_config(config: Config) -> bool:
    """Validate configuration values. Fail-fast on bad config.

    Args:
        config: Configuration instance to validate.

    Returns:
        True if valid.

    Raises:
        ConfigError: If configuration is invalid.
    """
    if config.cache_ttl_hours < 1:
        raise ConfigError(f"CACHE_TTL_HOURS must be >= 1, got {config.cache_ttl_hours}")

    if config.scanner_max_workers < 1 or config.scanner_max_workers > 32:
        raise ConfigError(f"SCANNER_MAX_WORKERS must be 1-32, got {config.scanner_max_workers}")

    if config.backtest_initial_capital <= 0:
        raise ConfigError(f"BACKTEST_INITIAL_CAPITAL must be > 0, got {config.backtest_initial_capital}")

    if config.monte_carlo_simulations < 100:
        raise ConfigError(f"MONTE_CARLO_SIMULATIONS must be >= 100, got {config.monte_carlo_simulations}")

    if config.prediction_horizon_days < 1:
        raise ConfigError(f"PREDICTION_HORIZON_DAYS must be >= 1, got {config.prediction_horizon_days}")

    if not 0.0 <= config.ensemble_min_confidence <= 1.0:
        raise ConfigError(f"ENSEMBLE_MIN_CONFIDENCE must be 0-1, got {config.ensemble_min_confidence}")

    if config.risk_free_rate < 0:
        raise ConfigError(f"RISK_FREE_RATE must be >= 0, got {config.risk_free_rate}")

    return True


cfg = Config()
validate_config(cfg)
