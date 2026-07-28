"""Quick-navigation command bar for the Streamlit dashboard."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Data Types ──────────────────────────────────────────────────────────


@dataclass
class CommandResult:
    """Parsed result from the command bar input.

    Attributes:
        page: Full page name with emoji prefix (e.g. ``"\U0001f50d Market Scanner"``),
            or empty string if not resolved.
        symbol: Extracted ticker symbol (upper-cased), or empty string.
        params: Extra key-value parameters parsed from the command.
        raw: The original input text.
    """

    page: str = ""
    symbol: str = ""
    params: dict[str, str] = field(default_factory=dict)
    raw: str = ""


# ── Page Alias Mapping ──────────────────────────────────────────────────

PAGE_ALIASES: dict[str, str] = {
    # Scanner
    "scan": "\U0001f4e1 Market Scanner",
    "scanner": "\U0001f4e1 Market Scanner",
    # Backtester
    "bt": "\U0001f4c8 Strategy Backtester",
    "backtest": "\U0001f4c8 Strategy Backtester",
    # Portfolio
    "port": "\U0001f4bc Portfolio Analyzer",
    "portfolio": "\U0001f4bc Portfolio Analyzer",
    # Factors & Macro
    "factors": "\U0001f3db\ufe0f Factors & Macro",
    "macro": "\U0001f3db\ufe0f Factors & Macro",
    # Predictions
    "predict": "\U0001f52e Predictions",
    "predictions": "\U0001f52e Predictions",
    # Performance
    "perf": "\U0001f4ca Performance",
    "performance": "\U0001f4ca Performance",
    # AI Analysis
    "ai": "\U0001f916 AI Analysis",
    # Risk
    "risk": "\U0001f6e1\ufe0f Risk Dashboard",
}

# Pattern: optional slash, command word, optional whitespace, optional symbol/args
_CMD_PATTERN = re.compile(
    r"^/?\s*(?P<cmd>[a-zA-Z_]+)"  # command (with optional leading slash)
    r"(?:\s+(?P<rest>.+))?$"      # optional rest (symbol, params)
)


def parse_command(text: str) -> CommandResult:
    """Parse a command-bar input string into a :class:`CommandResult`.

    Supported formats::

        /bt AAPL        -> page=Backtester, symbol=AAPL
        /predict MSFT   -> page=Predictions, symbol=MSFT
        /scan sp500      -> page=Scanner, params={"universe": "sp500"}
        AAPL             -> page="", symbol=AAPL (plain ticker lookup)
        /help            -> page=Guide, no symbol

    Args:
        text: Raw user input from the command bar.

    Returns:
        A :class:`CommandResult` with resolved page, symbol, and params.
    """
    raw = text.strip()
    if not raw:
        return CommandResult(raw=raw)

    match = _CMD_PATTERN.match(raw)
    if match is None:
        # Treat the entire input as a potential symbol
        return CommandResult(symbol=raw.upper(), raw=raw)

    cmd = match.group("cmd").lower()
    rest = (match.group("rest") or "").strip()

    # Check if the command maps to a page alias
    page = PAGE_ALIASES.get(cmd, "")

    if page:
        # Scanner special case: rest might be a universe name, not a symbol
        if cmd in ("scan", "scanner") and rest:
            return CommandResult(
                page=page,
                params={"universe": rest},
                raw=raw,
            )

        return CommandResult(
            page=page,
            symbol=rest.upper() if rest else "",
            raw=raw,
        )

    # No alias match -- treat the whole input (cmd + rest) as a plain symbol
    # This handles bare tickers like "AAPL" typed without a slash
    candidate = raw.lstrip("/").strip().upper()
    return CommandResult(symbol=candidate, raw=raw)
