"""Regression tests for the window a backtest actually covers.

The README's own example, `quant-tools backtest --ticker AAPL --strategy
sma_crossover --start 2020-01-01`, had two independent problems.

fetch_prices only honoured `start` when `end` was given too, so an open-ended
request fell through to `period="1y"` and silently backtested the last twelve
months instead of the six and a half years asked for.

The engine then filled unset dates with a hardcoded date(2020, 1, 1) to
date(2020, 12, 31), and the CLI never passed any, so every run printed that
same invented range no matter what it had actually run on.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_tools.backtest.engine import run_backtest


def _ohlcv(n: int = 60, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    close = np.linspace(100.0, 130.0, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
            "Adj Close": close,
        },
        index=idx,
    )


class TestOpenEndedRange:
    """fetch_prices must not throw away a start date."""

    def _call(self, tmp_db, **kwargs):
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data
        from quant_tools.retry import yfinance_breaker

        market_data._cache = CacheManager(tmp_db)
        yfinance_breaker.reset()

        with patch("quant_tools.data.market_data.yf") as mock_yf:
            mock_yf.download.return_value = _ohlcv()
            market_data.fetch_prices("AAPL", **kwargs)
            return mock_yf.download.call_args

    def test_start_without_end_is_honoured(self, tmp_db):
        """The bug: this used to become period="1y" and lose the start date."""
        call = self._call(tmp_db, start="2020-01-01")
        assert call.kwargs.get("start") == "2020-01-01"
        assert "period" not in call.kwargs

    def test_start_without_end_runs_to_today(self, tmp_db):
        call = self._call(tmp_db, start="2020-01-01")
        assert call.kwargs.get("end") == date.today().isoformat()

    def test_end_without_start_is_honoured(self, tmp_db):
        call = self._call(tmp_db, end="2020-06-01")
        assert call.kwargs.get("end") == "2020-06-01"
        assert "period" not in call.kwargs

    def test_neither_falls_back_to_period(self, tmp_db):
        """With no range at all, period is still the right behaviour."""
        call = self._call(tmp_db, period="5y")
        assert call.kwargs.get("period") == "5y"
        assert not call.kwargs.get("start")


class TestReportedDates:
    """The engine must not invent a window it was not given."""

    def _result(self, **kwargs):
        close = np.linspace(100.0, 120.0, 60)
        positions = np.where(np.arange(60) < 30, 1.0, 0.0)
        return run_backtest(close=close, positions=positions, **kwargs)

    def test_unset_dates_stay_none(self):
        """The bug: these came back as 2020-01-01 and 2020-12-31."""
        res = self._result()
        assert res.start_date is None
        assert res.end_date is None

    def test_supplied_dates_are_kept(self):
        res = self._result(start_date=date(2022, 3, 1), end_date=date(2023, 3, 1))
        assert res.start_date == date(2022, 3, 1)
        assert res.end_date == date(2023, 3, 1)

    def test_degenerate_input_does_not_invent_dates(self):
        """The empty-result path had the same hardcoded fallback."""
        res = run_backtest(close=np.array([100.0]), positions=np.array([1.0]))
        assert res.start_date is None
        assert res.end_date is None

    def test_printer_survives_missing_dates(self, capsys):
        from quant_tools.backtest.cli import _print_result

        _print_result(self._result())
        out = capsys.readouterr().out
        assert "2020-01-01" not in out
        assert "not recorded" in out


class TestCliReportsRealWindow:
    """The CLI knows the dates from the price index, so it must pass them."""

    def test_cli_reports_the_data_window(self, capsys, tmp_db):
        from quant_tools.backtest import cli as backtest_cli
        from quant_tools.cache import CacheManager
        from quant_tools.data import market_data
        from quant_tools.retry import yfinance_breaker

        market_data._cache = CacheManager(tmp_db)
        yfinance_breaker.reset()
        df = _ohlcv(n=120, start="2021-02-01")

        with patch("quant_tools.data.market_data.yf") as mock_yf:
            mock_yf.download.return_value = df
            result = backtest_cli.backtest_command(
                ["--ticker", "AAPL", "--strategy", "sma_crossover", "--start", "2021-02-01"]
            )

        assert result.start_date == df.index[0].date()
        assert result.end_date == df.index[-1].date()
        out = capsys.readouterr().out
        assert str(df.index[-1].date()) in out
        assert "2020-12-31" not in out
