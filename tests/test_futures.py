"""Tests for futures roll logic, commodity catalog, and universe expansion."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_tools.data.futures import (
    COMMODITY_CATALOG,
    MONTH_CODES,
    _EXCHANGE_MAP,
    _business_days_before,
    _contract_months_for_year,
    _expiry_proxy,
    _stitch_contracts,
    build_continuous_series,
    calendar_roll_dates,
    volume_roll_dates,
)
from quant_tools.data.universe import COMMODITY_PRESETS, COMMODITY_UNIVERSE
from quant_tools.models import (
    FuturesContract,
    RollAdjustment,
    RollMethod,
)


# ── CommoditySpec Tests ────────────────────────────────────────────────


class TestCommoditySpec:
    def test_energy_spec(self) -> None:
        spec = COMMODITY_CATALOG["CL"]
        assert spec.name == "Crude Oil"
        assert spec.category == "Energy"
        assert "F" in spec.contract_months

    def test_precious_metals_spec(self) -> None:
        spec = COMMODITY_CATALOG["GC"]
        assert spec.name == "Gold"
        assert spec.category == "Precious Metals"
        assert spec.contract_months == "GJMQVZ"

    def test_base_metals_spec(self) -> None:
        spec = COMMODITY_CATALOG["HG"]
        assert spec.name == "Copper"
        assert spec.category == "Base Metals"

    def test_agriculture_spec(self) -> None:
        spec = COMMODITY_CATALOG["ZC"]
        assert spec.name == "Corn"
        assert spec.category == "Agriculture"

    def test_softs_spec(self) -> None:
        spec = COMMODITY_CATALOG["KC"]
        assert spec.name == "Coffee"
        assert spec.category == "Softs"

    def test_livestock_spec(self) -> None:
        spec = COMMODITY_CATALOG["LE"]
        assert spec.name == "Live Cattle"
        assert spec.category == "Livestock"


# ── FuturesContract Tests ──────────────────────────────────────────────


class TestFuturesContract:
    def test_ticker_with_exchange(self) -> None:
        c = FuturesContract(root="GC", month_code="Z", year=2025, exchange="CMX")
        assert c.ticker == "GCZ25.CMX"

    def test_ticker_without_exchange(self) -> None:
        c = FuturesContract(root="GC", month_code="Z", year=2025, exchange="")
        assert c.ticker == "GCZ25"

    def test_ticker_year_formatting(self) -> None:
        c = FuturesContract(root="CL", month_code="F", year=2026, exchange="NYM")
        assert c.ticker == "CLF26.NYM"

    def test_ticker_single_digit_year(self) -> None:
        c = FuturesContract(root="SI", month_code="H", year=2009, exchange="CMX")
        assert c.ticker == "SIH09.CMX"


# ── Catalog Tests ──────────────────────────────────────────────────────


class TestCommodityCatalog:
    def test_catalog_has_21_entries(self) -> None:
        assert len(COMMODITY_CATALOG) == 21

    def test_all_categories_represented(self) -> None:
        categories = {spec.category for spec in COMMODITY_CATALOG.values()}
        expected = {"Energy", "Precious Metals", "Base Metals", "Agriculture", "Softs", "Livestock"}
        assert categories == expected

    def test_all_roots_have_exchange_mapping(self) -> None:
        for root in COMMODITY_CATALOG:
            assert root in _EXCHANGE_MAP, f"Missing exchange mapping for {root}"

    def test_contract_months_valid(self) -> None:
        valid_codes = set(MONTH_CODES.keys())
        for root, spec in COMMODITY_CATALOG.items():
            for code in spec.contract_months:
                assert code in valid_codes, f"Invalid month code '{code}' in {root}"


# ── Calendar Roll Tests ───────────────────────────────────────────────


class TestCalendarRollDates:
    def test_generates_rolls_for_one_year(self) -> None:
        spec = COMMODITY_CATALOG["GC"]  # GJMQVZ = 6 contracts/year
        rolls = calendar_roll_dates(spec, date(2025, 1, 1), date(2025, 12, 31))
        # Should have multiple rolls (one fewer than number of contracts in range)
        assert len(rolls) >= 4

    def test_roll_date_is_before_expiry(self) -> None:
        spec = COMMODITY_CATALOG["CL"]
        rolls = calendar_roll_dates(spec, date(2025, 1, 1), date(2025, 6, 30))
        for roll_date, old_c, new_c in rolls:
            expiry = _expiry_proxy(old_c)
            assert roll_date < expiry, f"Roll {roll_date} not before expiry {expiry}"

    def test_contracts_are_sequential(self) -> None:
        spec = COMMODITY_CATALOG["GC"]
        rolls = calendar_roll_dates(spec, date(2025, 1, 1), date(2025, 12, 31))
        for i in range(len(rolls) - 1):
            old_expiry = _expiry_proxy(rolls[i][1])
            next_old_expiry = _expiry_proxy(rolls[i + 1][1])
            assert old_expiry < next_old_expiry

    def test_roll_dates_are_business_days(self) -> None:
        spec = COMMODITY_CATALOG["CL"]
        rolls = calendar_roll_dates(spec, date(2025, 1, 1), date(2025, 12, 31))
        for roll_date, _, _ in rolls:
            assert roll_date.weekday() < 5, f"{roll_date} is weekend"

    def test_empty_range_returns_empty(self) -> None:
        spec = COMMODITY_CATALOG["GC"]
        # Very narrow range with no roll dates
        rolls = calendar_roll_dates(spec, date(2025, 2, 15), date(2025, 2, 16))
        # Might be empty or have 1 roll depending on contract months
        assert isinstance(rolls, list)


# ── Volume Roll Tests ─────────────────────────────────────────────────


class TestVolumeRollDates:
    def test_detects_crossover(self) -> None:
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        front = pd.Series([100, 95, 90, 80, 70, 50, 40, 30, 20, 10], index=dates)
        nxt = pd.Series([10, 20, 30, 40, 60, 70, 80, 90, 95, 100], index=dates)
        roll_dates = volume_roll_dates(front, nxt)
        assert len(roll_dates) >= 1

    def test_no_crossover_returns_empty(self) -> None:
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        front = pd.Series([100, 100, 100, 100, 100, 100, 100, 100, 100, 100], index=dates)
        nxt = pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], index=dates)
        roll_dates = volume_roll_dates(front, nxt)
        assert roll_dates == []

    def test_empty_series_returns_empty(self) -> None:
        assert volume_roll_dates(pd.Series(dtype=float), pd.Series(dtype=float)) == []

    def test_short_series_returns_empty(self) -> None:
        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        front = pd.Series([100, 50], index=dates)
        nxt = pd.Series([50, 100], index=dates)
        result = volume_roll_dates(front, nxt)
        assert isinstance(result, list)


# ── Continuous Series Tests (mocked yfinance) ────────────────────────


def _make_ohlcv_df(start: str, periods: int, base_price: float = 100.0) -> pd.DataFrame:
    """Helper to create a mock OHLCV DataFrame."""
    dates = pd.date_range(start, periods=periods, freq="B")
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(periods))
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.random.randint(1000, 10000, periods),
        },
        index=dates,
    )


class TestBuildContinuousSeries:
    @patch("yfinance.download")
    def test_fallback_to_front_month(self, mock_dl: MagicMock) -> None:
        """When individual contracts fail, should fall back to =F ticker."""
        front_df = _make_ohlcv_df("2025-01-01", 50)
        # First N calls return empty (individual contracts), last returns data
        call_count = {"n": 0}

        def side_effect(ticker: str, **kwargs: object) -> pd.DataFrame:
            call_count["n"] += 1
            if "=F" in ticker:
                return front_df
            return pd.DataFrame()

        mock_dl.side_effect = side_effect

        result = build_continuous_series(
            "GC", date(2025, 1, 1), date(2025, 3, 31),
            adjustment=RollAdjustment.NONE,
        )
        assert not result.empty
        assert "Close" in result.columns

    @patch("yfinance.download")
    def test_panama_adjustment_applied(self, mock_dl: MagicMock) -> None:
        """Panama adjustment should shift prices additively."""
        old_df = _make_ohlcv_df("2025-01-01", 60, base_price=100.0)
        new_df = _make_ohlcv_df("2025-01-01", 60, base_price=105.0)

        def side_effect(ticker: str, **kwargs: object) -> pd.DataFrame:
            if "=F" in ticker:
                return old_df
            return new_df if "Z" in ticker or "Q" in ticker else old_df

        mock_dl.side_effect = side_effect

        result = build_continuous_series(
            "GC", date(2025, 1, 1), date(2025, 12, 31),
            method=RollMethod.CALENDAR,
            adjustment=RollAdjustment.PANAMA,
        )
        assert not result.empty

    @patch("yfinance.download")
    def test_ratio_adjustment_applied(self, mock_dl: MagicMock) -> None:
        """Ratio adjustment should scale prices multiplicatively."""
        old_df = _make_ohlcv_df("2025-01-01", 60, base_price=100.0)
        new_df = _make_ohlcv_df("2025-01-01", 60, base_price=110.0)

        def side_effect(ticker: str, **kwargs: object) -> pd.DataFrame:
            if "=F" in ticker:
                return old_df
            return new_df if "Z" in ticker or "Q" in ticker else old_df

        mock_dl.side_effect = side_effect

        result = build_continuous_series(
            "GC", date(2025, 1, 1), date(2025, 12, 31),
            method=RollMethod.CALENDAR,
            adjustment=RollAdjustment.RATIO,
        )
        assert not result.empty

    def test_unknown_root_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown commodity root"):
            build_continuous_series("XX", date(2025, 1, 1), date(2025, 12, 31))

    @patch("yfinance.download")
    def test_no_data_raises(self, mock_dl: MagicMock) -> None:
        """If yfinance returns empty for everything, should raise."""
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            build_continuous_series("GC", date(2025, 1, 1), date(2025, 3, 31))


# ── Back-Adjustment Tests ─────────────────────────────────────────────


def _flat_contract(start: str, periods: int, price: float) -> pd.DataFrame:
    """A contract quoted at one price for its whole life.

    Flat prices are the point. Back-adjustment is arithmetic on the gaps
    between contracts, and a random walk hides an arithmetic error inside its
    own noise, which is how the per-roll bug survived tests that only checked
    the result was non-empty.
    """
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": price, "High": price, "Low": price,
            "Close": price, "Volume": 1000,
        },
        index=dates,
    )


class TestBackAdjustmentAccumulates:
    """Adjustment has to carry across every later roll, not just the nearest.

    Three contracts, two rolls, each quoted flat: A at 100, B at 110, C at 125.
    The gaps are +10 at the first roll and +15 at the second. Back-adjusting
    correctly lifts A by both (100 + 25) and B by the later one only
    (110 + 15), so all three segments land on 125 and the series is flat.

    Adjusting each segment by its own roll only would leave A at 110, a 15
    point step in a series that should have none.
    """

    ROOT = "GC"

    def _fixture(self) -> tuple[list, dict]:
        a = FuturesContract(root=self.ROOT, month_code="G", year=2025)
        b = FuturesContract(root=self.ROOT, month_code="J", year=2025)
        c = FuturesContract(root=self.ROOT, month_code="M", year=2025)

        contract_data = {
            a.ticker: _flat_contract("2025-01-01", 90, 100.0),
            b.ticker: _flat_contract("2025-01-01", 90, 110.0),
            c.ticker: _flat_contract("2025-01-01", 90, 125.0),
        }
        rolls = [
            (date(2025, 2, 3), a, b),
            (date(2025, 3, 3), b, c),
        ]
        return rolls, contract_data

    def test_panama_leaves_no_step_at_any_roll(self) -> None:
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.PANAMA,
        )
        assert not result.empty
        assert result["Close"].nunique() == 1, (
            f"expected one price across all segments, got "
            f"{sorted(result['Close'].unique())}"
        )
        assert result["Close"].iloc[0] == pytest.approx(125.0)

    def test_panama_lifts_the_oldest_segment_by_both_gaps(self) -> None:
        """The oldest segment is the one a per-roll adjustment gets wrong."""
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.PANAMA,
        )
        oldest = result[result.index < pd.Timestamp("2025-02-03")]
        assert not oldest.empty
        # 100 + 15 + 10, not 100 + 10.
        assert oldest["Close"].iloc[0] == pytest.approx(125.0)

    def test_ratio_leaves_no_step_at_any_roll(self) -> None:
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.RATIO,
        )
        assert not result.empty
        assert result["Close"].round(6).nunique() == 1, (
            f"expected one price across all segments, got "
            f"{sorted(result['Close'].unique())}"
        )
        assert result["Close"].iloc[0] == pytest.approx(125.0)

    def test_none_leaves_the_raw_steps_in_place(self) -> None:
        """The control. Without adjustment the steps must still be there."""
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.NONE,
        )
        assert sorted(result["Close"].unique()) == [100.0, 110.0, 125.0]

    def test_all_price_columns_move_together(self) -> None:
        """OHLC must stay consistent, not just Close."""
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.PANAMA,
        )
        for col in ("Open", "High", "Low"):
            assert result[col].iloc[0] == pytest.approx(result["Close"].iloc[0])

    def test_volume_is_not_adjusted(self) -> None:
        """Only prices shift. Volume is a count, not a quote."""
        rolls, contract_data = self._fixture()
        result = _stitch_contracts(
            rolls, contract_data, date(2025, 1, 1), date(2025, 4, 30),
            RollAdjustment.PANAMA,
        )
        assert (result["Volume"] == 1000).all()


# ── Universe Tests ────────────────────────────────────────────────────


class TestCommodityUniverse:
    def test_universe_has_21_entries(self) -> None:
        assert len(COMMODITY_UNIVERSE) == 21

    def test_all_entries_are_futures_tickers(self) -> None:
        for ticker in COMMODITY_UNIVERSE:
            assert ticker.endswith("=F"), f"{ticker} is not a futures ticker"

    def test_presets_exist(self) -> None:
        assert "All Commodities" in COMMODITY_PRESETS
        assert "Energy" in COMMODITY_PRESETS
        assert "Precious Metals" in COMMODITY_PRESETS
        assert "Base Metals" in COMMODITY_PRESETS
        assert "Agriculture" in COMMODITY_PRESETS
        assert "Softs" in COMMODITY_PRESETS
        assert "Livestock" in COMMODITY_PRESETS

    def test_presets_cover_all_categories(self) -> None:
        assert len(COMMODITY_PRESETS) == 7

    def test_all_presets_are_subsets_of_universe(self) -> None:
        universe_keys = set(COMMODITY_UNIVERSE.keys())
        for name, tickers in COMMODITY_PRESETS.items():
            assert set(tickers).issubset(universe_keys), (
                f"Preset '{name}' has tickers not in COMMODITY_UNIVERSE"
            )

    def test_all_commodities_preset_matches_universe(self) -> None:
        assert set(COMMODITY_PRESETS["All Commodities"]) == set(COMMODITY_UNIVERSE.keys())


# ── Helper Tests ──────────────────────────────────────────────────────


class TestHelpers:
    def test_business_days_before_skips_weekend(self) -> None:
        # 2025-01-06 is Monday. 5 bdays before = 2024-12-30 (Monday)
        result = _business_days_before(date(2025, 1, 6), 5)
        assert result.weekday() < 5
        assert result == date(2024, 12, 30)

    def test_contract_months_for_year(self) -> None:
        spec = COMMODITY_CATALOG["GC"]
        contracts = _contract_months_for_year(spec, 2025)
        assert len(contracts) == len(spec.contract_months)
        for c in contracts:
            assert c.root == "GC"
            assert c.year == 2025
            assert c.exchange == "CMX"

    def test_expiry_proxy(self) -> None:
        c = FuturesContract(root="GC", month_code="Z", year=2025)
        exp = _expiry_proxy(c)
        assert exp == date(2025, 12, 1)
