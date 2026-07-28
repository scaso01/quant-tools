"""Tests for predict CLI commands."""
from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from quant_tools.models import (
    AssetClass,
    ConsensusSignal,
    EnsemblePrediction,
    ModelPerformance,
    PredictionDirection,
    PredictionOutcome,
    PredictionResult,
)
from quant_tools.predict.cli import (
    build_predict_parser,
    build_resolve_parser,
    build_track_parser,
    cmd_predict,
    cmd_resolve,
    cmd_track,
)


# ── Parser Tests ────────────────────────────────────────────────────


class TestBuildPredictParser:
    def test_symbol_required(self):
        parser = build_predict_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_symbol_parsed(self):
        parser = build_predict_parser()
        args = parser.parse_args(["AAPL"])
        assert args.symbol == "AAPL"

    def test_default_horizon(self):
        parser = build_predict_parser()
        args = parser.parse_args(["AAPL"])
        # Default is None at parser level; cmd_predict applies default=5
        assert args.horizon is None

    def test_custom_horizon(self):
        parser = build_predict_parser()
        args = parser.parse_args(["AAPL", "--horizon", "20"])
        assert args.horizon == 20

    def test_default_asset_class_is_none(self):
        parser = build_predict_parser()
        args = parser.parse_args(["BTC/USDT"])
        assert args.asset_class is None

    def test_valid_asset_class_equity(self):
        parser = build_predict_parser()
        args = parser.parse_args(["AAPL", "--asset-class", "equity"])
        assert args.asset_class == "equity"

    def test_valid_asset_class_crypto(self):
        parser = build_predict_parser()
        args = parser.parse_args(["BTC/USDT", "--asset-class", "crypto"])
        assert args.asset_class == "crypto"

    def test_valid_asset_class_forex(self):
        parser = build_predict_parser()
        args = parser.parse_args(["EUR/USD", "--asset-class", "forex"])
        assert args.asset_class == "forex"

    def test_valid_asset_class_commodity(self):
        parser = build_predict_parser()
        args = parser.parse_args(["GC=F", "--asset-class", "commodity"])
        assert args.asset_class == "commodity"

    def test_invalid_asset_class_rejected(self):
        parser = build_predict_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["AAPL", "--asset-class", "bonds"])

    def test_prog_name(self):
        parser = build_predict_parser()
        assert parser.prog == "quant-tools predict"


class TestBuildTrackParser:
    def test_no_args_required(self):
        parser = build_track_parser()
        args = parser.parse_args([])
        assert args.symbol is None
        assert args.model is None

    def test_symbol_filter(self):
        parser = build_track_parser()
        args = parser.parse_args(["--symbol", "AAPL"])
        assert args.symbol == "AAPL"

    def test_model_filter(self):
        parser = build_track_parser()
        args = parser.parse_args(["--model", "gradient_boost"])
        assert args.model == "gradient_boost"

    def test_asset_class_filter(self):
        parser = build_track_parser()
        args = parser.parse_args(["--asset-class", "equity"])
        assert args.asset_class == "equity"

    def test_default_limit(self):
        parser = build_track_parser()
        args = parser.parse_args([])
        assert args.limit == 20

    def test_custom_limit(self):
        parser = build_track_parser()
        args = parser.parse_args(["--limit", "50"])
        assert args.limit == 50

    def test_prog_name(self):
        parser = build_track_parser()
        assert parser.prog == "quant-tools track"


class TestBuildResolveParser:
    def test_no_args_required(self):
        parser = build_resolve_parser()
        args = parser.parse_args([])
        assert args is not None

    def test_prog_name(self):
        parser = build_resolve_parser()
        assert parser.prog == "quant-tools resolve"


# ── Helpers ─────────────────────────────────────────────────────────


def _make_ensemble_prediction(**overrides) -> EnsemblePrediction:
    """Create a test EnsemblePrediction."""
    defaults = dict(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        consensus=ConsensusSignal.BUY,
        confidence=0.72,
        direction=PredictionDirection.UP,
        magnitude=0.035,
        horizon_days=5,
        target_date="2026-03-21",
        model_predictions=[
            PredictionResult(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                model_name="gradient_boost",
                direction=PredictionDirection.UP,
                magnitude=0.04,
                confidence=0.8,
                horizon_days=5,
                target_date="2026-03-21",
            ),
            PredictionResult(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                model_name="stats_forecast",
                direction=PredictionDirection.UP,
                magnitude=0.03,
                confidence=0.65,
                horizon_days=5,
                target_date="2026-03-21",
            ),
        ],
    )
    defaults.update(overrides)
    return EnsemblePrediction(**defaults)


def _make_prediction_dict(**overrides) -> dict:
    """Create a test prediction dict (as returned by tracker.get_predictions)."""
    defaults = dict(
        id=1,
        timestamp=datetime.now(),
        symbol="AAPL",
        asset_class="equity",
        model_name="gradient_boost",
        prediction_direction="up",
        predicted_magnitude=0.04,
        confidence=0.8,
        horizon_days=5,
        target_date="2026-03-21",
        consensus_signal="buy",
        is_ensemble=0,
        resolved=0,
    )
    defaults.update(overrides)
    return defaults


# ── cmd_predict Tests ───────────────────────────────────────────────


class TestCmdPredict:
    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_success_with_auto_detect(self, mock_reg_cls, mock_ens_cls, mock_trk_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction()

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 42

        result = cmd_predict(["AAPL"])

        assert result == 0
        mock_reg.detect_asset_class.assert_called_once_with("AAPL")
        mock_reg.fetch_ohlcv.assert_called_once_with("AAPL", period="2y")
        out = capsys.readouterr().out
        assert "PREDICTION: AAPL" in out
        assert "BUY" in out
        assert "#42" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_success_with_explicit_asset_class(self, mock_reg_cls, mock_ens_cls, mock_trk_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction(asset_class=AssetClass.CRYPTO)

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 7

        result = cmd_predict(["BTC/USDT", "--asset-class", "crypto"])

        assert result == 0
        mock_reg.detect_asset_class.assert_not_called()
        mock_ens.predict.assert_called_once()
        call_args = mock_ens.predict.call_args
        assert call_args[0][2] == AssetClass.CRYPTO

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_custom_horizon_passed(self, mock_reg_cls, mock_ens_cls, mock_trk_cls):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction(horizon_days=20)

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        result = cmd_predict(["AAPL", "--horizon", "20"])

        assert result == 0
        call_args = mock_ens.predict.call_args
        assert call_args[0][3] == 20

    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_no_data_returns_1(self, mock_reg_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_reg.fetch_ohlcv.return_value = None

        result = cmd_predict(["AAPL"])

        assert result == 1
        out = capsys.readouterr().out
        assert "No data available" in out

    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_empty_df_returns_1(self, mock_reg_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = True
        mock_reg.fetch_ohlcv.return_value = mock_df

        result = cmd_predict(["AAPL"])

        assert result == 1

    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_ensemble_exception_returns_1(self, mock_reg_cls, mock_ens_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.side_effect = RuntimeError("All models failed")

        result = cmd_predict(["AAPL"])

        assert result == 1
        out = capsys.readouterr().out
        assert "Error:" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_output_shows_model_breakdown(self, mock_reg_cls, mock_ens_cls, mock_trk_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction()

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        cmd_predict(["AAPL"])
        out = capsys.readouterr().out

        assert "Model Breakdown" in out
        assert "gradient_boost" in out
        assert "stats_forecast" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_output_with_none_magnitude(self, mock_reg_cls, mock_ens_cls, mock_trk_cls, capsys):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        pred = _make_ensemble_prediction(magnitude=None)
        pred.model_predictions[0].magnitude = None
        pred.model_predictions[1].magnitude = None

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = pred

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        result = cmd_predict(["AAPL"])
        assert result == 0
        out = capsys.readouterr().out
        assert "N/A" in out
        assert "Magnitude" not in out.split("PREDICTION")[1].split("Model Breakdown")[0]


# ── cmd_track Tests ─────────────────────────────────────────────────


class TestCmdTrack:
    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_with_predictions(self, mock_trk_cls, capsys):
        mock_trk = mock_trk_cls.return_value
        mock_trk.get_predictions.return_value = [
            _make_prediction_dict(),
            _make_prediction_dict(id=2, symbol="MSFT", resolved=1),
        ]
        mock_trk.get_performance.return_value = []

        result = cmd_track([])

        assert result == 0
        out = capsys.readouterr().out
        assert "Recent Predictions (2 shown)" in out
        assert "AAPL" in out
        assert "MSFT" in out
        assert "Pending" in out
        assert "Resolved" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_no_predictions(self, mock_trk_cls, capsys):
        mock_trk = mock_trk_cls.return_value
        mock_trk.get_predictions.return_value = []
        mock_trk.get_performance.return_value = []

        result = cmd_track([])

        assert result == 0
        out = capsys.readouterr().out
        assert "No predictions found" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_with_performance(self, mock_trk_cls, capsys):
        mock_trk = mock_trk_cls.return_value
        mock_trk.get_predictions.return_value = []
        mock_trk.get_performance.return_value = [
            ModelPerformance(
                model_name="gradient_boost",
                asset_class=AssetClass.EQUITY,
                total_predictions=100,
                wins=65,
                losses=35,
                win_rate=0.65,
                avg_confidence=0.72,
            ),
        ]

        result = cmd_track([])

        assert result == 0
        out = capsys.readouterr().out
        assert "Model Performance" in out
        assert "gradient_boost" in out
        assert "65" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_filters_passed_to_tracker(self, mock_trk_cls):
        mock_trk = mock_trk_cls.return_value
        mock_trk.get_predictions.return_value = []
        mock_trk.get_performance.return_value = []

        cmd_track(["--symbol", "AAPL", "--model", "garch", "--asset-class", "equity", "--limit", "10"])

        mock_trk.get_predictions.assert_called_once_with(
            symbol="AAPL", model="garch", asset_class="equity", limit=10,
        )

    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_exception_returns_1(self, mock_trk_cls, capsys):
        mock_trk_cls.side_effect = RuntimeError("DB error")

        result = cmd_track([])

        assert result == 1
        out = capsys.readouterr().out
        assert "Error:" in out


# ── cmd_resolve Tests ───────────────────────────────────────────────


class TestCmdResolve:
    @patch("quant_tools.predict.resolver.PredictionResolver")
    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_with_outcomes(self, mock_trk_cls, mock_res_cls, capsys):
        mock_res = mock_res_cls.return_value
        mock_res.resolve_all_pending.return_value = [
            PredictionOutcome(
                prediction_id=1,
                actual_direction=PredictionDirection.UP,
                actual_magnitude=0.03,
                correct=True,
                error_pct=0.01,
            ),
            PredictionOutcome(
                prediction_id=2,
                actual_direction=PredictionDirection.DOWN,
                actual_magnitude=-0.02,
                correct=False,
                error_pct=0.06,
            ),
        ]

        result = cmd_resolve([])

        assert result == 0
        out = capsys.readouterr().out
        assert "Resolved 2 predictions" in out
        assert "CORRECT" in out
        assert "WRONG" in out

    @patch("quant_tools.predict.resolver.PredictionResolver")
    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_no_pending(self, mock_trk_cls, mock_res_cls, capsys):
        mock_res = mock_res_cls.return_value
        mock_res.resolve_all_pending.return_value = []

        result = cmd_resolve([])

        assert result == 0
        out = capsys.readouterr().out
        assert "No pending predictions" in out

    @patch("quant_tools.predict.tracker.PredictionTracker")
    def test_exception_returns_1(self, mock_trk_cls, capsys):
        mock_trk_cls.side_effect = RuntimeError("DB error")

        result = cmd_resolve([])

        assert result == 1
        out = capsys.readouterr().out
        assert "Error:" in out


# ── CLI Dispatch Tests ──────────────────────────────────────────────


class TestCLIDispatch:
    @patch("quant_tools.predict.cli.cmd_predict")
    def test_dispatch_predict(self, mock_cmd):
        from quant_tools.cli import _cmd_predict
        _cmd_predict(["AAPL", "--horizon", "10"])
        mock_cmd.assert_called_once_with(["AAPL", "--horizon", "10"])

    @patch("quant_tools.predict.cli.cmd_track")
    def test_dispatch_track(self, mock_cmd):
        from quant_tools.cli import _cmd_track
        _cmd_track(["--symbol", "AAPL"])
        mock_cmd.assert_called_once_with(["--symbol", "AAPL"])

    @patch("quant_tools.predict.cli.cmd_resolve")
    def test_dispatch_resolve(self, mock_cmd):
        from quant_tools.cli import _cmd_resolve
        _cmd_resolve([])
        mock_cmd.assert_called_once_with([])

    def test_help_includes_predict(self, capsys):
        from quant_tools.cli import _print_help
        _print_help()
        out = capsys.readouterr().out
        assert "predict" in out
        assert "ML ensemble prediction" in out

    def test_help_includes_track(self, capsys):
        from quant_tools.cli import _print_help
        _print_help()
        out = capsys.readouterr().out
        assert "track" in out
        assert "model performance" in out

    def test_help_includes_resolve(self, capsys):
        from quant_tools.cli import _print_help
        _print_help()
        out = capsys.readouterr().out
        assert "resolve" in out
        assert "actuals" in out

    def test_help_includes_predict_example(self, capsys):
        from quant_tools.cli import _print_help
        _print_help()
        out = capsys.readouterr().out
        assert "quant-tools predict AAPL" in out

    @patch("quant_tools.cli._cmd_predict")
    def test_main_routes_predict(self, mock_handler):
        with patch.object(sys, "argv", ["quant-tools", "predict", "AAPL"]):
            from quant_tools.cli import main
            main()
            mock_handler.assert_called_once_with(["AAPL"])

    @patch("quant_tools.cli._cmd_track")
    def test_main_routes_track(self, mock_handler):
        with patch.object(sys, "argv", ["quant-tools", "track"]):
            from quant_tools.cli import main
            main()
            mock_handler.assert_called_once_with([])

    @patch("quant_tools.cli._cmd_resolve")
    def test_main_routes_resolve(self, mock_handler):
        with patch.object(sys, "argv", ["quant-tools", "resolve"]):
            from quant_tools.cli import main
            main()
            mock_handler.assert_called_once_with([])


# ── Intraday --hours CLI Tests ─────────────────────────────────────


class TestCmdPredictHours:
    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_hours_flag_basic(self, mock_reg_cls, mock_ens_cls, mock_trk_cls):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.CRYPTO
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            interval="1h",
        )

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        result = cmd_predict(["BTC/USDT", "--hours", "4"])

        assert result == 0
        mock_ens.predict.assert_called_once()
        call_kwargs = mock_ens.predict.call_args[1]
        assert call_kwargs["hours"] == 4
        assert call_kwargs["interval"] == "1h"

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_hours_flag_12h(self, mock_reg_cls, mock_ens_cls, mock_trk_cls):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.CRYPTO
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction(
            symbol="BTC/USDT",
            asset_class=AssetClass.CRYPTO,
            interval="4h",
        )

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        result = cmd_predict(["BTC/USDT", "--hours", "12"])

        assert result == 0
        call_kwargs = mock_ens.predict.call_args[1]
        assert call_kwargs["interval"] == "4h"

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_hours_non_standard_warns(self, mock_reg_cls, mock_ens_cls, mock_trk_cls, capsys):
        # --hours 48 is non-standard for crypto but should warn, not error
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.CRYPTO
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df
        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction(asset_class=AssetClass.CRYPTO)
        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1
        result = cmd_predict(["BTC/USDT", "--hours", "48"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Warning" in out or "not a standard horizon" in out

    def test_hours_and_horizon_mutually_exclusive(self):
        parser = build_predict_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["BTC/USDT", "--hours", "4", "--horizon", "5"])

    @patch("quant_tools.predict.tracker.PredictionTracker")
    @patch("quant_tools.predict.ensemble.PredictionEnsemble")
    @patch("quant_tools.data.providers.registry.ProviderRegistry")
    def test_horizon_flag_unchanged(self, mock_reg_cls, mock_ens_cls, mock_trk_cls):
        mock_reg = mock_reg_cls.return_value
        mock_reg.detect_asset_class.return_value = AssetClass.EQUITY
        mock_df = MagicMock()
        mock_df.empty = False
        mock_reg.fetch_ohlcv.return_value = mock_df

        mock_ens = mock_ens_cls.return_value
        mock_ens.predict.return_value = _make_ensemble_prediction()

        mock_trk = mock_trk_cls.return_value
        mock_trk.store_prediction.return_value = 1

        result = cmd_predict(["AAPL", "--horizon", "5"])

        assert result == 0
        # Should call predict with positional args (not hours/interval kwargs)
        call_args = mock_ens.predict.call_args
        assert call_args[0][3] == 5  # horizon_days positional arg
        # Fetch should use period="2y" (daily mode)
        mock_reg.fetch_ohlcv.assert_called_once_with("AAPL", period="2y")
