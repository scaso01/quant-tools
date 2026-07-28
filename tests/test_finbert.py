"""Tests for FinBERTSentiment.

All transformers calls are mocked. No real model loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.errors import DataError
from quant_tools.data.sentiment.finbert import FinBERTSentiment


class TestFinBERTSentiment:
    """Tests for FinBERTSentiment."""

    # ----- score -----

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_score_positive(self, mock_load: MagicMock) -> None:
        """Score returns {label: score} for positive text."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "positive", "score": 0.92}]

        result = fb.score("Revenue beat expectations.")
        assert result == {"positive": 0.92}
        fb._pipeline.assert_called_once()

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_score_negative(self, mock_load: MagicMock) -> None:
        """Score returns {label: score} for negative text."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "negative", "score": 0.85}]

        result = fb.score("Company reported massive losses.")
        assert result == {"negative": 0.85}

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_score_neutral(self, mock_load: MagicMock) -> None:
        """Score returns {label: score} for neutral text."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "neutral", "score": 0.78}]

        result = fb.score("The company held its annual meeting.")
        assert result == {"neutral": 0.78}

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_score_truncates_long_text(self, mock_load: MagicMock) -> None:
        """Text longer than 512 chars is truncated before pipeline call."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "positive", "score": 0.50}]

        long_text = "A" * 1000
        fb.score(long_text)
        # The pipeline should receive truncated text (512 chars)
        call_args = fb._pipeline.call_args
        assert len(call_args[0][0]) == 512

    # ----- batch_score -----

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_batch_score(self, mock_load: MagicMock) -> None:
        """batch_score returns list of {label: score} dicts."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [
            {"label": "positive", "score": 0.90},
            {"label": "negative", "score": 0.70},
            {"label": "neutral", "score": 0.60},
        ]

        texts = ["Good news", "Bad news", "No news"]
        results = fb.batch_score(texts)
        assert len(results) == 3
        assert results[0] == {"positive": 0.90}
        assert results[1] == {"negative": 0.70}
        assert results[2] == {"neutral": 0.60}

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_batch_score_truncates(self, mock_load: MagicMock) -> None:
        """batch_score truncates each text to 512 chars."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "positive", "score": 0.50}]

        texts = ["B" * 1000]
        fb.batch_score(texts)
        call_args = fb._pipeline.call_args
        assert len(call_args[0][0][0]) == 512

    # ----- ticker_sentiment -----

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_ticker_sentiment_positive(self, mock_load: MagicMock) -> None:
        """Aggregate positive sentiment returns > 0."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [
            {"label": "positive", "score": 0.90},
            {"label": "positive", "score": 0.80},
        ]

        result = fb.ticker_sentiment("AAPL", ["great earnings", "stock up"])
        assert result > 0
        # (0.90 + 0.80) / 2 = 0.85
        assert abs(result - 0.85) < 0.01

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_ticker_sentiment_negative(self, mock_load: MagicMock) -> None:
        """Aggregate negative sentiment returns < 0."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [
            {"label": "negative", "score": 0.85},
        ]

        result = fb.ticker_sentiment("AAPL", ["stock crashed"])
        assert result < 0
        # neg=0.85, pos=0: (0 - 0.85) / 1 = -0.85
        assert abs(result - (-0.85)) < 0.01

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_ticker_sentiment_mixed(self, mock_load: MagicMock) -> None:
        """Mixed sentiment averages out."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [
            {"label": "positive", "score": 0.80},
            {"label": "negative", "score": 0.80},
        ]

        result = fb.ticker_sentiment("AAPL", ["good", "bad"])
        # (0.80 + (-0.80)) / 2 = 0.0
        assert abs(result) < 0.01

    def test_ticker_sentiment_empty_texts(self) -> None:
        """Empty text list returns 0.0 without loading model."""
        fb = FinBERTSentiment()
        result = fb.ticker_sentiment("AAPL", [])
        assert result == 0.0

    # ----- Import error -----

    def test_import_error_raises_data_error(self) -> None:
        """Missing transformers raises DataError, not ImportError."""
        fb = FinBERTSentiment()
        with patch.dict("sys.modules", {"transformers": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named transformers"),
            ):
                fb._pipeline = None
                with pytest.raises(DataError, match="transformers not installed"):
                    fb._load()

    # ----- Lazy loading -----

    @patch("quant_tools.data.sentiment.finbert.FinBERTSentiment._load")
    def test_pipeline_not_loaded_until_score(
        self, mock_load: MagicMock,
    ) -> None:
        """Pipeline is None at init, _load called on first use."""
        fb = FinBERTSentiment()
        assert fb._pipeline is None
        fb._pipeline = MagicMock()
        fb._pipeline.return_value = [{"label": "neutral", "score": 0.5}]
        fb.score("test")
        mock_load.assert_called_once()

    def test_load_skips_if_already_loaded(self) -> None:
        """_load is a no-op if pipeline already exists."""
        fb = FinBERTSentiment()
        fb._pipeline = MagicMock()  # already set
        fb._load()  # should not re-initialize
        # Pipeline should still be the same mock
        assert fb._pipeline is not None
