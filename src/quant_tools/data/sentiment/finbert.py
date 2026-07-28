"""Local FinBERT model for financial sentiment analysis.

Uses the ProsusAI/finbert model via Hugging Face ``transformers`` pipeline.
The model is lazy-loaded on first call so import time stays near zero.
"""

from __future__ import annotations

import logging

from quant_tools.errors import DataError

log = logging.getLogger("quant_tools")


class FinBERTSentiment:
    """Financial sentiment analysis using FinBERT.

    Requires ``transformers`` and ``torch`` (optional dep ``[nlp]``).
    The pipeline is loaded lazily on first use.
    """

    def __init__(self) -> None:
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the FinBERT pipeline (CPU only)."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # CPU
            )
        except ImportError:
            raise DataError(
                "transformers not installed. pip install quant-tools[nlp]"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> dict[str, float]:
        """Score a single text.

        Args:
            text: Input text (truncated to 512 chars for model limit).

        Returns:
            ``{label: score}`` where label is ``positive``, ``negative``,
            or ``neutral``.
        """
        self._load()
        result = self._pipeline(text[:512], truncation=True)[0]
        return {result["label"]: result["score"]}

    def batch_score(self, texts: list[str]) -> list[dict[str, float]]:
        """Score multiple texts efficiently.

        Args:
            texts: List of input texts.

        Returns:
            List of ``{label: score}`` dicts.
        """
        self._load()
        results = self._pipeline(
            [t[:512] for t in texts],
            truncation=True,
            batch_size=16,
        )
        return [{r["label"]: r["score"]} for r in results]

    def ticker_sentiment(self, ticker: str, texts: list[str]) -> float:
        """Aggregate sentiment score for ticker mentions.

        Args:
            ticker: Ticker symbol (for logging).
            texts: List of texts mentioning the ticker.

        Returns:
            Aggregate score from ``-1.0`` (very negative) to ``1.0``
            (very positive).  Returns ``0.0`` for empty input.
        """
        if not texts:
            return 0.0

        scores = self.batch_score(texts)
        total = 0.0
        for s in scores:
            pos = s.get("positive", 0.0)
            neg = s.get("negative", 0.0)
            total += pos - neg
        return total / len(scores)
