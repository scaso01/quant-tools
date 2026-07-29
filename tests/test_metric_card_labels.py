"""Regression tests for metric card labelling.

``render_metric_card`` used to substitute the glossary key for the caller's
label whenever ``glossary_key`` was set, so the backtest page rendered two
different cards, Total Return and CAGR, both reading "CAGR". The glossary key
is documentation, not the label.
"""

from __future__ import annotations

from unittest.mock import patch

from quant_tools.dashboard.core.helpers import render_metric_card
from quant_tools.dashboard.core.page_help import GLOSSARY, jargon


class TestJargon:
    """The glossary helper decorates a label, it does not replace it."""

    def test_defaults_to_the_term(self) -> None:
        assert ">Sharpe Ratio</abbr>" in jargon("Sharpe Ratio")

    def test_shows_the_supplied_text_not_the_key(self) -> None:
        html = jargon("CAGR", "Total Return")
        assert ">Total Return</abbr>" in html
        assert ">CAGR</abbr>" not in html

    def test_definition_comes_from_the_key(self) -> None:
        html = jargon("CAGR", "Total Return")
        assert GLOSSARY["CAGR"][:20].replace('"', "&quot;") in html

    def test_unknown_key_falls_back_to_plain_text(self) -> None:
        assert jargon("NotAGlossaryTerm", "Some Label") == "Some Label"
        assert jargon("NotAGlossaryTerm") == "NotAGlossaryTerm"

    def test_quotes_in_a_definition_are_escaped(self) -> None:
        with patch.dict(GLOSSARY, {"Quoted": 'He said "hi"'}, clear=False):
            assert 'title="He said &quot;hi&quot;"' in jargon("Quoted")


class TestRenderMetricCard:
    """The card renders the label it was given."""

    def _render(self, **kwargs: object) -> str:
        with patch("quant_tools.dashboard.core.helpers.st") as mock_st:
            render_metric_card(**kwargs)  # type: ignore[arg-type]
            return mock_st.markdown.call_args[0][0]

    def test_label_survives_a_differing_glossary_key(self) -> None:
        html = self._render(
            label="Total Return", value="36.42%", glossary_key="CAGR"
        )
        assert "Total Return" in html
        assert ">CAGR</abbr>" not in html

    def test_matching_label_and_key_still_render_once(self) -> None:
        html = self._render(label="CAGR", value="7.94%", glossary_key="CAGR")
        assert html.count(">CAGR</abbr>") == 1

    def test_no_glossary_key_renders_a_bare_label(self) -> None:
        html = self._render(label="Total Trades", value="1")
        assert "Total Trades" in html
        assert "<abbr" not in html

    def test_the_two_backtest_cards_get_distinct_labels(self) -> None:
        """The exact pair that surfaced the bug."""
        total = self._render(
            label="Total Return", value="36.42%", glossary_key="CAGR"
        )
        cagr = self._render(label="CAGR", value="7.94%", glossary_key="CAGR")
        assert "Total Return" in total
        assert "Total Return" not in cagr
