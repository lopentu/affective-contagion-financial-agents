"""Tests for the decision-vocabulary scrub."""

from __future__ import annotations

import pytest

from affective_contagion.scrub import DECISION_VOCABULARY, scrub_decision_vocabulary


def test_removes_action_tokens_in_any_case() -> None:
    """The grammar can emit BUY/SELL/HOLD in upper case; the scrub is case-insensitive."""
    assert scrub_decision_vocabulary("I recommend SELL now") == "I recommend now"
    assert scrub_decision_vocabulary("we should hold") == "we should"


def test_removes_direction_terms_including_inflections() -> None:
    """Inflections are listed explicitly, because the VAD instrument does not lemmatise either."""
    text = "prices fell and margins are declining while volumes rose"
    assert scrub_decision_vocabulary(text) == "prices and margins are while volumes"


def test_keeps_affect_words_and_numbers() -> None:
    """The scrub removes decision vocabulary, not affect or facts."""
    text = "I am alarmed and frightened by the 18% outflow"
    assert scrub_decision_vocabulary(text) == text


def test_does_not_match_inside_longer_words() -> None:
    """Word-boundary matching: 'holding' and 'downstream' are not decision tokens here."""
    assert scrub_decision_vocabulary("holding downstream buyer") == "holding downstream buyer"


def test_risk_is_deliberately_not_scrubbed() -> None:
    """Recorded choice: 'risk' is a standing role-prompt feature, not forced by a forecast."""
    assert "risk" not in DECISION_VOCABULARY
    assert scrub_decision_vocabulary("tail risk is elevated") == "tail risk is elevated"


def test_collapses_whitespace() -> None:
    """The result stays tokenisable by the same splitter the lexicon uses."""
    assert scrub_decision_vocabulary("a sell b hold c") == "a b c"


@pytest.mark.parametrize("text", ["", "   ", "no decision words at all"])
def test_degenerate_inputs(text: str) -> None:
    """Empty and unmatched text pass through without error."""
    assert scrub_decision_vocabulary(text) == " ".join(text.split())
