"""Tests for strict model-output parsing."""

from __future__ import annotations

import itertools
import re

import pytest

from affective_contagion.messages import (
    COGNITION_OUTPUT_REGEX,
    DECISION_OUTPUT_REGEX,
    MAX_DECISION_MESSAGE_CHARS,
    DecisionParseError,
    fallback_hold,
    parse_agent_decision,
)


def test_parse_decision_with_surrounding_text() -> None:
    """A single complete object may be extracted from a model preamble."""
    raw = 'Result: {"message":"Reduce exposure.","action":"sell","size":0.25} trailing'
    result = parse_agent_decision(raw)
    assert result.message == "Reduce exposure."
    assert result.action == "SELL"
    assert result.size == 0.25


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"message":"x","action":"RUN","size":0.2}',
        '{"message":"x","action":"HOLD","size":true}',
        '{"message":"x","action":"BUY","size":1.2}',
        '{"message":"","action":"HOLD","size":0}',
    ],
)
def test_invalid_decision_is_not_silently_coerced(raw: str) -> None:
    """Parser health must remain observable rather than becoming hidden HOLD actions."""
    with pytest.raises(DecisionParseError):
        parse_agent_decision(raw)


def test_hold_fallback_is_explicit() -> None:
    """The caller can apply a separately logged fallback policy."""
    result = fallback_hold("malformed output", max_message_length=9)
    assert result.message == "malformed"
    assert result.action == "HOLD"
    assert result.size == 0


def test_decision_grammar_accepts_only_parseable_decisions() -> None:
    """The constrained-decoding grammar and the strict parser must agree.

    If the grammar admitted a string the parser rejects, constrained runs would still produce
    treatment-correlated HOLD fallbacks, which is the artifact the grammar exists to remove.
    """
    pattern = re.compile(f"^{DECISION_OUTPUT_REGEX}$")
    accepted = (
        '{"message":"Deposit outflow reported.","action":"HOLD","size":0}',
        '{"message":"Cutting exposure now.","action":"SELL","size":0.25}',
        '{"message":"Adding on weakness.","action":"BUY","size":1.0}',
        '{"message":"Full size.","action":"BUY","size":1}',
    )
    for text in accepted:
        assert pattern.match(text), text
        parsed = parse_agent_decision(text)
        assert 0 <= parsed.size <= 1

    # Malformations observed without constrained decoding, plus out-of-range sizes that JSON
    # Schema backends fail to exclude.
    rejected = (
        '{"message":"Mixed sentiment.","action":"SELL|0.15","size":0.85}',
        '{"message":"Market down.","action":"SELL","size:0.15"}',
        '{"message":"Oversized.","action":"BUY","size":5}',
        '{"message":"Negative.","action":"SELL","size":-1}',
        '{"message":"Truncated.","action":"HOLD","size":0.5',
    )
    for text in rejected:
        assert not pattern.match(text), text


def test_decision_grammar_bounds_message_length() -> None:
    """The grammar must cap message length so decoding cannot run away."""
    pattern = re.compile(f"^{DECISION_OUTPUT_REGEX}$")
    too_long = "x" * (MAX_DECISION_MESSAGE_CHARS + 1)
    assert not pattern.match(f'{{"message":"{too_long}","action":"HOLD","size":0}}')
    at_limit = "x" * MAX_DECISION_MESSAGE_CHARS
    assert pattern.match(f'{{"message":"{at_limit}","action":"HOLD","size":0}}')


def test_everything_the_cognition_grammar_admits_also_parses() -> None:
    """Everything the grammar admits must parse across boundary and generated cases.

    Under constrained decoding a model can only emit strings the grammar allows, so any admitted
    string that fails to parse becomes a `fallback_hold` and creates a treatment-correlated action
    artifact.

    The reverse direction is deliberately not required. The parser accepting more precision than the
    grammar allows (four decimals, say) is unreachable under constrained decoding and harmless in
    free-form runs, and closing it would mean parsing raw text rather than floats.
    """
    percents = [
        "0",
        "-0",
        "00",
        "01",
        "9",
        "99",
        "100",
        "-100",
        "99.9",
        "99.99",
        "-99.99",
        "100.0",
        "9.999",
        "+5",
        "5.",
        ".5",
        "1e2",
        "-9.05",
    ]
    units = ["0", "1", "0.5", "1.0", "1.000", "1.0001", "0.9999", "-0.1", "2", "00.5", "+0.5"]
    unsafe = []
    for percent, unit in itertools.product(percents, units):
        raw = (
            f'{{"message":"x","action":"HOLD","size":0,"expected_return":{percent},'
            f'"confidence":{unit},"downside_risk":0.5}}'
        )
        if not re.fullmatch(COGNITION_OUTPUT_REGEX, raw):
            continue
        try:
            parse_agent_decision(raw, require_cognition=True)
        except DecisionParseError:
            unsafe.append((percent, unit))
    assert unsafe == [], f"grammar admits strings the parser rejects: {unsafe}"


def test_the_cognition_fields_round_trip() -> None:
    """A well-formed decision must yield exactly the three RQ2 outcomes it stated."""
    raw = (
        '{"message":"Deposit outflow is accelerating.","action":"SELL","size":0.5,'
        '"expected_return":-6.5,"confidence":0.7,"downside_risk":0.8}'
    )
    assert re.fullmatch(COGNITION_OUTPUT_REGEX, raw)
    decision = parse_agent_decision(raw, require_cognition=True)
    actual = (decision.expected_return, decision.confidence, decision.downside_risk)
    assert actual == (-6.5, 0.7, 0.8)
    assert decision.has_cognition


def test_the_old_decision_format_is_unaffected() -> None:
    """Records without cognition fields must continue to parse identically."""
    old = '{"message":"No signal.","action":"HOLD","size":0}'
    decision = parse_agent_decision(old)
    assert decision.action == "HOLD"
    assert decision.has_cognition is False
    assert decision.expected_return is None
    assert re.fullmatch(DECISION_OUTPUT_REGEX, old)


def test_requiring_cognition_rejects_a_decision_that_omits_it() -> None:
    """A missing outcome must be a visible parse failure, not a silently absent measurement."""
    with pytest.raises(DecisionParseError, match="expected_return must be numeric"):
        parse_agent_decision(
            '{"message":"No signal.","action":"HOLD","size":0}', require_cognition=True
        )
