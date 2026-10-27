"""Strict parsing for generated financial-agent decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Literal, cast

DecisionAction = Literal["BUY", "SELL", "HOLD"]
_VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD"})

MAX_DECISION_MESSAGE_CHARS = 400

# Grammar for constrained decoding, kept beside the parser so the two cannot drift apart.
#
# Prompt instructions alone do not reliably produce parseable decisions from small models, which
# can emit malformed values such as "SELL|0.15" and {"action":"SELL","size:0.15"}. Every failure
# becomes a HOLD at size 0 through `fallback_hold`, so a parse-failure rate that differs by arm adds
# a treatment-correlated action artifact to the measured outcome.
#
# JSON Schema is not used because the schema backends do not enforce numeric `minimum` and
# `maximum`, so schema-constrained output can contain sizes such as -5 and 5 that this parser
# rejects. A regex constrains the size range itself, so every constrained generation satisfies
# `parse_agent_decision`.
DECISION_OUTPUT_REGEX = (
    r'\{"message":"[^"\\\n]{1,' + str(MAX_DECISION_MESSAGE_CHARS) + r'}",'
    r'"action":"(BUY|SELL|HOLD)",'
    r'"size":(0(\.[0-9]{1,3})?|1(\.0{1,3})?)\}'
)

# Unit-interval and signed-percent fragments, shared so grammar and parser bound the same ranges.
# JSON Schema is not usable for this; see the note above on `minimum`/`maximum`.
_UNIT = r"(0(\.[0-9]{1,3})?|1(\.0{1,3})?)"
# No leading zeros, because "00" and "01" are not valid JSON numbers and the parser would reject
# them. The bound is at most two integer digits, i.e. magnitude below 100, since a regex cannot cap
# 99.99 at 99 without enumerating cases. The parser uses the same bound, so everything this admits
# also parses.
_PERCENT = r"-?(0|[1-9][0-9]?)(\.[0-9]{1,2})?"

# Extended grammar adding the three cognition outcomes RQ2 names. It is opt-in because it changes
# the prompt, so a run using it is not comparable with a run that does not.
COGNITION_OUTPUT_REGEX = (
    r'\{"message":"[^"\\\n]{1,' + str(MAX_DECISION_MESSAGE_CHARS) + r'}",'
    r'"action":"(BUY|SELL|HOLD)",'
    r'"size":' + _UNIT + r","
    r'"expected_return":' + _PERCENT + r","
    r'"confidence":' + _UNIT + r","
    r'"downside_risk":' + _UNIT + r"\}"
)


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """Validated agent communication and action."""

    message: str
    action: DecisionAction
    size: float
    # RQ2's cognition level: expected return in percent, confidence in [0,1], and the agent's own
    # probability of an adverse move in [0,1]. None when the run did not ask for them, which keeps
    # records without these fields parseable and comparable.
    expected_return: float | None = None
    confidence: float | None = None
    downside_risk: float | None = None

    @property
    def has_cognition(self) -> bool:
        """Report whether this decision carries the RQ2 outcomes."""
        return None not in (self.expected_return, self.confidence, self.downside_risk)


class DecisionParseError(ValueError):
    """Raised when model output does not contain one valid decision object."""


def _extract_json_object(raw: str) -> dict[str, object]:
    """Decode the first complete JSON object while allowing surrounding text."""
    start = raw.find("{")
    if start < 0:
        msg = "no JSON object found"
        raise DecisionParseError(msg)
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(raw[start:])
    except JSONDecodeError as error:
        msg = "invalid JSON decision"
        raise DecisionParseError(msg) from error
    if not isinstance(value, dict):
        msg = "decision must be a JSON object"
        raise DecisionParseError(msg)
    return cast("dict[str, object]", value)


def _unit_field(value: object, name: str) -> float:
    """Validate a [0, 1] field, refusing bools and out-of-range numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be numeric"
        raise DecisionParseError(msg)
    number = float(value)
    if not 0 <= number <= 1:
        msg = f"{name} must be within [0, 1]"
        raise DecisionParseError(msg)
    return number


def parse_agent_decision(raw: str, *, require_cognition: bool = False) -> AgentDecision:
    """Parse a decision without silently coercing invalid actions or sizes.

    ``require_cognition`` demands the three RQ2 fields and rejects a decision lacking them. Off by
    default so records without those fields still parse unchanged. A run that asks for the fields
    must also require them, or a model omitting one would silently produce a decision with a missing
    outcome rather than a parse failure the run health can see.
    """
    value = _extract_json_object(raw)
    message_value = value.get("message")
    action_value = value.get("action")
    size_value = value.get("size")

    if not isinstance(message_value, str) or not message_value.strip():
        msg = "message must be a non-empty string"
        raise DecisionParseError(msg)
    if not isinstance(action_value, str) or action_value.upper() not in _VALID_ACTIONS:
        msg = "action must be BUY, SELL, or HOLD"
        raise DecisionParseError(msg)
    if isinstance(size_value, bool) or not isinstance(size_value, int | float):
        msg = "size must be numeric"
        raise DecisionParseError(msg)
    size = float(size_value)
    if not 0 <= size <= 1:
        msg = "size must be within [0, 1]"
        raise DecisionParseError(msg)

    action = cast("DecisionAction", action_value.upper())
    cognition: dict[str, float | None] = {
        "expected_return": None,
        "confidence": None,
        "downside_risk": None,
    }
    if require_cognition:
        forecast = value.get("expected_return")
        if isinstance(forecast, bool) or not isinstance(forecast, int | float):
            msg = "expected_return must be numeric"
            raise DecisionParseError(msg)
        # Bounded to match `_PERCENT`: magnitude strictly below 100. Anything tighter and a
        # constrained run could emit a grammar-legal string that fails to parse, and every such
        # rejection becomes a fallback HOLD, creating a treatment-correlated action artifact.
        if not -100 < float(forecast) < 100:
            msg = "expected_return must be within (-100, 100) percent"
            raise DecisionParseError(msg)
        cognition["expected_return"] = float(forecast)
        cognition["confidence"] = _unit_field(value.get("confidence"), "confidence")
        cognition["downside_risk"] = _unit_field(value.get("downside_risk"), "downside_risk")
    return AgentDecision(
        message=message_value.strip(),
        action=action,
        size=size,
        **cognition,  # type: ignore[arg-type]
    )


def fallback_hold(raw: str, *, max_message_length: int = 400) -> AgentDecision:
    """Create an explicit HOLD fallback for a separately logged parse failure."""
    if max_message_length <= 0:
        msg = "max_message_length must be positive"
        raise ValueError(msg)
    return AgentDecision(message=raw[:max_message_length], action="HOLD", size=0.0)
