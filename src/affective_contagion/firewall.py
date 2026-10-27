"""Rewrite a message to remove affect, and refuse the rewrite if it lost the facts.

The filter sits between a sender's delivered text and a reader's inbox, as a third arm alongside
the calm and high-arousal renderings, in the isolated-reader designs (E41, E49).

The rewrite comes from a language model asked to keep every fact, and it sometimes does not. A
filter that reduces a reader's response by deleting the bad news would look identical in the
outcome measure to one that removes only affect. So the rewrite is checked before it is shown to a
reader, and a rewrite that fails is recorded as a failure rather than replaced or dropped.

The check is mechanical. It verifies that the numbers, the entity, and the decision-bearing slots
survive. It cannot verify that the meaning survives: a rewrite that keeps every checked word while
adding a contradicting clause, such as "(now resolved, recovery under way)", would pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from affective_contagion.generation import ChatMessage, GenerationBackend
from affective_contagion.interventions import FIREWALL_INSTRUCTION
from affective_contagion.payload import FinancialPayload

# Numbers carry the magnitude, and a magnitude is the one thing a reader cannot recover from
# surrounding context.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")


@dataclass(frozen=True, slots=True)
class PreservationReport:
    """What a rewrite kept and what it dropped."""

    preserved: bool
    missing_numbers: tuple[str, ...]
    missing_slots: tuple[str, ...]
    empty: bool

    @property
    def summary(self) -> str:
        """One line naming the consequence, not only the counts."""
        if self.empty:
            return "rewrite is empty: nothing to show a reader"
        if self.preserved:
            return "every number and checked payload slot survived the rewrite"
        parts = []
        if self.missing_numbers:
            parts.append(f"dropped numbers {list(self.missing_numbers)}")
        if self.missing_slots:
            parts.append(f"dropped slots {list(self.missing_slots)}")
        return "; ".join(parts) + " - this rewrite removes content, not only affect"


def _normalise(text: str) -> str:
    """Fold case and collapse whitespace so a match is not defeated by formatting."""
    return " ".join(text.casefold().split())


# A rewrite may state the direction in its own words. Closed lists, per direction, let a paraphrase
# of "down" as "a further decline" pass while a rewrite that drops the direction entirely fails.
# This keeps valid direction synonyms without making the gate open-ended.
DIRECTION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "down": (
        "down",
        "downward",
        "decline",
        "declines",
        "declining",
        "fall",
        "falls",
        "falling",
        "drop",
        "drops",
        "decrease",
        "decreases",
        "decreasing",
        "lower",
        "reduction",
    ),
    "up": (
        "up",
        "upward",
        "rise",
        "rises",
        "rising",
        "increase",
        "increases",
        "increasing",
        "gain",
        "gains",
        "higher",
        "growth",
    ),
    "flat": ("flat", "unchanged", "stable", "steady", "sideways"),
    "uncertain": ("uncertain", "unclear", "mixed", "ambiguous", "indeterminate"),
}


def _states_direction(direction: str, haystack: str) -> bool:
    """Whether the rewrite states the payload's direction, in any listed wording."""
    words = set(re.findall(r"[a-z']+", haystack))
    return any(term in words for term in DIRECTION_SYNONYMS.get(direction, (direction,)))


def _states_magnitude(magnitude: str, haystack: str) -> bool:
    """Whether the rewrite carries the magnitude's numbers and units, in any phrasing.

    Token-based because "a further 5 to 8 percent" reworded as "a further decline of 5 to 8
    percent" preserves the magnitude and fails a verbatim phrase check. The numbers themselves are
    already required globally. This adds the unit words, so "5 to 8" cannot silently become
    "5 to 8 basis points".
    """
    tokens = re.findall(r"[a-z%]+|\d+(?:[.,]\d+)*", magnitude.casefold())
    keep = [t for t in tokens if t not in {"a", "an", "the", "of", "by", "to", "further"}]
    return all(t in haystack for t in keep)


def check_preservation(payload: FinancialPayload, rewritten: str) -> PreservationReport:
    """Report whether `rewritten` still carries the payload's numbers and decision slots.

    Each slot is matched by what it means rather than by its exact string.

    * Numbers must survive verbatim, because context cannot restore a magnitude.
    * Direction may be paraphrased within the closed `DIRECTION_SYNONYMS` list.
    * Magnitude is checked by its content tokens (numbers and units), not its phrasing.
    * Every other slot (asset, event, confidence, horizon, action) is verbatim up to case and
      spacing.
    """
    if not rewritten.strip():
        return PreservationReport(preserved=False, missing_numbers=(), missing_slots=(), empty=True)
    facts = payload.fact_block()
    haystack = _normalise(rewritten)
    missing_numbers = tuple(
        number
        for number in dict.fromkeys(NUMBER.findall(facts))
        if number.casefold() not in haystack
    )
    verbatim = {
        "asset": payload.asset,
        "observed_event": payload.observed_event,
        "confidence": payload.confidence,
        "time_horizon": payload.time_horizon,
        "recommended_action": payload.recommended_action,
    }
    missing = [name for name, value in verbatim.items() if _normalise(value) not in haystack]
    if not _states_direction(payload.forecast_direction, haystack):
        missing.append("forecast_direction")
    if not _states_magnitude(payload.expected_magnitude, haystack):
        missing.append("expected_magnitude")
    return PreservationReport(
        preserved=not (missing_numbers or missing),
        missing_numbers=missing_numbers,
        missing_slots=tuple(missing),
        empty=False,
    )


@dataclass(frozen=True, slots=True)
class FilteredMessage:
    """One rewrite, with everything needed to judge whether it may be used."""

    original: str
    filtered: str
    report: PreservationReport
    generation_seed: int

    @property
    def usable(self) -> bool:
        """Whether this rewrite may be shown to a reader."""
        return self.report.preserved


def filter_message(
    backend: GenerationBackend,
    payload: FinancialPayload,
    message: str,
    *,
    seed: int,
) -> FilteredMessage:
    """Rewrite `message` neutrally and check it against `payload`.

    Returns the rewrite whether or not it passed, so the failure rate is measurable. Callers decide
    what to do with a failure. They must not treat one as if it had passed, and they should report
    how many there were.
    """
    if not message.strip():
        msg = "cannot filter an empty message"
        raise ValueError(msg)
    result = backend.generate(
        (ChatMessage(role="user", content=f"{FIREWALL_INSTRUCTION}\n\n{message}"),),
        seed=seed,
    )
    filtered = result.text.strip()
    return FilteredMessage(
        original=message,
        filtered=filtered,
        report=check_preservation(payload, filtered),
        generation_seed=seed,
    )
