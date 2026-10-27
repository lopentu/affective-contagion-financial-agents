"""Run-health diagnostics kept separate from scientific outcome metrics."""

from __future__ import annotations

from dataclasses import dataclass

from affective_contagion.simulation import SimulationResult


@dataclass(frozen=True, slots=True)
class RunHealth:
    """Basic indicators of formatting failures and treatment uptake."""

    parse_failure_rate: float
    hold_rate: float
    shock_window_keyword_hits: int
    shock_window_messages: int


def evaluate_run_health(
    result: SimulationResult,
    *,
    shock_round: int,
    shock_duration: int,
    shock_keywords: tuple[str, ...],
    treated_role: str | None = None,
) -> RunHealth:
    """Measure parser health and whether shock-period messages mention prespecified concepts.

    With a treated role, only that agent read the stimulus, so only its messages are a direct
    response to it. Counting every agent would dilute the uptake rate by the five that could not
    have seen the text.
    """
    window = [
        row
        for row in result.records
        if shock_round <= row.round_index < shock_round + shock_duration
        and (treated_role is None or row.role == treated_role)
    ]
    normalized_keywords = tuple(item.casefold() for item in shock_keywords)
    hits = sum(
        any(keyword in row.decision.message.casefold() for keyword in normalized_keywords)
        for row in window
    )
    hold_rate = (
        0.0
        if not result.records
        else sum(row.decision.action == "HOLD" for row in result.records) / len(result.records)
    )
    return RunHealth(result.parse_failure_rate, hold_rate, hits, len(window))
