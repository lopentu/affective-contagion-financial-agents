"""Tests for the isolated-reader filter and its preservation gate.

Most of these check ways a rewrite can lose content while still looking like a rewrite. A filter
that reduces a reader's response by deleting the bad news would score identically to one that
removes only affect, and the preservation gate is what tells them apart.
"""

import numpy as np
import pytest

from affective_contagion.firewall import (
    FilteredMessage,
    check_preservation,
    filter_message,
)
from affective_contagion.generation import BackendGeneration, ChatMessage, FloatVector
from affective_contagion.payload import FinancialPayload


class ScriptedBackend:
    """Return a fixed rewrite, validating input as strictly as the real backend does.

    The real generator refuses a constrained grammar on a free-text rewrite and requires a non-empty
    user turn, so this double asserts both. It does not check that the seed produces reproducible
    text.
    """

    def __init__(self, text: str) -> None:
        """Store the rewrite to return, and a log of the seeds this double was called with."""
        self.text = text
        self.seeds: list[int] = []

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        output_regex: str | None = None,
    ) -> BackendGeneration:
        """Return the scripted rewrite after checking the call is well formed."""
        assert output_regex is None, "a free-text rewrite must not be grammar-constrained"
        assert messages, "a rewrite needs a user turn"
        assert "Preserve every factual claim" in messages[0].content
        self.seeds.append(seed)
        return BackendGeneration(self.text)

    def read_activation(self, messages: tuple[ChatMessage, ...], *, layer: int) -> FloatVector:
        """Unused; present for protocol completeness."""
        assert messages
        assert layer >= 0
        return np.zeros(1)


@pytest.fixture
def payload() -> FinancialPayload:
    """Return a payload whose fact block carries numbers and every decision slot."""
    return FinancialPayload(
        asset="Meridian Bank",
        observed_event="an 18% deposit outflow over 48 hours",
        evidence=("Several counterparties reduced exposure",),
        forecast_direction="down",
        expected_magnitude="a further 5 to 8 percent",
        confidence="moderate",
        time_horizon="the next five sessions",
        recommended_action="SELL",
        position_size=0.5,
        urgency="review before the next session",
        synthetic=True,
    )


def test_the_payloads_own_fact_block_preserves_itself(payload: FinancialPayload) -> None:
    """A rewrite that is the fact block cannot be said to have dropped anything."""
    report = check_preservation(payload, payload.fact_block())
    assert report.preserved
    assert report.missing_numbers == ()
    assert report.missing_slots == ()


def test_a_dropped_number_is_caught(payload: FinancialPayload) -> None:
    """Removing the magnitude is the failure that would look like a working filter."""
    stripped = payload.fact_block().replace("18%", "some")
    report = check_preservation(payload, stripped)
    assert not report.preserved
    assert "18" in " ".join(report.missing_numbers)
    assert "removes content" in report.summary


def test_a_dropped_recommendation_is_caught(payload: FinancialPayload) -> None:
    """A rewrite that loses SELL has lost the decision, whatever else it kept."""
    report = check_preservation(payload, payload.fact_block().replace("SELL", "act"))
    assert not report.preserved
    assert "recommended_action" in report.missing_slots


def test_an_empty_rewrite_is_a_failure_not_a_pass(payload: FinancialPayload) -> None:
    """An empty string contains no forbidden content, which must not read as preservation."""
    report = check_preservation(payload, "   ")
    assert not report.preserved
    assert report.empty
    assert "nothing to show a reader" in report.summary


def test_case_and_spacing_do_not_defeat_the_check(payload: FinancialPayload) -> None:
    """A rewrite is not penalised for reformatting what it kept."""
    reflowed = "  ".join(payload.fact_block().upper().split())
    assert check_preservation(payload, reflowed).preserved


def test_filter_reports_a_failing_rewrite_rather_than_raising(payload: FinancialPayload) -> None:
    """Measure the failure rate: a bad rewrite comes back labelled, not as an exception."""
    backend = ScriptedBackend("Things are fine.")
    result = filter_message(backend, payload, "PANIC! an 18% deposit outflow!", seed=7)
    assert isinstance(result, FilteredMessage)
    assert not result.usable
    assert result.filtered == "Things are fine."
    assert result.original == "PANIC! an 18% deposit outflow!"
    assert backend.seeds == [7]


def test_filter_accepts_a_rewrite_that_kept_everything(payload: FinancialPayload) -> None:
    """The passing path, so a preserved rewrite is usable and carries its seed."""
    backend = ScriptedBackend(payload.fact_block())
    result = filter_message(backend, payload, "PANIC!! " + payload.fact_block(), seed=11)
    assert result.usable
    assert result.generation_seed == 11


def test_filtering_an_empty_message_is_a_programming_error(payload: FinancialPayload) -> None:
    """There is no sensible neutral rewrite of nothing, so this fails loudly."""
    with pytest.raises(ValueError, match="cannot filter an empty message"):
        filter_message(ScriptedBackend("x"), payload, "   ", seed=1)


def test_the_gate_cannot_see_a_reversed_claim(payload: FinancialPayload) -> None:
    """The gate is mechanical and cannot see a reversed claim.

    A rewrite that keeps every number and slot while inverting the meaning passes.
    """
    reversed_claim = payload.fact_block().replace("down", "down (now resolved, recovery under way)")
    assert check_preservation(payload, reversed_claim).preserved


def test_a_paraphrased_direction_passes(payload: FinancialPayload) -> None:
    """A direction restated as a synonym, such as "down" as "decline", passes.

    The gate accepts a closed list of direction words, so semantically exact rewrites pass while a
    rewrite that drops the direction entirely still fails.
    """
    paraphrased = payload.fact_block().replace(
        "The forecast is down by a further 5 to 8 percent",
        "The forecast indicates a further decline of 5 to 8 percent",
    )
    assert check_preservation(payload, paraphrased).preserved


def test_a_missing_direction_still_fails(payload: FinancialPayload) -> None:
    """Paraphrase tolerance must not become deletion tolerance."""
    gone = payload.fact_block().replace(
        "The forecast is down by a further 5 to 8 percent over the next five sessions",
        "The forecast covers the next five sessions",
    )
    report = check_preservation(payload, gone)
    assert not report.preserved
    assert "forecast_direction" in report.missing_slots
    assert "expected_magnitude" in report.missing_slots


def test_magnitude_units_cannot_silently_change(payload: FinancialPayload) -> None:
    """Reject '5 to 8 basis points' for '5 to 8 percent': every digit kept, claim changed."""
    switched = payload.fact_block().replace("5 to 8 percent", "5 to 8 basis points")
    report = check_preservation(payload, switched)
    assert not report.preserved
    assert "expected_magnitude" in report.missing_slots
