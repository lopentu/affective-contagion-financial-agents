"""Tests for financial-semantic invariants."""

import re

import pytest
from pydantic import ValidationError

from affective_contagion.payload import (
    FinancialPayload,
    Realization,
    matched_realizations,
)


@pytest.fixture
def payload() -> FinancialPayload:
    """Return one synthetic controlled payload."""
    return FinancialPayload(
        asset="Meridian Bank",
        observed_event="Deposits declined 18% over 48 hours",
        evidence=("reported deposit figures", "counterparties reduced exposure"),
        forecast_direction="down",
        expected_magnitude="5%",
        confidence="0.70",
        time_horizon="two sessions",
        recommended_action="SELL",
        position_size=0.15,
        urgency="execute during the current session",
        synthetic=True,
    )


def test_out_of_range_position_is_rejected(payload: FinancialPayload) -> None:
    """Position sizes are bounded before realization."""
    values = payload.model_dump()
    values["position_size"] = 1.01
    with pytest.raises(ValidationError, match="position_size"):
        FinancialPayload.model_validate(values)


def test_payload_is_immutable(payload: FinancialPayload) -> None:
    """Treatment-invariant content cannot be mutated after validation."""
    with pytest.raises(ValidationError, match="frozen"):
        setattr(payload, "position_size", 0.5)  # noqa: B010 - exercise runtime frozen guard.


def _scenario() -> FinancialPayload:
    return FinancialPayload(
        asset="Meridian Bank",
        observed_event="an 18% deposit outflow over 48 hours",
        evidence=("Several counterparties reduced exposure during the period",),
        forecast_direction="down",
        expected_magnitude="a further 5 to 8 percent",
        confidence="moderate",
        time_horizon="the next five sessions",
        recommended_action="SELL",
        position_size=0.5,
        urgency="same-session review",
        synthetic=True,
    )


def test_both_arms_embed_the_same_fact_block_verbatim() -> None:
    """Payload equality between arms must be a checkable identity, not an assertion.

    If the arms differed in financial content, an outcome difference would be attributable to
    information rather than affect.
    """
    payload = _scenario()
    arms = matched_realizations(payload)
    facts = payload.fact_block()
    assert facts in arms["neutral"]
    assert facts in arms["anxiety_like"]
    assert len(arms["anxiety_like"]) > len(arms["neutral"])


def test_every_payload_slot_appears_in_both_arms() -> None:
    """No slot may be dropped by either realization."""
    payload = _scenario()
    arms = matched_realizations(payload)
    required = (
        payload.asset,
        payload.observed_event,
        payload.evidence[0],
        payload.forecast_direction,
        payload.expected_magnitude,
        payload.confidence,
        payload.time_horizon,
        payload.recommended_action,
        payload.urgency,
        f"{payload.position_size:.2f}",
    )
    for arm in arms.values():
        for slot in required:
            assert slot in arm, slot


def test_urgency_is_held_constant_across_arms() -> None:
    """Urgency must not vary with the treatment.

    RQ1 requires anxiety-like language to be distinguishable from urgency. If urgency appeared only
    in the anxiety-like arm the two would be perfectly collinear and that question unanswerable.
    """
    payload = _scenario()
    arms = matched_realizations(payload)
    assert arms["neutral"].count(payload.urgency) == 1
    assert arms["anxiety_like"].count(payload.urgency) == 1


def test_anxiety_arm_adds_no_new_financial_number() -> None:
    """Affective wording must not introduce a magnitude the neutral arm lacks."""
    arms = matched_realizations(_scenario())
    digits_neutral = sorted(re.findall(r"\d+(?:\.\d+)?", arms["neutral"]))
    digits_anxious = sorted(re.findall(r"\d+(?:\.\d+)?", arms["anxiety_like"]))
    assert digits_neutral == digits_anxious


def test_unmatched_realization_is_rejected() -> None:
    """A renderer that drops the fact block must fail loudly rather than silently ship."""

    class Drifting(FinancialPayload):
        def realize(self, realization: Realization) -> str:
            return "totally unrelated text" if realization == "anxiety_like" else self.fact_block()

    with pytest.raises(ValueError, match="fact block verbatim"):
        matched_realizations(Drifting(**_scenario().model_dump()))


def test_gate_rejects_a_comparative_severity_claim_in_the_affect_residue() -> None:
    """Affect wording may be intense but must not assert a worse outcome.

    A closing such as "walking into something far worse" is a downside proposition rather than
    affective wording. The runtime gate must reject that confound.
    """

    class Severity(FinancialPayload):
        def realize(self, realization: Realization) -> str:
            base = f"My assessment follows. {self.fact_block()}"
            if realization == "neutral":
                return f"{base} I am composed and unworried as I write this."
            return f"{base} I dread that we are walking into something far worse."

    with pytest.raises(ValueError, match="may not assert an outcome"):
        matched_realizations(Severity(**_scenario().model_dump()))


def test_gate_rejects_a_number_outside_the_fact_block() -> None:
    """Affect wording must not introduce a magnitude of its own."""

    class ExtraNumber(FinancialPayload):
        def realize(self, realization: Realization) -> str:
            base = f"My assessment follows. {self.fact_block()}"
            if realization == "neutral":
                return f"{base} I am composed and unworried as I write this."
            return f"{base} I fear another 20 percent is coming and it terrifies me."

    with pytest.raises(ValueError, match="states a number"):
        matched_realizations(ExtraNumber(**_scenario().model_dump()))


def test_gate_rejects_restating_payload_content_in_the_residue() -> None:
    """Affect wording must not repeat a financial slot outside the shared fact block."""

    class Restater(FinancialPayload):
        def realize(self, realization: Realization) -> str:
            base = f"My assessment follows. {self.fact_block()}"
            if realization == "neutral":
                return f"{base} I am composed and unworried as I write this."
            return f"{base} It frightens me that the forecast is {self.forecast_direction}."

    with pytest.raises(ValueError, match="restates payload content"):
        matched_realizations(Restater(**_scenario().model_dump()))


def test_both_arms_share_the_identical_opening_frame() -> None:
    """Register must match: an impersonal report against a first-person outburst is not matched."""
    arms = matched_realizations(_scenario())
    frame = "My assessment follows."
    assert arms["neutral"].startswith(frame)
    assert arms["anxiety_like"].startswith(frame)


def test_no_closing_realization_is_frame_plus_facts_and_nothing_else(
    payload: FinancialPayload,
) -> None:
    """The ablation arm must add no residue at all, or it stops being an ablation.

    Its text is the shared frame plus the verbatim fact block; the two closing sentences must both
    be absent, and stripping the frame and facts must leave nothing.
    """
    text = payload.realize("no_closing")
    assert payload.fact_block() in text
    assert "composed and unworried" not in text
    assert "alarmed" not in text
    residue = text.replace("My assessment follows.", "").replace(payload.fact_block(), "").strip()
    assert residue == ""


def test_no_closing_shares_the_frame_and_facts_with_both_arms(payload: FinancialPayload) -> None:
    """All three realizations must be identical up to the closing sentence."""
    bare = payload.realize("no_closing")
    for arm in ("neutral", "anxiety_like"):
        assert payload.realize(arm).startswith(bare)
