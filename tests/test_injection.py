"""Tests for the injected-message treatment PR5 uses."""

from __future__ import annotations

from affective_contagion.interventions import (
    IdentityIntervention,
    InjectedMessageIntervention,
)
from affective_contagion.simulation import _announce_round


def _intervention() -> InjectedMessageIntervention:
    return InjectedMessageIntervention(text="INJECTED", sender_index=2, round_index=8)


def test_replaces_only_the_treated_sender_at_the_treated_round() -> None:
    """One sender, one round, and nothing else is touched."""
    injection = _intervention()
    injection.advance_round(8)
    treated = injection.apply("original", sender_index=2)
    assert treated.delivered_text == "INJECTED"
    assert treated.intervention_name == "injected_message"


def test_other_senders_pass_through_untouched_at_the_treated_round() -> None:
    """Every other agent's message must be delivered byte-identically."""
    injection = _intervention()
    injection.advance_round(8)
    for sender in (0, 1, 3, 4, 5):
        delivery = injection.apply(f"message from {sender}", sender_index=sender)
        assert delivery.delivered_text == f"message from {sender}"
        assert delivery.intervention_name == "identity"


def test_the_treated_sender_passes_through_at_every_other_round() -> None:
    """Before and after the injection round the treated agent speaks for itself."""
    injection = _intervention()
    for round_index in (0, 7, 9, 11):
        injection.advance_round(round_index)
        delivery = injection.apply("own words", sender_index=2)
        assert delivery.delivered_text == "own words"
        assert delivery.intervention_name == "identity"


def test_nothing_is_injected_before_a_round_is_declared() -> None:
    """The round has to be set explicitly, so a caller that forgets injects nothing."""
    injection = _intervention()
    assert injection.apply("own words", sender_index=2).delivered_text == "own words"


def test_two_arms_differ_in_exactly_one_delivered_message() -> None:
    """Arms N and A share every byte except the treated sender's round-8 text."""
    round_index, sender = 8, 2
    originals = {(r, s): f"r{r}s{s}" for r in range(12) for s in range(6)}
    arms = {}
    for arm, text in (("N", "neutral rendering"), ("A", "anxious rendering")):
        injection = InjectedMessageIntervention(
            text=text, sender_index=sender, round_index=round_index
        )
        delivered = {}
        for r in range(12):
            injection.advance_round(r)
            for s in range(6):
                delivered[(r, s)] = injection.apply(
                    originals[(r, s)], sender_index=s
                ).delivered_text
        arms[arm] = delivered
    differing = [key for key in arms["N"] if arms["N"][key] != arms["A"][key]]
    assert differing == [(round_index, sender)]


def test_simulation_announces_the_round_to_the_intervention() -> None:
    """The simulator must tell a round-targeting intervention which round it is running.

    Without this the injection never fires, and the two arms are byte-identical. That is a silent
    null rather than a crash.
    """
    injection = InjectedMessageIntervention(text="X", sender_index=0, round_index=3)
    _announce_round(injection, 3)
    assert injection.apply("own", sender_index=0).delivered_text == "X"


def test_announcing_a_round_is_optional_for_other_interventions() -> None:
    """A round-agnostic intervention has no such method and must not break."""
    _announce_round(IdentityIntervention(), 5)  # must not raise
