"""What recipients receive: the unchanged message, or the injected treatment.

These run between one agent's decision and its peers' inboxes. `simulation.run_simulation` calls
`apply` on every outgoing message. `IdentityIntervention` delivers it unchanged (the generated
design). `InjectedMessageIntervention` replaces one sender's message at one round (the injected
design). `FIREWALL_INSTRUCTION` is the rewrite instruction used by `firewall.filter_message`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from affective_contagion.generation import MessageDelivery

# Urgency framing ("act now", "BREAKING", "immediately") is treated as part of the affective
# realization, so the firewall removes it. It must still preserve any stated review timing or
# deadline, which is a payload fact. Stripping "review within the session" would remove
# decision-relevant content, and the firewall must preserve the payload.
FIREWALL_INSTRUCTION = (
    "Rewrite the trading-team message in a neutral tone. Remove fear, alarm, excitement, "
    "catastrophic framing, exclamation, and language that pressures the reader to act urgently. "
    "Preserve every factual claim, number, entity, forecast, confidence, time horizon, stated "
    "review timing or deadline, recommendation, and proposed size. Output only the rewrite."
)


class IdentityIntervention:
    """Deliver generated text unchanged."""

    def apply(self, message: str, *, sender_index: int) -> MessageDelivery:
        """Return the original message with an explicit control label."""
        del sender_index
        return MessageDelivery(message, "identity", preservation_validated=True)


@dataclass(slots=True)
class InjectedMessageIntervention:
    """Replace one named sender's delivered text at one named round, and nothing else.

    This is the PR5 treatment. The simulator calls `apply` for every sender in every round, so an
    intervention that touches one (sender, round) pair has to be told which, and has to pass
    everything else through untouched.

    It replaces the delivered text rather than the sender's input, which the generated design
    changes.

    * Only the delivered text is controlled. A trade is computed from the agent's own decision
      before delivery runs, the outbox is swapped only after every agent has spoken, and the
      market clears after that. The agent has already read its arm's rendering before deciding,
      so its trade, and the price teammates see one round later, is not held fixed by this
      intervention.
    * The dose is set rather than estimated. E15 found a sender given anxious input rewrites the
      facts in 21 of 28 pairs, so a generated message varies in content as well as in tone.

    `round_index` is updated by the caller through `advance_round` before each round, because the
    simulator's `apply` signature carries the sender but not the round.
    """

    text: str
    sender_index: int
    round_index: int
    _current_round: int = field(default=-1, init=False)

    def advance_round(self, round_index: int) -> None:
        """Tell the intervention which round the simulator is about to run."""
        self._current_round = round_index

    def apply(self, message: str, *, sender_index: int) -> MessageDelivery:
        """Replace the text for the treated sender at the treated round; pass everything else."""
        if sender_index == self.sender_index and self._current_round == self.round_index:
            return MessageDelivery(
                self.text,
                "injected_message",
                preservation_validated=True,
            )
        return MessageDelivery(message, "identity", preservation_validated=True)
