"""Replay one peer's turn outside the simulation, with everything but the message held fixed.

This runs beside the round loop, not inside it. `simulation.run_simulation` models what a team
does over time. This module asks a narrower question: does an anxiety-like wording of the same
facts change what a reader writes, with everything else held fixed.

The inbox is supplied by the caller. It can be the whole inbox a real peer would get, with the
treated message as one of five, or the treated message alone, as in the context comparison's
`bare` condition. When the four untreated peers' messages are included, they come from a verified
matched pair, where they are byte-identical across arms, so only the treated message differs.

Reading outside the loop differs from the full simulation in three ways.

**Power.** A 7-round run yields one number per arm, so each seed gives one paired observation.
Here each draw is one generation, so the same compute gives many readings. Draws sharing a stimulus
are not independent, so the analysis still pairs by seed (see `clustered`).

**Price.** In the simulation a peer sees the treated agent's message and a price its trade may
have moved. Here price is an input held identical across arms, so only the text differs.

**Reproducibility.** A simulation pair must stay byte-identical for several rounds before the
treatment lands, and any divergence cascades through inboxes and inflates the paired spread. Here
there is no history, so nothing can desynchronise.

It does not model peer-to-peer dynamics, market feedback, or anything past one hop. Those need the
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from affective_contagion.generation import (
    DecisionGenerator,
    GenerationContext,
    InboundMessage,
)
from affective_contagion.messages import AgentDecision, DecisionParseError, parse_agent_decision
from affective_contagion.text_affect import VADLexicon, VADScore


@dataclass(frozen=True, slots=True)
class StimulusInbox:
    """The messages a reader is shown, with only the treated one differing by arm.

    `peer_messages` holds the untreated agents' shock-round messages, or is empty when the reader
    sees the treated message alone (the context comparison's `bare` condition). The runners store
    all five untreated agents' messages; `read_in_isolation` drops the reader's own, so a reader
    sees the other four. In a verified matched pair these messages are byte-identical across arms,
    because those agents have read nothing yet and see the same price, so only `treated_text`
    differs between arms.
    """

    seed: int
    arm: str
    treated_role: str
    treated_text: str
    # (sender_index, sender_role, text) for the untreated peers, identical across both arms.
    peer_messages: tuple[tuple[int, str, str], ...]
    treated_index: int


@dataclass(frozen=True, slots=True)
class Reading:
    """One isolated agent's response to one reconstructed inbox."""

    seed: int
    arm: str
    reader_role: str
    draw: int
    generation_seed: int
    raw_output: str
    message: str
    action: str
    size: float
    parse_succeeded: bool
    affect: VADScore
    # PR7's cognition outcomes, or None when the run did not request them.
    expected_return: float | None = None
    confidence: float | None = None
    downside_risk: float | None = None
    # Which model produced this, recorded per reading rather than inferred from the filename, so a
    # reading file merged from several models stays distinguishable.
    repository: str = ""
    revision: str = ""
    # What the reader was shown, so a reading can be traced to its stimulus from the record alone
    # rather than by re-deriving it from the source pair.
    inbox_texts: tuple[str, ...] = ()
    inbox_senders: tuple[str, ...] = ()

    @property
    def cluster(self) -> tuple[int, str]:
        """Identify the group this reading shares a stimulus with.

        Draws sharing a (seed, reader_role) read the *same* pair of messages, so they are not
        independent observations. Any analysis must model this rather than counting draws as n.
        """
        return (self.seed, self.reader_role)


def read_in_isolation(
    generator: DecisionGenerator,
    stimulus: StimulusInbox,
    *,
    reader_role: str,
    reader_index: int,
    price: float,
    fundamental_value: float,
    draws: int,
    vad_lexicon: VADLexicon,
    round_index: int = 0,
) -> tuple[Reading, ...]:
    """Generate ``draws`` independent responses to one inbox from one reader.

    ``price`` and ``fundamental_value`` are caller-supplied and must be identical across arms.
    They are required rather than derived from the run the stimulus came from, because the arms'
    recorded prices could differ and would reintroduce the price channel.

    ``round_index`` participates in the generation seed, so each draw gets a distinct, reproducible
    seed while the prompt stays fixed.

    The reader is excluded from its own inbox, matching `full` topology: no agent hears itself.
    """
    if draws < 1:
        msg = "draws must be positive"
        raise ValueError(msg)
    # The treated agent's message plus any peers' messages, minus the reader itself, ordered by
    # sender index so both arms present them identically.
    #
    # An empty `treated_text` contributes nothing, rather than an empty message. The simulation
    # drops empty outbox entries too (`if outbox[sender]`), so a silent sender is absent from a
    # peer's inbox. A blank message would turn the no-communication baseline into a message from
    # nobody, which is a different condition.
    entries = [
        *(
            [(stimulus.treated_index, stimulus.treated_role, stimulus.treated_text)]
            if stimulus.treated_text
            else []
        ),
        *(item for item in stimulus.peer_messages if item[0] != reader_index and item[2]),
    ]
    inbox = tuple(
        InboundMessage(sender_index=index, sender_role=role, text=text)
        for index, role, text in sorted(entries)
    )
    # Same single switch as the round loop: whatever the generator asked for is what is required.
    requires_cognition = bool(getattr(generator, "cognition_outcomes", False))
    readings: list[Reading] = []
    for draw in range(draws):
        context = GenerationContext(
            round_index=round_index + draw,
            agent_index=reader_index,
            role=reader_role,
            price=price,
            fundamental_value=fundamental_value,
            # No stimulus. This reader is a peer, and peers never read the report directly, so its
            # only source is its inbox.
            news=None,
            inbox=inbox,
        )
        generation = generator.generate(context)
        try:
            decision = parse_agent_decision(
                generation.raw_output, require_cognition=requires_cognition
            )
            parse_succeeded = True
        except DecisionParseError:
            decision = AgentDecision(message=generation.raw_output[:400], action="HOLD", size=0.0)
            parse_succeeded = False
        affect = vad_lexicon.score(decision.message)
        readings.append(
            Reading(
                repository=generation.repository,
                revision=generation.revision or "",
                inbox_texts=tuple(item.text for item in inbox),
                inbox_senders=tuple(item.sender_role for item in inbox),
                expected_return=decision.expected_return,
                confidence=decision.confidence,
                downside_risk=decision.downside_risk,
                seed=stimulus.seed,
                arm=stimulus.arm,
                reader_role=reader_role,
                draw=draw,
                generation_seed=generation.generation_seed,
                raw_output=generation.raw_output,
                message=decision.message,
                action=decision.action,
                size=decision.size,
                parse_succeeded=parse_succeeded,
                affect=affect,
            )
        )
    return tuple(readings)
