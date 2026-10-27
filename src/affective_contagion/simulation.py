"""Run the round loop that turns a config into records and a price path.

The loop makes four decisions per round:

* **who reads the stimulus**: only `config.treated_role`, so peers can learn of the event only
  through a message, or through the price if the treated agent's trade moves it. Broadcasting
  instead makes simultaneous movement indistinguishable from transmission.
* **who reads whom**: `topology.inbound_neighbors`, and only the *previous* round's messages, so no
  agent can see output produced in the same round.
* **what reaches the market**: the fill the position limit can absorb, so `positions` and the price
  path cannot disagree.
* **what is recorded**: one immutable `AgentRecord` per agent per round, carrying the prompt, the
  raw output, the parsed decision, the delivered message, and the message's affect score.

Model inference is injected, never imported, so the whole loop is testable without downloading
weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from affective_contagion.config import ExperimentConfig
from affective_contagion.generation import (
    DecisionGeneration,
    DecisionGenerator,
    GenerationContext,
    InboundMessage,
    MessageDelivery,
    MessageIntervention,
)
from affective_contagion.interventions import IdentityIntervention
from affective_contagion.market import MarketStep, SimpleMarket
from affective_contagion.messages import (
    AgentDecision,
    DecisionParseError,
    fallback_hold,
    parse_agent_decision,
)
from affective_contagion.text_affect import VADLexicon, VADScore
from affective_contagion.topology import inbound_neighbors


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """Auditable result of one agent step."""

    round_index: int
    agent_index: int
    role: str
    context: GenerationContext
    generation: DecisionGeneration
    decision: AgentDecision
    delivery: MessageDelivery
    parse_succeeded: bool
    position_after: float
    message_affect: VADScore | None

    @property
    def raw_output(self) -> str:
        """Return the original model output."""
        return self.generation.raw_output

    @property
    def delivered_message(self) -> str:
        """Return text seen by recipients after the intervention."""
        return self.delivery.delivered_text


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable records returned by one completed simulation."""

    records: tuple[AgentRecord, ...]
    market_steps: tuple[MarketStep, ...]

    @property
    def affect_direction_captured(self) -> bool:
        """Report whether any activation projection was recorded.

        With no affect direction loaded every score is None, so the saved activation matrix is
        all-NaN. The manifest records this, so a reader can see that no activation measurement is
        available; the message, position and price series are unaffected.
        """
        return any(record.generation.affect_score is not None for record in self.records)

    @property
    def parse_failure_rate(self) -> float:
        """Return the share of model outputs that required explicit fallback."""
        if not self.records:
            return 0.0
        failures = sum(not record.parse_succeeded for record in self.records)
        return failures / len(self.records)

    def affect_matrix(self, n_agents: int) -> NDArray[np.float64]:
        """Return round-by-agent activation projections, with NaN for unavailable scores."""
        if n_agents < 1:
            msg = "n_agents must be positive"
            raise ValueError(msg)
        n_rounds = 0 if not self.records else max(row.round_index for row in self.records) + 1
        values = np.full((n_rounds, n_agents), np.nan, dtype=np.float64)
        for row in self.records:
            values[row.round_index, row.agent_index] = (
                np.nan if row.generation.affect_score is None else row.generation.affect_score
            )
        return values

    def position_matrix(self, n_agents: int) -> NDArray[np.float64]:
        """Return round-by-agent end-of-step positions."""
        if n_agents < 1:
            msg = "n_agents must be positive"
            raise ValueError(msg)
        n_rounds = 0 if not self.records else max(row.round_index for row in self.records) + 1
        values = np.full((n_rounds, n_agents), np.nan, dtype=np.float64)
        for row in self.records:
            values[row.round_index, row.agent_index] = row.position_after
        return values

    def message_arousal_matrix(self, n_agents: int) -> NDArray[np.float64]:
        """Return round-by-agent lexical arousal, with NaN when no scorer was supplied."""
        if n_agents < 1:
            msg = "n_agents must be positive"
            raise ValueError(msg)
        n_rounds = 0 if not self.records else max(row.round_index for row in self.records) + 1
        values = np.full((n_rounds, n_agents), np.nan, dtype=np.float64)
        for row in self.records:
            values[row.round_index, row.agent_index] = (
                np.nan if row.message_affect is None else row.message_affect.arousal
            )
        return values


MAX_ABS_POSITION = 1.0


@dataclass(frozen=True, slots=True)
class StageEvent:
    """One executed pipeline stage, emitted as the loop runs.

    Events are recorded as the loop runs rather than reconstructed from a fixed stage list, so a
    replay describes the stages that actually executed.
    """

    round_index: int
    agent_index: int | None
    role: str | None
    stage: str
    detail: str


class StageTrace:
    """Collect stage events. Passing one is optional and costs nothing when omitted."""

    def __init__(self) -> None:
        """Start empty."""
        self.events: list[StageEvent] = []

    def emit(
        self,
        *,
        round_index: int,
        stage: str,
        detail: str,
        agent_index: int | None = None,
        role: str | None = None,
    ) -> None:
        """Append one executed stage."""
        self.events.append(StageEvent(round_index, agent_index, role, stage, detail))


def _wants_cognition(generator: DecisionGenerator) -> bool:
    """Report whether this generator asked for the RQ2 cognition outcomes.

    Read off the generator rather than passed separately, because the prompt text, the decoding
    grammar and the parser's requirement must move together. Independent flags would let a run
    request the fields, constrain decoding to a grammar forbidding them, then parse without
    demanding them, and record an empty outcome as a success.
    """
    return bool(getattr(generator, "cognition_outcomes", False))


def _signed_order(decision: AgentDecision) -> float:
    """Convert a bounded action and size to signed demand."""
    direction = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}[decision.action]
    return direction * decision.size


def _fill_within_position_limit(current_position: float, requested_order: float) -> float:
    """Return the part of a requested order the position limit can absorb.

    The position bound and the market must see the same trade. Clamping only the position would
    let an agent already at the limit keep pushing price, so `positions` and the price path would
    disagree.
    """
    target = min(MAX_ABS_POSITION, max(-MAX_ABS_POSITION, current_position + requested_order))
    return target - current_position


def _stage_noter(
    trace: StageTrace | None, round_index: int, agent_index: int, role: str
) -> Callable[[str, str], None]:
    """Return a `(stage, detail)` emitter bound to the agent step currently executing."""

    def note(stage: str, detail: str) -> None:
        if trace is not None:
            trace.emit(
                round_index=round_index,
                agent_index=agent_index,
                role=role,
                stage=stage,
                detail=detail,
            )

    return note


def _parse_detail(decision: AgentDecision, *, parse_succeeded: bool) -> str:
    """Describe the parse outcome, naming the fallback when it fired."""
    if parse_succeeded:
        return f"{decision.action} size {decision.size:.2f}"
    return "PARSE FAILED -> messages.fallback_hold: HOLD size 0.00"


def _delivery_detail(delivery: MessageDelivery) -> str:
    """Describe what recipients will receive."""
    body = (
        f"{len(delivery.delivered_text)} chars delivered"
        if delivery.delivered_text
        else "nothing delivered"
    )
    return f"{delivery.intervention_name}: {body}"


def _context_detail(price: float, news: str | None, treated_role: str | None) -> str:
    """Describe the observable state, naming how the stimulus was delivered.

    The delivery mode has to be read from the config rather than assumed: with no treated role the
    stimulus is broadcast to everyone, and labelling each agent "treated role" would misreport the
    design in exactly the runs where that distinction carries the causal claim.
    """
    if news is None:
        return f"price {price:.4f}; stimulus absent"
    delivery = "treated role" if treated_role else "broadcast"
    return f"price {price:.4f}; stimulus PRESENT ({delivery})"


def _affect_detail(affect: VADScore) -> str:
    """Describe the message-level measurement."""
    return (
        f"arousal {affect.arousal:.4f}; valence {affect.valence:.4f}; "
        f"coverage {affect.coverage:.3f}"
    )


def _round_stimulus(
    config: ExperimentConfig, shock_map: dict[str, str], round_index: int
) -> str | None:
    """Return the stimulus text for this round, or None outside the shock window."""
    in_shock_window = config.shock_round <= round_index < config.shock_round + config.shock_duration
    return shock_map.get(config.shock_type) if in_shock_window else None


def _announce_round(transform: MessageIntervention, round_index: int) -> None:
    """Tell an intervention which round is about to run, if it cares.

    An intervention that targets a single round needs this, because `apply` carries the sender
    but not the round. Round-agnostic interventions need not implement `advance_round`.
    """
    advance_round = getattr(transform, "advance_round", None)
    if advance_round is not None:
        advance_round(round_index)


def run_simulation(
    config: ExperimentConfig,
    generator: DecisionGenerator,
    market: SimpleMarket,
    *,
    shocks: dict[str, str] | None = None,
    intervention: MessageIntervention | None = None,
    vad_lexicon: VADLexicon | None = None,
    trace: StageTrace | None = None,
) -> SimulationResult:
    """Run agents using previous-round messages and an injected generation backend.

    Model sampling and market randomness are owned by their injected components. This prevents the
    orchestrator from coupling random-number consumption across scientific conditions.
    """
    if market.parameters.fundamental_value != config.fundamental_value:
        msg = "market and experiment fundamental values must match"
        raise ValueError(msg)
    shock_map = shocks or {}
    if config.shock_type != "none" and config.shock_type not in shock_map:
        msg = f"missing text for shock type: {config.shock_type}"
        raise ValueError(msg)

    transform = intervention or IdentityIntervention()
    neighbors = inbound_neighbors(config.topology, config.n_agents)
    outbox = dict.fromkeys(range(config.n_agents), "")
    positions = [0.0] * config.n_agents
    records: list[AgentRecord] = []

    for round_index in range(config.n_rounds):
        stimulus = _round_stimulus(config, shock_map, round_index)
        _announce_round(transform, round_index)
        orders: list[float] = []
        next_outbox: dict[int, str] = {}

        for agent_index, role in enumerate(config.roles):
            # With a treated role named, only that agent reads the stimulus. Every other agent can
            # learn of it only through peer messages, or through the price if the treated agent's
            # trade moves it, rather than by reading the same text.
            treated = config.treated_role is None or role == config.treated_role
            news = stimulus if treated else None
            inbox = tuple(
                InboundMessage(
                    sender_index=sender,
                    sender_role=config.roles[sender],
                    text=outbox[sender],
                )
                for sender in neighbors[agent_index]
                if outbox[sender]
            )
            note = _stage_noter(trace, round_index, agent_index, role)
            senders = ", ".join(config.roles[item.sender_index] for item in inbox)
            note(
                "topology.inbound_neighbors",
                f"{len(inbox)} inbound message(s)" + (f" from {senders}" if inbox else ""),
            )
            context = GenerationContext(
                round_index=round_index,
                agent_index=agent_index,
                role=role,
                price=market.price,
                fundamental_value=config.fundamental_value,
                news=news,
                inbox=inbox,
            )
            note(
                "GenerationContext",
                _context_detail(market.price, news, config.treated_role),
            )
            generation = generator.generate(context)
            note(
                "generator.generate",
                f"seed {generation.generation_seed}; {len(generation.raw_output)} chars",
            )
            raw = generation.raw_output
            try:
                decision = parse_agent_decision(raw, require_cognition=_wants_cognition(generator))
                parse_succeeded = True
            except DecisionParseError:
                decision = fallback_hold(raw)
                parse_succeeded = False
            note(
                "messages.parse_agent_decision",
                _parse_detail(decision, parse_succeeded=parse_succeeded),
            )

            filled = _fill_within_position_limit(positions[agent_index], _signed_order(decision))
            positions[agent_index] += filled
            note(
                "simulation._fill_within_position_limit",
                f"filled {filled:+.3f}; position "
                f"{positions[agent_index] - filled:+.2f} -> {positions[agent_index]:+.2f}",
            )
            delivery = transform.apply(decision.message, sender_index=agent_index)
            next_outbox[agent_index] = delivery.delivered_text
            orders.append(filled)
            note("intervention.apply", _delivery_detail(delivery))
            affect = None if vad_lexicon is None else vad_lexicon.score(decision.message)
            if affect is not None:
                note("text_affect.VADLexicon.score", _affect_detail(affect))
            records.append(
                AgentRecord(
                    round_index=round_index,
                    agent_index=agent_index,
                    role=role,
                    context=context,
                    generation=generation,
                    decision=decision,
                    delivery=delivery,
                    parse_succeeded=parse_succeeded,
                    position_after=positions[agent_index],
                    message_affect=affect,
                )
            )

        outbox = next_outbox
        step = market.clear(orders)
        if trace is not None:
            trace.emit(
                round_index=round_index,
                stage="market.SimpleMarket.clear",
                detail=(
                    f"imbalance {step.order_imbalance:+.4f}; price "
                    f"{step.previous_price:.4f} -> {step.price:.4f}; noise {step.noise_draw:+.4f}"
                ),
            )

    return SimulationResult(records=tuple(records), market_steps=tuple(market.history))
