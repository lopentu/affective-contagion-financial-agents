"""End-to-end tests for model-independent orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from affective_contagion.config import ExperimentConfig
from affective_contagion.diagnostics import evaluate_run_health
from affective_contagion.generation import (
    DecisionGeneration,
    GenerationContext,
    InboundMessage,
    MessageDelivery,
)
from affective_contagion.market import MarketParameters, SimpleMarket
from affective_contagion.simulation import _fill_within_position_limit, run_simulation
from affective_contagion.text_affect import VADLexicon


class ZeroNoise:
    """Deterministic market stream."""

    def normal(self) -> float:
        """Return no random innovation."""
        return 0.0


@dataclass
class RecordingGenerator:
    """Deterministic fake backend that records every context."""

    contexts: list[GenerationContext] = field(default_factory=list)

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Return a valid HOLD decision labeled by sender."""
        self.contexts.append(context)
        raw = json.dumps(
            {
                "message": f"message-from-{context.agent_index}-round-{context.round_index}",
                "action": "HOLD",
                "size": 0,
            }
        )
        return DecisionGeneration(
            raw_output=raw,
            model_key="fake",
            repository="fake/repo",
            revision="test",
            generation_seed=context.agent_index,
            activation_layer=None,
            affect_score=None,
        )


class PrefixIntervention:
    """Observable deterministic delivery transformation."""

    def __init__(self) -> None:
        """Start with no routed senders recorded."""
        self.senders: list[int] = []

    def apply(self, message: str, *, sender_index: int) -> MessageDelivery:
        """Prefix delivered text and record which sender was routed."""
        self.senders.append(sender_index)
        return MessageDelivery(f"neutral::{message}", "test_prefix", preservation_validated=True)


def test_messages_arrive_only_on_next_round_from_every_other_agent() -> None:
    """No recipient should observe same-round output or its own message."""
    config = ExperimentConfig(
        roles=("sender", "middle", "recipient"),
        topology="full",
        n_rounds=2,
        shock_round=1,
        shock_duration=1,
        shock_type="none",
    )
    generator = RecordingGenerator()
    market = SimpleMarket(MarketParameters(), ZeroNoise())
    result = run_simulation(
        config,
        generator,
        market,
        intervention=PrefixIntervention(),
        vad_lexicon=VADLexicon({"message": (0.5, 0.6, 0.5)}),
    )

    first_round = generator.contexts[:3]
    second_round = generator.contexts[3:]
    assert all(not context.inbox for context in first_round)
    sent = {
        0: InboundMessage(0, "sender", "neutral::message-from-0-round-0"),
        1: InboundMessage(1, "middle", "neutral::message-from-1-round-0"),
        2: InboundMessage(2, "recipient", "neutral::message-from-2-round-0"),
    }
    assert second_round[0].inbox == (sent[1], sent[2])
    assert second_round[1].inbox == (sent[0], sent[2])
    assert second_round[2].inbox == (sent[0], sent[1])
    assert len(result.market_steps) == 2
    assert result.parse_failure_rate == 0
    assert result.message_arousal_matrix(3)[0, 0] == 0.6
    health = evaluate_run_health(
        result,
        shock_round=1,
        shock_duration=1,
        shock_keywords=("message-from",),
    )
    assert health.hold_rate == 1
    assert health.shock_window_keyword_hits == 3


def test_shock_window_and_parse_failure_are_auditable() -> None:
    """News timing and malformed outputs should be visible in records."""
    config = ExperimentConfig(
        roles=("analyst",),
        n_rounds=3,
        shock_round=1,
        shock_duration=1,
        shock_type="affective",
    )

    class MalformedOnShock:
        """Return invalid output exactly when news is present."""

        def generate(self, context: GenerationContext) -> DecisionGeneration:
            """Generate valid JSON except in the treatment window."""
            raw = "malformed" if context.news else '{"message":"stable","action":"HOLD","size":0}'
            return DecisionGeneration(raw, "fake", "fake/repo", None, 0, None, None)

    market = SimpleMarket(MarketParameters(), ZeroNoise())
    result = run_simulation(
        config,
        MalformedOnShock(),
        market,
        shocks={"affective": "synthetic shock"},
    )
    assert [record.context.news for record in result.records] == [
        None,
        "synthetic shock",
        None,
    ]
    assert result.records[1].parse_succeeded is False
    assert result.records[1].decision.action == "HOLD"
    assert result.parse_failure_rate == 1 / 3
    assert result.affect_matrix(1).shape == (3, 1)


@dataclass
class SaturatingGenerator:
    """Fake generator that always trades at full size so the position limit binds."""

    action: str = "BUY"
    size: float = 1.0

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Return a maximal order for every agent and round."""
        raw = json.dumps({"message": "saturate", "action": self.action, "size": self.size})
        return DecisionGeneration(
            raw_output=raw,
            model_key="fake",
            repository="fake/repo",
            revision="test",
            generation_seed=context.agent_index,
            activation_layer=None,
            affect_score=None,
        )


def test_market_receives_only_the_fill_the_position_limit_allows() -> None:
    """The position bound and the market must see the same trade.

    If the position were clamped while the unclamped order went to the market, an agent already at
    the limit could keep pushing price. `positions.npy` and the price path would then disagree, and
    any outcome built on both would be incoherent.
    """
    assert _fill_within_position_limit(0.9, 0.5) == pytest.approx(0.1)
    assert _fill_within_position_limit(-0.9, -0.5) == pytest.approx(-0.1)
    assert _fill_within_position_limit(1.0, 0.5) == pytest.approx(0.0)
    assert _fill_within_position_limit(-1.0, -1.0) == pytest.approx(0.0)
    # Unconstrained orders pass through untouched, and reducing exposure is always allowed.
    assert _fill_within_position_limit(0.2, 0.5) == pytest.approx(0.5)
    assert _fill_within_position_limit(1.0, -0.5) == pytest.approx(-0.5)


def test_positions_stay_consistent_with_submitted_orders() -> None:
    """Every recorded position change must equal the order the market cleared."""
    config = ExperimentConfig(
        roles=("a", "b"),
        topology="full",
        n_rounds=6,
        shock_round=1,
        shock_duration=1,
        shock_type="none",
        var_maxlags=1,
        irf_horizon=2,
    )
    market = SimpleMarket(MarketParameters(100.0, 0.02, 0.01, 0.0), ZeroNoise())
    # Always BUY at full size, so both agents saturate the position limit within two rounds.
    result = run_simulation(config, SaturatingGenerator(), market)
    positions = result.position_matrix(config.n_agents)
    assert float(np.nanmax(np.abs(positions))) <= 1.0
    for agent in range(config.n_agents):
        series = positions[:, agent]
        assert series[-1] == pytest.approx(1.0)
        assert np.all(np.diff(series) >= -1e-12)


def test_only_the_treated_role_reads_the_stimulus() -> None:
    """Naming a treated role must confine the stimulus to that one agent.

    Broadcasting to every agent leaves cross-agent coupling unidentified: simultaneous movement is
    then equally explained by shared exposure to the same text as by transmission between agents.
    """
    config = ExperimentConfig(
        roles=("a", "b", "c"),
        topology="full",
        n_rounds=3,
        shock_round=1,
        shock_duration=1,
        shock_type="affective",
        treated_role="b",
        var_maxlags=1,
        irf_horizon=2,
    )
    generator = RecordingGenerator()
    market = SimpleMarket(MarketParameters(100.0, 0.02, 0.01, 0.0), ZeroNoise())
    run_simulation(config, generator, market, shocks={"affective": "STIMULUS"})
    saw_news = {(c.round_index, c.role) for c in generator.contexts if c.news is not None}
    assert saw_news == {(1, "b")}


def test_no_treated_role_preserves_broadcast_behaviour() -> None:
    """With no treated role named, the stimulus is broadcast to every agent."""
    config = ExperimentConfig(
        roles=("a", "b"),
        topology="full",
        n_rounds=2,
        shock_round=0,
        shock_duration=1,
        shock_type="affective",
        var_maxlags=1,
        irf_horizon=2,
    )
    generator = RecordingGenerator()
    market = SimpleMarket(MarketParameters(100.0, 0.02, 0.01, 0.0), ZeroNoise())
    run_simulation(config, generator, market, shocks={"affective": "STIMULUS"})
    assert {c.role for c in generator.contexts if c.news is not None} == {"a", "b"}
