"""Interfaces separating prompts, model inference, routing, and simulation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from affective_contagion.activation import AffectDirection
from affective_contagion.messages import COGNITION_OUTPUT_REGEX, DECISION_OUTPUT_REGEX
from affective_contagion.roster import AgentSpec, ModelRoster

FloatVector = NDArray[np.float64]
ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """Previous-round message with explicit sender provenance."""

    sender_index: int
    sender_role: str
    text: str


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Complete observable context for one agent decision."""

    round_index: int
    agent_index: int
    role: str
    price: float
    fundamental_value: float
    news: str | None
    inbox: tuple[InboundMessage, ...]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Model-neutral chat message."""

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class BackendGeneration:
    """Raw backend output and optional captured residual-stream activation."""

    text: str
    activation: FloatVector | None = None
    activation_layer: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionGeneration:
    """Routed generation with model provenance and optional affect projection."""

    raw_output: str
    model_key: str
    repository: str
    revision: str | None
    generation_seed: int
    activation_layer: int | None
    affect_score: float | None
    prompt_messages: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class MessageDelivery:
    """Delivered text and intervention audit metadata."""

    delivered_text: str
    intervention_name: str
    intervention_seed: int | None = None
    preservation_validated: bool | None = None


class GenerationBackend(Protocol):
    """One loaded model backend reusable by any number of agent roles."""

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        output_regex: str | None = None,
    ) -> BackendGeneration:
        """Generate text and optionally capture the configured activation.

        When ``output_regex`` is set the backend must restrict decoding to that grammar. Callers
        that need free text, such as firewall rewrites, leave it unset.
        """
        ...

    def read_activation(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        layer: int,
    ) -> FloatVector:
        """Read the last input-token residual stream without generation."""
        ...


class PromptBuilder(Protocol):
    """Turn typed context and role definition into model-neutral messages."""

    def build(self, context: GenerationContext, agent: AgentSpec) -> tuple[ChatMessage, ...]:
        """Build one chat request."""
        ...


class DecisionGenerator(Protocol):
    """Generator that returns rich model provenance for one agent context."""

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Generate raw output for strict decision parsing."""
        ...


class MessageIntervention(Protocol):
    """Transformation applied only to messages delivered to recipients."""

    def apply(self, message: str, *, sender_index: int) -> MessageDelivery:
        """Return delivered text and intervention metadata.

        ``sender_index`` identifies the agent whose message is being delivered, so an
        intervention can route to that agent's own model instead of an arbitrary one.
        """
        ...


def derive_generation_seed(run_seed: int, round_index: int, agent_index: int) -> int:
    """Derive a stable independent 63-bit seed without consuming another RNG stream."""
    material = f"{run_seed}:{round_index}:{agent_index}".encode()
    digest = hashlib.blake2b(material, digest_size=8, person=b"affectgen").digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


class RoutedDecisionGenerator:
    """Route each agent to a shared or distinct model backend and affect direction."""

    def __init__(
        self,
        *,
        roster: ModelRoster,
        backends: Mapping[str, GenerationBackend],
        prompt_builder: PromptBuilder,
        run_seed: int,
        affect_directions: dict[str, AffectDirection] | None = None,
    ) -> None:
        """Validate all model assignments before inference begins."""
        expected = set(roster.models_by_key)
        if set(backends) != expected:
            msg = f"backend keys {sorted(backends)} must equal roster keys {sorted(expected)}"
            raise ValueError(msg)
        directions = affect_directions or {}
        unknown_directions = set(directions) - expected
        if unknown_directions:
            msg = f"directions reference unknown model keys: {sorted(unknown_directions)}"
            raise ValueError(msg)
        for key, direction in directions.items():
            model = roster.models_by_key[key]
            if (
                direction.model_key != key
                or direction.repository != model.repository
                or direction.revision != model.revision
                or direction.layer != model.activation_layer
            ):
                msg = f"affect direction metadata does not match model {key!r}"
                raise ValueError(msg)
        self._roster = roster
        self._backends = dict(backends)
        self._prompt_builder = prompt_builder
        self._run_seed = run_seed
        self._directions = directions

    @property
    def cognition_outcomes(self) -> bool:
        """Report whether this generator asks for and requires the RQ2 cognition fields.

        Read from the prompt builder rather than stored separately, so the instruction text, the
        decoding grammar and the parser's `require_cognition` all follow one switch. Independent
        flags would let a run request the fields, constrain output to a grammar that forbids them,
        and then parse without requiring them, which records a silently empty outcome.
        """
        return getattr(self._prompt_builder, "cognition_outcomes", False)

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Generate with the routed backend and project its model-specific activation."""
        if not 0 <= context.agent_index < len(self._roster.agents):
            msg = f"agent index outside roster: {context.agent_index}"
            raise IndexError(msg)
        agent = self._roster.agents[context.agent_index]
        model = self._roster.models_by_key[agent.model_key]
        messages = self._prompt_builder.build(context, agent)
        # The grammar is derived from the prompt builder, never chosen independently. Asking for the
        # cognition fields while constraining output to the three-field grammar would make every
        # generation unparseable, so the two cannot be set apart.
        grammar = COGNITION_OUTPUT_REGEX if self.cognition_outcomes else DECISION_OUTPUT_REGEX
        seed = derive_generation_seed(self._run_seed, context.round_index, context.agent_index)
        generated = self._backends[agent.model_key].generate(
            messages,
            seed=seed,
            output_regex=grammar if model.constrained_decision_format else None,
        )

        score: float | None = None
        direction = self._directions.get(agent.model_key)
        if direction is not None:
            if generated.activation is None:
                msg = f"backend {agent.model_key!r} did not return its configured activation"
                raise ValueError(msg)
            if generated.activation_layer != direction.layer:
                msg = "captured activation layer does not match affect direction"
                raise ValueError(msg)
            score = direction.project(generated.activation)

        return DecisionGeneration(
            raw_output=generated.text,
            model_key=agent.model_key,
            repository=model.repository,
            revision=model.revision,
            generation_seed=seed,
            activation_layer=generated.activation_layer,
            affect_score=score,
            prompt_messages=messages,
        )
