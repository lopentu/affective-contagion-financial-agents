"""Tests for deterministic multi-model routing and provenance."""

import numpy as np
import pytest

from affective_contagion.activation import AffectDirection
from affective_contagion.generation import (
    BackendGeneration,
    ChatMessage,
    FloatVector,
    GenerationBackend,
    GenerationContext,
    RoutedDecisionGenerator,
)
from affective_contagion.messages import COGNITION_OUTPUT_REGEX, DECISION_OUTPUT_REGEX
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.roster import AgentSpec, ModelRoster, ModelSpec


class FakeBackend:
    """Backend with an observable activation."""

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        output_regex: str | None = None,
    ) -> BackendGeneration:
        """Return a stable decision and consume the deterministic seed contract."""
        del output_regex
        assert seed >= 0
        assert messages
        return BackendGeneration(
            '{"message":"ok","action":"HOLD","size":0}', np.array([2.0, 0.0]), 1
        )

    def read_activation(self, messages: tuple[ChatMessage, ...], *, layer: int) -> FloatVector:
        """Return an activation from the requested layer."""
        assert layer == 1
        assert messages
        return np.array([2.0, 0.0])


def test_router_supports_distinct_models_and_projects_matching_direction() -> None:
    """Each role should use only its assigned backend/direction metadata."""
    model = ModelSpec(key="q", repository="Qwen/test", revision="abc", activation_layer=1)
    roster = ModelRoster(
        models=(model,), agents=(AgentSpec(role="trader", model_key="q", system_prompt="Trade."),)
    )
    backends: dict[str, GenerationBackend] = {"q": FakeBackend()}
    router = RoutedDecisionGenerator(
        roster=roster,
        backends=backends,
        prompt_builder=FinancialPromptBuilder(),
        run_seed=7,
        affect_directions={"q": AffectDirection("q", "Qwen/test", "abc", 1, np.array([1.0, 0.0]))},
    )
    generated = router.generate(GenerationContext(2, 0, "trader", 100, 100, None, ()))
    assert generated.model_key == "q"
    assert generated.affect_score == 2


def test_router_rejects_direction_from_another_checkpoint() -> None:
    """Checkpoint metadata should prevent cross-space direction reuse."""
    model = ModelSpec(key="q", repository="Qwen/test", revision="abc", activation_layer=1)
    roster = ModelRoster(
        models=(model,), agents=(AgentSpec(role="trader", model_key="q", system_prompt="Trade."),)
    )
    backends: dict[str, GenerationBackend] = {"q": FakeBackend()}
    with pytest.raises(ValueError, match="metadata"):
        RoutedDecisionGenerator(
            roster=roster,
            backends=backends,
            prompt_builder=FinancialPromptBuilder(),
            run_seed=0,
            affect_directions={"q": AffectDirection("q", "Qwen/test", "other", 1, np.ones(2))},
        )


class RecordingBackend:
    """Backend that records the grammar constraint each call requests."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.regexes: list[str | None] = []

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        seed: int,
        output_regex: str | None = None,
    ) -> BackendGeneration:
        """Record the requested constraint and return a parseable decision."""
        assert seed >= 0
        assert messages
        self.regexes.append(output_regex)
        return BackendGeneration('{"message":"ok","action":"HOLD","size":0}')

    def read_activation(self, messages: tuple[ChatMessage, ...], *, layer: int) -> FloatVector:
        """Return an unused activation for protocol completeness."""
        assert messages
        assert layer >= 0
        return np.zeros(1)


def _router(backend: RecordingBackend, *, constrained: bool) -> RoutedDecisionGenerator:
    model = ModelSpec(key="q", repository="Qwen/test", constrained_decision_format=constrained)
    roster = ModelRoster(
        models=(model,), agents=(AgentSpec(role="trader", model_key="q", system_prompt="Trade."),)
    )
    backends: dict[str, GenerationBackend] = {"q": backend}
    return RoutedDecisionGenerator(
        roster=roster,
        backends=backends,
        prompt_builder=FinancialPromptBuilder(),
        run_seed=3,
    )


def test_constrained_roster_requests_the_decision_grammar() -> None:
    """An enabled spec must constrain decoding to the grammar the parser accepts."""
    backend = RecordingBackend()
    _router(backend, constrained=True).generate(
        GenerationContext(0, 0, "trader", 100, 100, None, ())
    )
    assert backend.regexes == [DECISION_OUTPUT_REGEX]


def test_unconstrained_roster_preserves_free_form_decoding() -> None:
    """An unconstrained roster decodes free-form, with no grammar."""
    backend = RecordingBackend()
    _router(backend, constrained=False).generate(
        GenerationContext(0, 0, "trader", 100, 100, None, ())
    )
    assert backend.regexes == [None]


def test_the_cognition_grammar_follows_the_prompt_builder() -> None:
    """Prompt, grammar and parser requirement must move together, or a run records empty outcomes.

    Three independent flags could request the cognition fields, constrain decoding to a grammar
    forbidding them, then parse without requiring them. Each mismatch turns a missing measurement
    into an apparent success, so the grammar is derived from the prompt builder.
    """
    for cognition, expected in (
        (False, DECISION_OUTPUT_REGEX),
        (True, COGNITION_OUTPUT_REGEX),
    ):
        backend = RecordingBackend()
        model = ModelSpec(key="q", repository="Qwen/test", constrained_decision_format=True)
        roster = ModelRoster(
            models=(model,),
            agents=(AgentSpec(role="trader", model_key="q", system_prompt="Trade."),),
        )
        backends: dict[str, GenerationBackend] = {"q": backend}
        generator = RoutedDecisionGenerator(
            roster=roster,
            backends=backends,
            prompt_builder=FinancialPromptBuilder(cognition_outcomes=cognition),
            run_seed=3,
        )
        assert generator.cognition_outcomes is cognition
        generator.generate(GenerationContext(0, 0, "trader", 100, 100, None, ()))
        assert backend.regexes == [expected]
