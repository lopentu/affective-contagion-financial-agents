"""Tests for portable financial prompt construction."""

from pathlib import Path

import pytest

from affective_contagion.configuration import load_yaml_model
from affective_contagion.generation import GenerationContext, InboundMessage
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.roster import AgentSpec, ModelRoster

ROSTERS = Path(__file__).resolve().parent.parent / "configs" / "models"
PRIMARY_ROSTER = ROSTERS / "qwen3_4b_shared.yaml"


def _agent(role: str) -> AgentSpec:
    """Return the primary model's configured agent for one role."""
    roster = load_yaml_model(PRIMARY_ROSTER, ModelRoster)
    return next(agent for agent in roster.agents if agent.role == role)


def test_prompt_orders_news_market_and_previous_round_provenance() -> None:
    """Shock salience and sender identity should be explicit without same-round leakage."""
    agent = _agent("trader")
    context = GenerationContext(
        80,
        4,
        "trader",
        95.0,
        100.0,
        "deposit outflow",
        (InboundMessage(0, "sentiment_analyst", "fear is rising"),),
    )
    messages = FinancialPromptBuilder().build(context, agent)
    user = messages[1].content
    assert user.index("INCOMING REPORT") < user.index("MARKET") < user.index("TEAM MESSAGES")
    # The delivery template must not add urgency framing the firewall is told to strip.
    assert "BREAKING" not in user
    assert "agent0(sentiment_analyst)" in user
    with pytest.raises(ValueError, match="does not match"):
        FinancialPromptBuilder().build(context, _agent("sentiment_analyst"))


def test_action_instruction_avoids_imitable_enum_separator() -> None:
    """A pipe-separated enum invites values like "SELL|0.2" that the strict parser rejects.

    Those rejections cluster in post-shock rounds, so they would inject a treatment-correlated
    HOLD-at-size-0 artifact into action-level outcomes.
    """
    agent = _agent("trader")
    context = GenerationContext(0, 4, "trader", 100.0, 100.0, None, ())
    system = FinancialPromptBuilder().build(context, agent)[0].content
    assert "BUY|SELL|HOLD" not in system
    assert "|" not in system
    for action in ("BUY", "SELL", "HOLD"):
        assert action in system


def _names_own_role(agent: AgentSpec) -> bool:
    """Report whether an agent's prompt contains its own role name, such as "risk manager"."""
    return agent.role.replace("_", " ") in agent.system_prompt.lower()


def test_no_configured_prompt_names_its_own_role() -> None:
    """The paper compares task instructions, so a reader's prompt must not carry its role name.

    Role names still label teammates' inbox messages.
    """
    rosters = sorted(ROSTERS.rglob("*.yaml"))
    assert rosters
    named = [
        f"{path.relative_to(ROSTERS)}: {agent.role}"
        for path in rosters
        for agent in load_yaml_model(path, ModelRoster).agents
        if _names_own_role(agent)
    ]
    assert named == []


def test_role_name_check_detects_a_named_role() -> None:
    """The check above must detect a prompt that opens with its role name."""
    named = AgentSpec(
        role="trader",
        model_key="qwen3-4b",
        system_prompt="You are the trader. Use the team's evidence to choose a bounded trade.",
    )
    assert _names_own_role(named)
    assert not _names_own_role(_agent("trader"))
