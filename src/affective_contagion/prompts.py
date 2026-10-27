"""Render what an agent sees on a given round.

The builder sits between the simulation loop and the model. It takes a `GenerationContext`
(stimulus if this agent is treated, current price, last round's inbox), renders the chat request,
and asks for the decision as JSON.

The delivery template adds no rhetoric of its own. The stimulus section is headed `[INCOMING
REPORT]` rather than `[BREAKING NEWS]`, because "BREAKING" is urgency framing, the firewall is
instructed to remove urgency framing, and a template that injected it would add treatment the
experiment did not intend.

Role prompts are shared by every condition, so they cannot confound a contrast. They can suppress
an effect, for example through wording that tells an agent to discount rhetoric.
"""

from __future__ import annotations

from affective_contagion.generation import ChatMessage, GenerationContext
from affective_contagion.roster import AgentSpec

# The action slot is described in prose rather than as a pipe-separated enum. Writing it as
# "BUY|SELL|HOLD" invites the model to imitate the separator and emit values such as "SELL|0.2",
# which the strict parser rejects. Those rejections fall back to HOLD at size 0, and if they
# cluster in post-shock rounds they add a treatment-correlated action artifact.
ACTION_INSTRUCTION = (
    "If an INCOMING REPORT section is present, react to it explicitly from your role "
    "perspective. "
    'Reply with one JSON object only: {"message":"<team message under 60 words>",'
    '"action":"<one of BUY, SELL, HOLD>","size":<number from 0 to 1>}. '
    "The action value must be exactly one of the three words BUY, SELL, or HOLD, "
    "and size must be a bare number."
)

# Adds the three RQ2 outcomes to the requested JSON. A separate constant keeps the base instruction
# unchanged for runs that do not request these outcomes.
COGNITION_ACTION_INSTRUCTION = (
    "If an INCOMING REPORT section is present, react to it explicitly from your role "
    "perspective. "
    'Reply with one JSON object only: {"message":"<team message under 60 words>",'
    '"action":"<one of BUY, SELL, HOLD>","size":<number from 0 to 1>,'
    '"expected_return":<percent change you expect in this asset over the time horizon in '
    "the report, or over the current session if none is stated; negative for a fall>,"
    '"confidence":<number from 0 to 1: how certain you are of that forecast>,'
    '"downside_risk":<number from 0 to 1: your probability that this asset falls over that same '
    "horizon>}. "
    "The action value must be exactly one of the three words BUY, SELL, or HOLD, "
    "and every numeric value must be a bare number between -99.99 and 99.99."
)


class FinancialPromptBuilder:
    """Build model-neutral chat messages from a typed simulation context.

    ``cognition_outcomes`` asks for expected return, confidence and downside risk alongside the
    action. It is off by default because turning it on changes the prompt every agent sees, so
    results are not comparable with any run that had it off.
    """

    def __init__(self, *, cognition_outcomes: bool = False) -> None:
        """Record whether the RQ2 outcomes are requested."""
        self._cognition_outcomes = cognition_outcomes

    @property
    def cognition_outcomes(self) -> bool:
        """Report whether this builder asks for the RQ2 outcomes."""
        return self._cognition_outcomes

    def build(self, context: GenerationContext, agent: AgentSpec) -> tuple[ChatMessage, ...]:
        """Render news first, then market state and previous-round team messages."""
        if context.role != agent.role:
            msg = f"context role {context.role!r} does not match routed role {agent.role!r}"
            raise ValueError(msg)
        instruction = (
            COGNITION_ACTION_INSTRUCTION if self._cognition_outcomes else ACTION_INSTRUCTION
        )
        system = ChatMessage(role="system", content=f"{agent.system_prompt} {instruction}")
        sections: list[str] = []
        if context.news:
            # Not "BREAKING", because the delivery template must not add urgency framing that
            # FIREWALL_INSTRUCTION is told to strip from messages.
            sections.append(f"[INCOMING REPORT]\n{context.news}")
        deviation = (context.price / context.fundamental_value - 1) * 100
        sections.append(
            f"[MARKET]\nCurrent price: {context.price:.4f}. "
            f"Known fundamental value: {context.fundamental_value:.4f}. "
            f"Deviation: {deviation:+.3f}%."
        )
        if context.inbox:
            rendered_inbox = "\n".join(
                f"agent{message.sender_index}({message.sender_role}): {message.text}"
                for message in context.inbox
            )
            sections.append(f"[TEAM MESSAGES: PREVIOUS ROUND]\n{rendered_inbox}")
        return system, ChatMessage(role="user", content="\n\n".join(sections))
