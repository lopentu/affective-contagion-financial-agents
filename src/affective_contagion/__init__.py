"""Core components for controlled affective-contagion experiments."""

from affective_contagion.config import ExperimentConfig
from affective_contagion.messages import AgentDecision, parse_agent_decision
from affective_contagion.payload import FinancialPayload

__all__ = [
    "AgentDecision",
    "ExperimentConfig",
    "FinancialPayload",
    "parse_agent_decision",
]

__version__ = "0.1.0"
