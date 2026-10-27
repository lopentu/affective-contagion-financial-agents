"""Definition of one experimental cell.

An `ExperimentConfig` is loaded before any model and defines the experiment: which condition,
which topology, who is treated, when the stimulus arrives, and which seed. Everything downstream
reads it and nothing writes it.

It is validated at the YAML boundary so a typo fails before a run starts. If `treated_role` is
unset, the stimulus is broadcast to every agent, which leaves cross-agent coupling unidentified.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Topology = Literal["full"]
ShockType = Literal["none", "affective", "neutral"]
DeliveryType = Literal["identity", "inject_message"]

DEFAULT_ROLES = (
    "sentiment_analyst",
    "fundamental_analyst",
    "bull_researcher",
    "bear_researcher",
    "trader",
    "risk_manager",
)


class ExperimentConfig(BaseModel):
    """Immutable model-independent settings for one simulation run.

    Pydantic is used because this schema is a boundary between external YAML/JSON and expensive
    experiment execution. Numerical kernels remain framework-independent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    roles: tuple[str, ...] = DEFAULT_ROLES
    topology: Topology = "full"
    n_rounds: int = Field(default=200, gt=0)
    shock_round: int = Field(default=80, ge=0)
    shock_duration: int = Field(default=3, gt=0)
    shock_type: ShockType = "affective"
    # Which role receives the stimulus. None broadcasts it to every agent, which is not causally
    # identified. When all agents are shocked at once, cross-agent coefficients absorb the shared
    # exposure, so estimated transmission cannot be told apart from correlated response to the same
    # text. Causal runs name a single treated role.
    treated_role: str | None = None
    message_intervention: DeliveryType = "identity"
    fundamental_value: float = Field(default=100.0, gt=0)
    price_impact: float = Field(default=0.02, ge=0)
    mean_reversion: float = Field(default=0.01, ge=0)
    noise_sigma: float = Field(default=0.003, ge=0)
    seed: int = 0
    var_maxlags: int = Field(default=3, gt=0)
    irf_horizon: int = Field(default=10, gt=0)

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require unique, non-empty role identifiers."""
        normalized = tuple(role.strip() for role in value)
        if (
            not normalized
            or any(not role for role in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            msg = "roles must be non-empty and unique"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_treated_role(self) -> Self:
        """Require a named treated role to exist, so a typo cannot silently broadcast."""
        if self.treated_role is not None and self.treated_role not in self.roles:
            msg = f"treated_role {self.treated_role!r} is not one of the configured roles"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_shock_window(self) -> Self:
        """Require the complete intervention window to fit inside the run."""
        if self.shock_round >= self.n_rounds:
            msg = "shock_round must be within the simulation"
            raise ValueError(msg)
        if self.shock_round + self.shock_duration > self.n_rounds:
            msg = "shock window must fit within the simulation"
            raise ValueError(msg)
        return self

    @property
    def n_agents(self) -> int:
        """Return the number of agents implied by the role list."""
        return len(self.roles)

    @property
    def run_name(self) -> str:
        """Return a stable human-readable run label."""
        intervention = {
            "identity": "fwOFF",
            "inject_message": "INJ",
        }[self.message_intervention]
        return f"{self.topology}_{self.shock_type}_{intervention}_seed{self.seed}"
