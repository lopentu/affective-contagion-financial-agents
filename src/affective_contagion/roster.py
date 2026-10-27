"""Validated model and agent assignments for homogeneous or heterogeneous teams."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DType = Literal["auto", "float16", "bfloat16", "float32"]
Quantization = Literal["none", "4bit"]


class ModelSpec(BaseModel):
    """Immutable loading and decoding specification for one reusable backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    repository: str = Field(min_length=1)
    revision: str | None = None
    activation_layer: int | None = Field(default=None, ge=0)
    dtype: DType = "auto"
    device_map: str = "auto"
    quantization: Quantization = "none"
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    # Constrain decision generation to the parseable decision grammar. Defaults to False, which
    # keeps free-form decoding. Research runs should enable it, because an uncontrolled
    # parse-failure rate biases action outcomes. The filter's rewrites (`firewall.filter_message`)
    # are free text and are never constrained.
    constrained_decision_format: bool = False
    # Execute the checkpoint's own modeling code, which some architectures require. Off by default
    # because it runs third-party Python at load time. Pinning `revision` makes it auditable: the
    # code cannot change under a fixed commit, so it can be read once and trusted for that revision
    # only.
    trust_remote_code: bool = False
    # Force deterministic kernels. On by default because without it a rerun with an identical seed
    # can diverge, and the divergence cascades because one agent's message enters every peer's
    # inbox. Matched arms should be byte-identical until the treatment lands, so pre-treatment
    # divergence adds nuisance variance to the paired difference. Turn it off only to measure its
    # cost.
    deterministic: bool = True

    @field_validator("key", "repository", "device_map")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Normalize required identifiers."""
        return value.strip()


class AgentSpec(BaseModel):
    """One role and its assigned model backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)

    @field_validator("role", "model_key", "system_prompt")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Normalize agent text fields."""
        return value.strip()


class ModelRoster(BaseModel):
    """Versioned assignment of agents to one or more model backends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    models: tuple[ModelSpec, ...] = Field(min_length=1)
    agents: tuple[AgentSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """Require unique keys/roles and valid agent-to-model references."""
        model_keys = tuple(model.key for model in self.models)
        roles = tuple(agent.role for agent in self.agents)
        if len(set(model_keys)) != len(model_keys):
            msg = "model keys must be unique"
            raise ValueError(msg)
        if len(set(roles)) != len(roles):
            msg = "agent roles must be unique"
            raise ValueError(msg)
        missing = sorted({agent.model_key for agent in self.agents} - set(model_keys))
        if missing:
            msg = f"agents reference unknown model keys: {missing}"
            raise ValueError(msg)
        return self

    @property
    def models_by_key(self) -> dict[str, ModelSpec]:
        """Index models by stable routing key."""
        return {model.key: model for model in self.models}
