"""Tests for experiment settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from affective_contagion.config import ExperimentConfig
from affective_contagion.configuration import load_yaml_model


def test_config_defaults_are_explicit() -> None:
    """The defaults are the full-run settings."""
    config = ExperimentConfig()
    assert config.n_agents == 6
    assert config.n_rounds == 200
    assert config.shock_round == 80
    assert config.run_name == "full_affective_fwOFF_seed0"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_rounds": 0}, "n_rounds"),
        ({"shock_round": 200}, "shock_round"),
        ({"shock_duration": 121}, "shock window"),
        ({"fundamental_value": 0}, "fundamental_value"),
        ({"roles": ("trader", "trader")}, "roles"),
        ({"message_intervention": "silent_failure"}, "message_intervention"),
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, object], message: str) -> None:
    """Invalid settings must fail before a costly inference run."""
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(changes)


def test_unknown_config_field_is_rejected() -> None:
    """Misspelled external config keys must not silently use defaults."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentConfig.model_validate({"n_round": 100})


def test_config_is_immutable() -> None:
    """Resolved settings cannot drift during a run."""
    config = ExperimentConfig()
    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, "seed", 3)  # noqa: B010 - exercise Pydantic's runtime frozen guard.


def test_no_communication_condition_has_unambiguous_run_name() -> None:
    """Delivery controls should not be encoded through ad hoc runner changes."""
    config = ExperimentConfig(message_intervention="inject_message")
    assert config.run_name == "full_affective_INJ_seed0"


def test_run_configs_deliver_one_round_to_one_role() -> None:
    """Every run config delivers the report once, to one named role.

    An unset treated role would send the report to every agent.
    """
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs/experiments").rglob("*.yaml"))
    assert configs
    for path in configs:
        config = load_yaml_model(path, ExperimentConfig)
        assert config.shock_duration == 1, path.name
        assert config.treated_role is not None, path.name
        assert config.treated_role in config.roles, path.name
