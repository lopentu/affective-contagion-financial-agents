"""Tests for validated YAML boundaries."""

from pathlib import Path

from affective_contagion.configuration import load_yaml_model
from affective_contagion.roster import ModelRoster


def test_repository_roster_yaml_loads_through_schema() -> None:
    """The primary model roster loads through its schema."""
    root = Path(__file__).resolve().parents[1]
    roster = load_yaml_model(root / "configs/models/qwen3_4b_shared.yaml", ModelRoster)
    assert roster.models[0].repository == "Qwen/Qwen3-4B-Instruct-2507"
    assert len(roster.agents) == 6
