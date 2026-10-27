"""Every shipped configuration file must validate against the schema that consumes it.

The schema classes are unit-tested against synthetic dictionaries elsewhere. These tests check that
the files in `configs/` themselves load, so a malformed config fails here rather than at run time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from affective_contagion.config import ExperimentConfig
from affective_contagion.configuration import load_yaml_model
from affective_contagion.payload import FinancialPayload, matched_realizations
from affective_contagion.roster import ModelRoster

ROOT = Path(__file__).resolve().parents[1]

SCHEMAS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("configs/models", ModelRoster),
    ("configs/experiments", ExperimentConfig),
    ("configs/scenarios", FinancialPayload),
)


def _cases() -> list[tuple[Path, type[BaseModel]]]:
    cases = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        relative = path.relative_to(ROOT).as_posix()
        schema = next((s for prefix, s in SCHEMAS if relative.startswith(prefix)), None)
        assert schema is not None, f"{relative} has no declared schema"
        cases.append((path, schema))
    return cases


@pytest.mark.parametrize(("path", "schema"), _cases(), ids=lambda v: getattr(v, "name", ""))
def test_shipped_config_validates(path: Path, schema: type[BaseModel]) -> None:
    """Each committed config loads through its schema with no extra or missing fields."""
    assert load_yaml_model(path, schema) is not None


def test_every_config_directory_is_covered() -> None:
    """A new configs/ subdirectory must be given a schema rather than silently skipped."""
    directories = {
        p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").iterdir() if p.is_dir()
    }
    declared = {prefix.removeprefix("configs/") for prefix, _ in SCHEMAS}
    assert {d.removeprefix("configs/") for d in directories} <= declared


def test_shipped_scenarios_are_labelled_synthetic_and_render_matched_arms() -> None:
    """Scenarios must be labelled synthetic and must produce a matched treatment pair."""
    scenarios = sorted((ROOT / "configs/scenarios").glob("*.yaml"))
    assert scenarios, "no scenario is shipped"
    for path in scenarios:
        payload = load_yaml_model(path, FinancialPayload)
        assert payload.synthetic is True, path.name
        arms = matched_realizations(payload)
        assert payload.fact_block() in arms["neutral"]
        assert payload.fact_block() in arms["anxiety_like"]
