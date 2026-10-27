"""The committed sample run must stay loadable as the pipeline changes.

Otherwise a schema change silently leaves `docs/examples/` describing a pipeline that no longer
exists, and its replay page fails to open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from affective_contagion.replay import load_run

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "docs/examples/sample-run"


def test_sample_run_still_loads_through_the_replay_builder() -> None:
    """The replay builder must consume the committed sample without special-casing."""
    data = load_run(SAMPLE)
    agent_steps = [step for step in data["steps"] if step["kind"] == "agent"]
    assert data["traced"] is True, "the sample must carry a recorded trace, not a reconstruction"
    assert len(agent_steps) == 72
    assert data["treated_role"] == "sentiment_analyst"
    assert data["roles"]
    assert len(data["roles"]) == 6


def test_sample_run_exposes_prompts_and_raw_output() -> None:
    """Inputs and outputs are the point of the sample; both must be present."""
    data = load_run(SAMPLE)
    assert data["system_prompts"], "system prompts must be carried"
    step = next(s for s in data["steps"] if s["kind"] == "agent" and s["treated"])
    for field in ("user_prompt", "raw", "message", "stimulus"):
        assert step[field], f"{field} is empty in the treated step"
    assert step["trace"], "the treated step must carry its executed stages"


def test_sample_run_manifest_records_its_own_limitations() -> None:
    """The manifest records the run's parse health and that no activation was measured."""
    manifest = json.loads((SAMPLE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parse_failure_rate"] == pytest.approx(0.0)
    # No affect direction was loaded, so the activation series is all-NaN, and the manifest says so.
    assert manifest["affect_direction_captured"] is False
    assert manifest["payload"]["synthetic"] is True
    assert manifest["experiment"]["shock_duration"] == 1


def test_sample_replay_page_is_present_and_self_contained() -> None:
    """The committed page must open with no network access."""
    page = (ROOT / "docs/examples/sample-run.html").read_text(encoding="utf-8")
    assert "__RUN_DATA__" not in page, "the template placeholder was not substituted"
    for forbidden in ("http://", "https://", "<script src", '<link rel="stylesheet"'):
        assert forbidden not in page, f"page references external resource: {forbidden}"
