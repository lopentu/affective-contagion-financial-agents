"""Tests for immutable run persistence."""

import json
from pathlib import Path

from affective_contagion.config import ExperimentConfig
from affective_contagion.generation import DecisionGeneration, DecisionGenerator, GenerationContext
from affective_contagion.market import MarketParameters, SimpleMarket
from affective_contagion.provenance import RuntimeProvenance
from affective_contagion.roster import AgentSpec, ModelRoster, ModelSpec
from affective_contagion.simulation import run_simulation
from affective_contagion.storage import (
    append_run_status,
    create_run_directory,
    save_simulation_result,
)


class ZeroNoise:
    """Deterministic market stream."""

    def normal(self) -> float:
        """Return zero market innovation."""
        return 0.0


class HoldGenerator:
    """Stable simulation generator."""

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Return one valid hold decision."""
        assert context.round_index >= 0
        return DecisionGeneration(
            '{"message":"stable","action":"HOLD","size":0}', "q", "Qwen/test", "abc", 1, None, None
        )


def test_run_directory_is_non_overwriting_and_provenance_rich(tmp_path: Path) -> None:
    """Completed runs should be self-describing and non-overwriting."""
    config = ExperimentConfig(
        roles=("trader",), n_rounds=2, shock_round=1, shock_duration=1, shock_type="none"
    )
    roster = ModelRoster(
        models=(ModelSpec(key="q", repository="Qwen/test", revision="abc"),),
        agents=(AgentSpec(role="trader", model_key="q", system_prompt="Trade."),),
    )
    generator: DecisionGenerator = HoldGenerator()
    result = run_simulation(config, generator, SimpleMarket(MarketParameters(), ZeroNoise()))
    destination = create_run_directory(tmp_path, "run-001")
    provenance = RuntimeProvenance(
        git_commit="deadbeef",
        git_dirty=False,
        uv_lock_sha256="lockhash",
        python_version="3.11",
        operating_system="test-os",
        accelerator=None,
        torch_version=None,
        transformers_version=None,
    )
    save_simulation_result(
        destination, config=config, roster=roster, result=result, provenance=provenance
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["roster"]["models"][0]["revision"] == "abc"
    assert manifest["runtime_provenance"]["uv_lock_sha256"] == "lockhash"
    assert (destination / "a_internal.npy").exists()
    append_run_status(destination, "verified")
    assert '"status": "verified"' in (destination / "status.jsonl").read_text(encoding="utf-8")
