"""Immutable, provenance-rich simulation run persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from affective_contagion.config import ExperimentConfig
from affective_contagion.diagnostics import RunHealth
from affective_contagion.provenance import RuntimeProvenance
from affective_contagion.roster import ModelRoster
from affective_contagion.simulation import SimulationResult, StageTrace


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    msg = f"cannot serialize {type(value).__name__}"
    raise TypeError(msg)


def create_run_directory(root: Path, run_id: str) -> Path:
    """Create a new run directory and refuse to merge with or overwrite prior evidence."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        msg = "run_id must be one safe path segment"
        raise ValueError(msg)
    destination = root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def save_simulation_result(
    destination: Path,
    *,
    config: ExperimentConfig,
    roster: ModelRoster,
    result: SimulationResult,
    provenance: RuntimeProvenance,
    run_health: RunHealth | None = None,
    stimuli: Mapping[str, str] | None = None,
    payload: Mapping[str, object] | None = None,
    stimulus_arousal_delta: float | None = None,
    trace: StageTrace | None = None,
    intervention_seed: int | None = None,
    vad_lexicon_source: str | None = None,
) -> None:
    """Write one complete result only into an already-created empty directory.

    ``stimuli``, ``intervention_seed``, and ``vad_lexicon_source`` are recorded because they are
    treatment-defining inputs that live in the runner rather than in ``ExperimentConfig``. Without
    them the manifest cannot describe which stimulus a condition actually delivered.
    """
    unexpected = {item.name for item in destination.iterdir()} - {"status.jsonl"}
    if unexpected:
        msg = "destination contains unexpected files"
        raise FileExistsError(msg)
    manifest = {
        "schema_version": 3,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "completed",
        "run_id": destination.name,
        "runtime_provenance": asdict(provenance),
        "experiment": config.model_dump(mode="json"),
        "roster": roster.model_dump(mode="json"),
        "payload": dict(payload) if payload is not None else None,
        "stimuli": dict(stimuli) if stimuli is not None else None,
        "stimulus_arousal_delta": stimulus_arousal_delta,
        "delivered_stimulus": None if stimuli is None else stimuli.get(config.shock_type),
        "intervention_seed": intervention_seed,
        "vad_lexicon_source": vad_lexicon_source,
        "affect_direction_captured": result.affect_direction_captured,
        "parse_failure_rate": result.parse_failure_rate,
        "run_health": None if run_health is None else asdict(run_health),
        # A fixed label kept so new manifests match the released ones. No gain is computed here.
        "gain_definition": "max_h spectral norm of reduced-form cumulative IRF",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (destination / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in result.records:
            stream.write(json.dumps(asdict(row), default=_json_default, ensure_ascii=False) + "\n")
    with (destination / "market.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for step in result.market_steps:
            stream.write(json.dumps(asdict(step), default=_json_default) + "\n")
    np.save(destination / "a_internal.npy", result.affect_matrix(config.n_agents))
    np.save(destination / "positions.npy", result.position_matrix(config.n_agents))
    np.save(destination / "vad_arousal.npy", result.message_arousal_matrix(config.n_agents))
    np.save(destination / "prices.npy", np.asarray([step.price for step in result.market_steps]))
    if trace is not None:
        # Executed pipeline stages, in order. Consumed by scripts/replay_run.py so a replay shows
        # what actually ran rather than a hand-maintained description of it.
        with (destination / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for event in trace.events:
                stream.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def append_run_status(destination: Path, status: str, detail: str | None = None) -> None:
    """Append a UTC status event without deleting earlier lifecycle evidence."""
    event = {"at_utc": datetime.now(UTC).isoformat(), "status": status, "detail": detail}
    with (destination / "status.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
