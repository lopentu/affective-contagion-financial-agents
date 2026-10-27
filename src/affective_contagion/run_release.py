"""Archive complete synthetic runs with the manifests needed to recompute their outcomes.

The archive retains records, market history and lifecycle events. Pair identity comes from the
payload and seed in each manifest, so scenario names and sender roles need no filename convention.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_source_inventory(root: Path, expected: list[dict]) -> dict:
    """Compare complete source runs with an explicit catalogue of resolved planned inputs.

    Each expected item contains payload, experiment and roster. Filenames never decide which
    scenario, seed, arm, sender or model a run represents.
    """

    def key(manifest: dict) -> tuple[str, int, str]:
        experiment = manifest["experiment"]
        return (
            json.dumps(manifest["payload"], sort_keys=True),
            experiment["seed"],
            experiment["shock_type"],
        )

    catalogue = {key(item): item for item in expected}
    if len(catalogue) != len(expected) or not catalogue:
        msg = "planned input catalogue is empty or contains duplicate cells"
        raise ValueError(msg)
    seen = set()
    commits: dict[str, int] = {}
    dirty = 0
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest = _read_run(run)["manifest"]
        identity = key(manifest)
        if identity not in catalogue or identity in seen:
            msg = f"{run.name}: unexpected or duplicate scenario/seed/arm"
            raise ValueError(msg)
        for field in ("payload", "experiment", "roster"):
            if json.dumps(manifest[field], sort_keys=True) != json.dumps(
                catalogue[identity][field], sort_keys=True
            ):
                msg = f"{run.name}: {field} differs from planned inputs"
                raise ValueError(msg)
        seen.add(identity)
        provenance = manifest.get("runtime_provenance", {})
        commit = str(provenance.get("git_commit"))
        commits[commit] = commits.get(commit, 0) + 1
        dirty += provenance.get("git_dirty") is True
    if seen != set(catalogue):
        msg = f"planned inventory has {len(catalogue)} runs, found {len(seen)}"
        raise ValueError(msg)
    return {"runs": len(seen), "source_commits": commits, "dirty_source_runs": dirty}


def _read_run(run: Path) -> dict:
    """Validate one completed run before allowing it into a release."""
    names = ("manifest.json", "records.jsonl", "market.jsonl", "status.jsonl")
    sources = {name: (run / name).read_bytes() for name in names}
    manifest = json.loads(sources["manifest.json"])
    if manifest["status"] != "completed" or manifest["run_id"] != run.name:
        msg = f"{run.name}: incomplete run or inconsistent run ID"
        raise ValueError(msg)
    if manifest["payload"].get("synthetic") is not True:
        msg = f"{run.name}: only explicitly synthetic payloads may be exported"
        raise ValueError(msg)
    rows = {
        name.removesuffix(".jsonl"): [
            json.loads(line) for line in sources[name].splitlines() if line
        ]
        for name in names[1:]
    }
    if not rows["status"] or rows["status"][-1]["status"] != "completed":
        msg = f"{run.name}: lifecycle does not end in completed"
        raise ValueError(msg)
    experiment = manifest["experiment"]
    roles = experiment["roles"]
    if experiment["treated_role"] not in roles or len(set(roles)) != len(roles):
        msg = f"{run.name}: invalid treated role or duplicate roles"
        raise ValueError(msg)
    cells = [(row["round_index"], row["role"]) for row in rows["records"]]
    expected = {(r, role) for r in range(experiment["n_rounds"]) for role in roles}
    if len(cells) != len(expected) or set(cells) != expected:
        msg = f"{run.name}: missing or duplicate generation cells"
        raise ValueError(msg)
    market_rounds = [row["round_index"] for row in rows["market"]]
    if sorted(market_rounds) != list(range(experiment["n_rounds"])):
        msg = f"{run.name}: missing or duplicate market rounds"
        raise ValueError(msg)
    for row in rows["records"]:
        if roles[row["agent_index"]] != row["role"]:
            msg = f"{run.name}: role and agent index disagree"
            raise ValueError(msg)
        # The release promises no lexicon-derived tables. Keep raw text and recompute these
        # cached scores with a separately obtained NRC lexicon; source files remain untouched.
        if "message_affect" in row:
            row["message_affect"] = None
    manifest["stimulus_arousal_delta"] = None
    return {
        "run_id": run.name,
        "manifest": manifest,
        **rows,
        "source_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in sources.items()
        },
    }


def _validate_pairs(rows: list[dict]) -> None:
    """Require both arms, matching payloads and every treatment-invariant input."""
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        manifest = row["manifest"]
        experiment = manifest["experiment"]
        payload = json.dumps(manifest["payload"], sort_keys=True)
        key = (payload, experiment["seed"])
        arm = experiment["shock_type"]
        if arm not in {"neutral", "affective"} or arm in grouped[key]:
            msg = f"{row['run_id']}: invalid or duplicate arm"
            raise ValueError(msg)
        grouped[key][arm] = manifest
    for arms in grouped.values():
        if set(arms) != {"neutral", "affective"}:
            msg = "a run is missing its partner"
            raise ValueError(msg)
        left, right = arms["neutral"], arms["affective"]
        configs = [
            {key: value for key, value in item["experiment"].items() if key != "shock_type"}
            for item in (left, right)
        ]
        if configs[0] != configs[1]:
            msg = "paired experiment inputs differ beyond shock_type"
            raise ValueError(msg)
        for field in ("roster", "stimuli", "intervention_seed"):
            if left[field] != right[field]:
                msg = f"paired {field} inputs differ"
                raise ValueError(msg)


def export_runs(
    root: Path,
    output: Path,
    *,
    name: str,
    expected_runs: int,
    commit: str,
    dirty: bool = False,
) -> Path:
    """Write a hashed archive after validating all runs; refuse to overwrite an existing release."""
    if dirty or not commit:
        msg = "complete-run export requires a committed, clean source tree"
        raise ValueError(msg)
    if not name or Path(name).name != name or name in {".", ".."}:
        msg = "release name must be one path component"
        raise ValueError(msg)
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if expected_runs <= 0 or len(runs) != expected_runs:
        msg = f"expected {expected_runs} runs, found {len(runs)}"
        raise ValueError(msg)
    rows = [_read_run(run) for run in runs]
    _validate_pairs(rows)
    models = {
        (model["repository"], model["revision"])
        for row in rows
        for model in row["manifest"]["roster"]["models"]
    }
    if not models or any(not revision for _, revision in models):
        msg = "every model must have a pinned revision"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}-runs.jsonl.gz"
    destination = output / f"{name}.manifest.json"
    if destination.exists():
        msg = f"release {name} already exists"
        raise FileExistsError(msg)
    release = {
        "schema_version": 2,
        "kind": "complete_synthetic_runs",
        "name": name,
        "runs": {
            "file": archive.name,
            "sha256": "",
            "rows": len(rows),
        },
        "models": [
            {"repository": repository, "revision": revision}
            for repository, revision in sorted(models)
        ],
        "exported_at_commit": commit,
        "exported_from_dirty_tree": dirty,
        "synthetic": True,
        "transformation": (
            "JSON reserialized with sorted keys; cached message_affect and stimulus_arousal_delta "
            "set to null to exclude lexicon-derived scores; raw text and other source fields "
            "retained; original source file hashes recorded per run"
        ),
    }
    with TemporaryDirectory(prefix=f".{name}-", dir=output) as temporary:
        staged_archive = Path(temporary) / archive.name
        staged_manifest = Path(temporary) / destination.name
        with (
            staged_archive.open("xb") as stream,
            gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped,
        ):
            for row in rows:
                zipped.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        release["runs"]["sha256"] = hashlib.sha256(staged_archive.read_bytes()).hexdigest()
        staged_manifest.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
        _publish(staged_archive, staged_manifest, archive, destination)
    return destination


def _publish(staged_archive: Path, staged_manifest: Path, archive: Path, manifest: Path) -> None:
    """Publish complete files exclusively, with the manifest as the completion marker.

    A killed exporter may leave an archive without its manifest. A retry may reuse that archive
    only if its bytes match exactly. Existing complete releases are never overwritten.
    """
    created = False
    try:
        os.link(staged_archive, archive)
        created = True
    except FileExistsError:
        if archive.read_bytes() != staged_archive.read_bytes():
            msg = f"incomplete release has different archive bytes at {archive}"
            raise FileExistsError(msg) from None
    try:
        os.link(staged_manifest, manifest)
    except OSError:
        if created and archive.exists() and archive.samefile(staged_archive):
            archive.unlink()
        raise
