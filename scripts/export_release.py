"""Export model generations and their provenance as compressed, reproducible archives.

Reader exports contain readings, the supplied inboxes and a SHA-256 manifest. Complete-run
exports also retain run manifests, market histories and lifecycle records. Existing releases
are never overwritten.

Usage:
    uv run python scripts/export_release.py --readings READINGS.jsonl
        --pairs-root DIR --seeds 6 8 9 --name NAME --output-dir data/generations
        --repository Qwen/Qwen3-4B-Instruct-2507
        --revision cdbee75f17c01a7cc42f958dc650907174af0554
    uv run python scripts/export_release.py --runs-root outputs/runs/e51
        --expected-runs 288 --name e51-injected-rerun --output-dir data/generations
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from affective_contagion.run_release import export_runs

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    """Hash a file in chunks, so a large export does not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, required: bool = False) -> str:
    """Return Git output; required provenance checks propagate any command failure."""
    try:
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        if required:
            raise
        return ""


def _shock_round(run: Path) -> int:
    """Return the round at which the stimulus was delivered."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["experiment"]["shock_round"])


def _inbox(run: Path) -> list[dict[str, str]]:
    """Return every shock-round message in one run, as delivered and as decided.

    Both are kept. The delivered text is what a reader saw; the decision text is what the sender
    wrote. They are identical under the identity intervention and differ under any other, and a
    release that kept only one could not tell the two apart later.
    """
    shock = _shock_round(run)
    out: list[dict[str, str]] = []
    for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["round_index"] != shock:
            continue
        out.append(
            {
                "role": str(record["role"]),
                "agent_index": str(record["agent_index"]),
                "delivered_text": str(record["delivery"]["delivered_text"]),
                "decision_message": str(record["decision"]["message"]),
            }
        )
    return out


def _write_gz(path: Path, rows: list[dict]) -> None:
    """Write rows as gzipped JSONL, with mtime zeroed so the hash is reproducible.

    Without `mtime=0` gzip stamps the current time into the header, so two exports of identical
    content would hash differently and the manifest check would fail for no reason.
    """
    with gzip.GzipFile(path, "wb", mtime=0) as raw:
        for row in rows:
            raw.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))


def _collect_inboxes(pairs_root: Path, pair_prefix: str, seeds: list[int]) -> list[dict] | None:
    """Read every pair's shock-round inbox, or return None if a run is missing."""
    inboxes: list[dict] = []
    for seed in seeds:
        for arm, suffix in (("neutral", "neutral"), ("anxiety_like", "affective")):
            run = pairs_root / f"{pair_prefix}-{suffix}-s{seed}"
            if not run.exists():
                print(f"missing run for seed {seed} arm {arm}: {run}")
                return None
            inboxes.append({"seed": seed, "arm": arm, "messages": _inbox(run)})
    return inboxes


def main() -> int:
    """Export the readings and inboxes, then write the manifest that pins them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--readings", type=Path)
    parser.add_argument("--pairs-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--pair-prefix", default="s30")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="e36-role-context")
    parser.add_argument("--repository")
    parser.add_argument("--revision")
    arguments = parser.parse_args()

    if arguments.runs_root is not None:
        if arguments.expected_runs is None:
            parser.error("--runs-root requires --expected-runs")
        if any(
            (
                arguments.readings,
                arguments.pairs_root,
                arguments.seeds,
                arguments.repository,
                arguments.revision,
            )
        ):
            parser.error("--runs-root cannot be combined with isolated-reader arguments")
        destination = export_runs(
            arguments.runs_root,
            arguments.output_dir,
            name=arguments.name,
            expected_runs=arguments.expected_runs,
            commit=_git("rev-parse", "HEAD", required=True),
            dirty=bool(_git("status", "--porcelain", required=True)),
        )
        print(f"{destination}: {arguments.expected_runs} complete runs archived")
        return 0
    if not all(
        (
            arguments.readings,
            arguments.pairs_root,
            arguments.seeds,
            arguments.repository,
            arguments.revision,
        )
    ):
        parser.error(
            "isolated-reader export requires --readings, --pairs-root, --seeds, "
            "--repository and --revision"
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(x)
        for x in arguments.readings.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if not rows:
        print(f"no readings in {arguments.readings}")
        return 1

    readings_path = arguments.output_dir / f"{arguments.name}.jsonl.gz"
    inbox_path = arguments.output_dir / f"{arguments.name}-inboxes.jsonl.gz"
    manifest_path = arguments.output_dir / f"{arguments.name}.manifest.json"
    existing = [path for path in (readings_path, inbox_path, manifest_path) if path.exists()]
    if existing:
        print(f"refusing to overwrite an existing release: {', '.join(map(str, existing))}")
        return 1

    inboxes = _collect_inboxes(arguments.pairs_root, arguments.pair_prefix, arguments.seeds)
    if inboxes is None:
        return 1

    _write_gz(readings_path, rows)
    _write_gz(inbox_path, inboxes)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": arguments.name,
        "readings": {
            "file": readings_path.name,
            "sha256": _sha256(readings_path),
            "rows": len(rows),
        },
        "inboxes": {
            "file": inbox_path.name,
            "sha256": _sha256(inbox_path),
            "rows": len(inboxes),
        },
        "seeds": sorted(arguments.seeds),
        "repository": arguments.repository,
        "revision": arguments.revision,
        "exported_at_commit": _git("rev-parse", "HEAD"),
        "exported_from_dirty_tree": bool(_git("status", "--porcelain")),
        "synthetic": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    size = readings_path.stat().st_size / 1024
    print(f"{readings_path.name}: {len(rows)} readings, {size:.0f} KB")
    print(f"{inbox_path.name}: {len(inboxes)} pair-arms, {inbox_path.stat().st_size / 1024:.0f} KB")
    print(f"{manifest_path.name}: written at commit {manifest['exported_at_commit'][:12] or '?'}")
    if manifest["exported_from_dirty_tree"]:
        print("  NOTE: exported from a dirty tree, so the commit does not fully describe it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
