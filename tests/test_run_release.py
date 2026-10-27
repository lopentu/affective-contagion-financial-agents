"""A released run must retain the evidence needed to catch sender and price mistakes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from affective_contagion.run_release import export_runs, validate_source_inventory


def _fixture(root: Path) -> None:
    """Write complete manifests with different senders and scenario payloads at the same seed."""
    for scenario, sender in (("one", "reader"), ("two", "trader")):
        for arm in ("neutral", "affective"):
            run = root / f"{scenario}-{arm}"
            run.mkdir(parents=True)
            manifest = {
                "status": "completed",
                "run_id": run.name,
                "payload": {"asset": scenario, "synthetic": True},
                "experiment": {
                    "roles": ["reader", "trader"],
                    "treated_role": sender,
                    "shock_round": 1,
                    "n_rounds": 3,
                    "seed": 7,
                    "shock_type": arm,
                },
                "roster": {"models": [{"repository": "fixture/model", "revision": "abc123"}]},
                "stimuli": {"neutral": "calm", "affective": "alarmed"},
                "intervention_seed": 42,
                "stimulus_arousal_delta": 0.1,
                "runtime_provenance": {"git_commit": "original", "git_dirty": True},
            }
            records = [
                {
                    "round_index": r,
                    "role": role,
                    "agent_index": index,
                    "context": {
                        "round_index": r,
                        "agent_index": index,
                        "role": role,
                        "price": 100 + r,
                        "fundamental_value": 100,
                        "news": "synthetic news" if r == 1 and role == sender else None,
                        "inbox": [
                            {
                                "sender_index": 1 - index,
                                "sender_role": "peer",
                                "text": "delivered inbox",
                            }
                        ],
                    },
                    "generation": {"raw_output": "original raw output"},
                    "decision": {"message": arm, "action": "HOLD", "size": 0},
                    "delivery": {"delivered_text": "injected text"},
                    "position_after": 0,
                    "parse_succeeded": r != 2,
                    "message_affect": {"arousal": 0.2},
                }
                for r in range(3)
                for index, role in enumerate(("reader", "trader"))
            ]
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for name, rows in (
                ("records", records),
                ("market", [{"round_index": r, "price": 100 + r} for r in range(3)]),
                ("status", [{"status": "started"}, {"status": "completed"}]),
            ):
                (run / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
                )


def _export(root: Path, output: Path) -> Path:
    return export_runs(root, output, name="test", expected_runs=4, commit="export-commit")


def test_inventory_checks_planned_inputs(tmp_path: Path) -> None:
    """Correct run counts cannot hide a wrong sender or omitted planned scenario."""
    root = tmp_path / "runs"
    _fixture(root)
    expected = [json.loads(path.read_text()) for path in sorted(root.glob("*/manifest.json"))]
    assert validate_source_inventory(root, expected)["runs"] == 4
    expected[0]["experiment"]["treated_role"] = "trader"
    with pytest.raises(ValueError, match="experiment differs"):
        validate_source_inventory(root, expected)


def test_e51_preflight_builds_the_actual_config_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read real configuration files, but let no test inspect substantive run outputs."""
    entry = runpy.run_path("scripts/e51_preflight.py")

    def inspect_catalogue(root: Path, expected: list[dict]) -> dict:
        assert root == tmp_path
        assert len(expected) == 288
        assert len({json.dumps(item["payload"], sort_keys=True) for item in expected}) == 4
        assert {item["experiment"]["seed"] for item in expected} == set(range(300, 336))
        assert len({item["experiment"]["treated_role"] for item in expected}) == 6
        return {"runs": 288}

    monkeypatch.setitem(entry["main"].__globals__, "validate_source_inventory", inspect_catalogue)
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight", "--runs", str(tmp_path), "--output", str(tmp_path / "inventory.json")],
    )
    assert entry["main"]() == 0


def test_dirty_export_is_refused(tmp_path: Path) -> None:
    """A release intended to leave the checkout must trace to committed exporter code."""
    with pytest.raises(ValueError, match="committed, clean"):
        export_runs(
            tmp_path,
            tmp_path / "release",
            name="test",
            expected_runs=4,
            commit="commit",
            dirty=True,
        )


def test_git_status_failure_aborts_full_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable Git status must never be interpreted as a clean tree."""
    entry = runpy.run_path("scripts/export_release.py")

    def fail_status(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1] == "status":
            raise subprocess.CalledProcessError(128, args)
        assert args == ["git", "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(args, 0, stdout="commit\n")

    monkeypatch.setattr(subprocess, "run", fail_status)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--runs-root",
            str(tmp_path),
            "--expected-runs",
            "4",
            "--output-dir",
            str(tmp_path / "release"),
        ],
    )
    with pytest.raises(subprocess.CalledProcessError):
        entry["main"]()
    assert not (tmp_path / "release").exists()


def test_failed_manifest_publication_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An I/O failure leaves neither a final archive nor a misleading completion marker."""
    root = tmp_path / "runs"
    _fixture(root)
    real_link = os.link

    def fail_manifest(source: Path, destination: Path) -> None:
        if str(destination).endswith(".manifest.json"):
            msg = "simulated disk failure"
            raise OSError(msg)
        real_link(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", fail_manifest)
        with pytest.raises(OSError, match="simulated disk failure"):
            _export(root, tmp_path / "release")
    assert not list((tmp_path / "release").iterdir())
    assert _export(root, tmp_path / "release").exists()


def test_interrupted_publication_reuses_only_identical_archive(tmp_path: Path) -> None:
    """A retry can finish after a kill between archive publication and its manifest."""
    root = tmp_path / "runs"
    _fixture(root)
    path = _export(root, tmp_path / "release")
    archive = path.parent / "test-runs.jsonl.gz"
    before = archive.read_bytes()
    path.unlink()
    assert _export(root, path.parent).exists()
    assert archive.read_bytes() == before
    path.unlink()
    archive.write_bytes(b"different interrupted export")
    with pytest.raises(FileExistsError, match="different archive bytes"):
        _export(root, path.parent)
    assert archive.read_bytes() == b"different interrupted export"


def test_legacy_isolated_reader_export_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The established CLI still writes its schema-1 readings and inbox files."""
    root = tmp_path / "runs"
    _fixture(root)
    for run in root.iterdir():
        run.rename(run.with_name(f"{run.name}-s7"))
    readings = tmp_path / "readings.jsonl"
    readings.write_text('{"seed": 7, "message": "fixture reply"}\n', encoding="utf-8")
    output = tmp_path / "release"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_release.py",
            "--readings",
            str(readings),
            "--pairs-root",
            str(root),
            "--pair-prefix",
            "one",
            "--seeds",
            "7",
            "--output-dir",
            str(output),
            "--name",
            "legacy",
            "--repository",
            "fixture/model",
            "--revision",
            "abc123",
        ],
    )
    entry = runpy.run_path("scripts/export_release.py")
    assert entry["main"]() == 0
    manifest = json.loads((output / "legacy.manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["readings"]["rows"] == 1
    assert manifest["inboxes"]["rows"] == 2
    with gzip.open(output / manifest["readings"]["file"], "rt") as stream:
        assert json.loads(stream.read()) == {"seed": 7, "message": "fixture reply"}

    before = (output / manifest["readings"]["file"]).read_bytes()
    readings.write_text('{"seed": 7, "message": "a different reply"}\n', encoding="utf-8")
    assert runpy.run_path("scripts/export_release.py")["main"]() == 1
    assert (output / manifest["readings"]["file"]).read_bytes() == before


def test_export_retains_all_evidence_and_repeats_byte_for_byte(tmp_path: Path) -> None:
    """Two scenarios at one seed stay separate; decided and delivered text both remain."""
    root = tmp_path / "runs"
    _fixture(root)
    path = _export(root, tmp_path / "first")
    release = json.loads(path.read_text())
    archive = path.parent / release["runs"]["file"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == release["runs"]["sha256"]
    with gzip.open(archive, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 4
    assert {row["manifest"]["experiment"]["treated_role"] for row in rows} == {"reader", "trader"}
    for row in rows:
        source = root / row["run_id"]
        expected_manifest = json.loads((source / "manifest.json").read_text())
        expected_manifest["stimulus_arousal_delta"] = None
        assert row["manifest"] == expected_manifest
        assert (
            row["source_sha256"]["manifest.json"]
            == hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        )
        for field in ("records", "market", "status"):
            content = (source / f"{field}.jsonl").read_bytes()
            expected_rows = [json.loads(line) for line in content.splitlines()]
            if field == "records":
                for record in expected_rows:
                    record["message_affect"] = None
            assert row[field] == expected_rows
            assert row["source_sha256"][f"{field}.jsonl"] == hashlib.sha256(content).hexdigest()
    second = _export(root, tmp_path / "second")
    assert archive.read_bytes() == (second.parent / archive.name).read_bytes()
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        _export(root, path.parent)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("defect", ["sender", "partial", "real", "revision", "duplicate", "market"])
def test_invalid_run_fails_before_any_release_is_written(tmp_path: Path, defect: str) -> None:
    """Fixtures model actual manifests; validation rejects corrupt or incomplete evidence."""
    root = tmp_path / "runs"
    _fixture(root)
    run = root / "one-affective"
    manifest = json.loads((run / "manifest.json").read_text())
    if defect == "sender":
        manifest["experiment"]["treated_role"] = "trader"
    elif defect == "partial":
        manifest["status"] = "running"
    elif defect == "real":
        manifest["payload"]["synthetic"] = False
    elif defect == "revision":
        for item in root.iterdir():
            value = json.loads((item / "manifest.json").read_text())
            value["roster"]["models"][0]["revision"] = None
            (item / "manifest.json").write_text(json.dumps(value))
        manifest["roster"]["models"][0]["revision"] = None
    elif defect == "duplicate":
        path = run / "records.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([*lines[:-1], lines[0]]) + "\n")
    elif defect == "market":
        (run / "market.jsonl").write_text('{"round_index": 0, "price": 100}\n')
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"inputs differ|incomplete|synthetic|revision|duplicate"):
        _export(root, tmp_path / "release")
    assert not (tmp_path / "release").exists()
