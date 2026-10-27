"""Tests for cross-platform runtime provenance."""

from pathlib import Path

from affective_contagion.provenance import collect_runtime_provenance


def test_collect_runtime_provenance_hashes_lock_and_reads_git() -> None:
    """A run should record exact source and environment identifiers where available."""
    root = Path(__file__).resolve().parents[1]
    provenance = collect_runtime_provenance(root, root / "uv.lock")
    assert len(provenance.uv_lock_sha256) == 64
    assert provenance.python_version
    assert provenance.operating_system
    assert provenance.git_commit is None or len(provenance.git_commit) == 40
