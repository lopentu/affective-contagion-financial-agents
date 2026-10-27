"""Independent-reader checks pin Student t intervals and reported-size semantics."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from affective_contagion.reader_verification import (
    reader_numbers,
    student_summary,
)


def _bundle(root: Path, *, missing_draw: bool = False) -> Path:
    rows = []
    inboxes = []
    for seed in range(24):
        sender = "source_a" if seed % 2 else "source_b"
        for arm in ("neutral", "anxiety_like"):
            inboxes.append({"seed": seed, "arm": arm, "treated_role": sender})
            for role in ("reader_a", "reader_b"):
                for draw in range(2):
                    if missing_draw and (seed, arm, role, draw) == (0, "neutral", "reader_a", 0):
                        continue
                    rows.append(
                        {
                            "reader_role": role,
                            "seed": seed,
                            "arm": arm,
                            "condition": arm,
                            "draw": draw,
                            "action": "HOLD",
                            "size": 0.4 if arm == "neutral" else 0.6,
                            "downside_risk": 0.3,
                            "confidence": 0.7,
                            "repository": "test/model",
                            "revision": "revision",
                            "inbox_senders": [sender],
                        }
                    )
    manifest: dict = {"seeds": list(range(24)), "repository": "test/model", "revision": "revision"}
    for part, items in (("readings", rows), ("inboxes", inboxes)):
        archive = root / f"{part}.jsonl.gz"
        archive.write_bytes(gzip.compress("\n".join(json.dumps(row) for row in items).encode()))
        manifest[part] = {
            "file": archive.name,
            "rows": len(items),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    path = root / "test.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_student_interval_uses_24_pair_critical_value() -> None:
    """Known symmetric differences have SE=1/sqrt(23), t(23)=2.0686576104."""
    got = student_summary([-1.0, 1.0] * 12)
    half = 2.0686576104190406 / math.sqrt(23)
    assert got["ci"] == pytest.approx([-half, half], abs=1e-10)
    assert got["ci"][1] != pytest.approx(1.98 / math.sqrt(23))


def test_student_p_is_not_a_normal_approximation() -> None:
    """For df=1, t=1, the exact two-sided probability is one half."""
    assert student_summary([0.0, 2.0])["p"] == pytest.approx(0.5)


def test_raw_size_includes_hold_and_does_not_choose_a_sender(tmp_path: Path) -> None:
    """Every reader row is measured, even for HOLD or a different sender label."""
    result = reader_numbers(_bundle(tmp_path))
    assert result["size"]["reader_a"]["value"] == pytest.approx(0.2)
    assert result["counts"]["nonzero_hold_size"] == 192
    assert result["counts"]["neutral_SELL"] == 0


def test_incomplete_pair_is_rejected_even_when_manifest_hash_matches(tmp_path: Path) -> None:
    """A valid checksum cannot excuse an incomplete paired design."""
    with pytest.raises(ValueError, match="Missing role"):
        reader_numbers(_bundle(tmp_path, missing_draw=True))


def test_released_action_intervals_and_multiplicity() -> None:
    """Pin the published calculations to the archived replies."""
    root = Path(__file__).resolve().parents[1] / "data/generations"
    sell = reader_numbers(root / "e46-action-sell.manifest.json")
    buy = reader_numbers(root / "e47-action-buy.manifest.json")
    assert sell["downside_risk"]["trader"]["holm_p"] == pytest.approx(0.0547269061)
    assert sum(r["holm_p"] <= 0.05 for r in sell["downside_risk"].values()) == 3
    assert buy["size"]["bull_researcher"]["ci"] == pytest.approx([-0.2420, -0.1149], abs=0.00005)
    assert buy["counts"]["nonzero_hold_size"] == 100
