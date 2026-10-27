"""Tests for the matched-pair health check.

The failure this catches is silent: a pair that diverged before its treatment still produces a clean
looking paired interval, just a wider one than the design warrants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from affective_contagion.pairing import check_pair

ROLES = ("sentiment_analyst", "fundamental_analyst", "trader")


def _write(root: Path, name: str, *, seed: int, texts: dict[tuple[int, str], str]) -> Path:
    """Write a run whose generations are exactly `texts`."""
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"experiment": {"seed": seed, "treated_role": ROLES[0], "shock_round": 2}}),
        encoding="utf-8",
    )
    rows = [
        {
            "round_index": round_index,
            "role": role,
            "generation": {"raw_output": texts[(round_index, role)]},
        }
        for round_index in (0, 1, 2)
        for role in ROLES
    ]
    (run / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return run


def _texts(*, marker: str = "base", diverge: tuple[int, str] | None = None) -> dict:
    """Build generations, optionally making one cell differ."""
    out = {}
    for round_index in (0, 1, 2):
        for role in ROLES:
            out[(round_index, role)] = f"{round_index}-{role}"
    if diverge is not None:
        out[diverge] = f"{out[diverge]}-{marker}"
    return out


def test_a_clean_pair_is_matched(tmp_path: Path) -> None:
    """Identical before the treatment, differing only in the treated cell at the shock round."""
    treated_cell = (2, ROLES[0])
    neutral = _write(tmp_path, "n", seed=0, texts=_texts())
    affective = _write(tmp_path, "a", seed=0, texts=_texts(marker="shock", diverge=treated_cell))
    report = check_pair(neutral, affective)
    assert report.matched
    assert report.pre_treatment_intact
    assert report.treatment_round_intact
    assert report.treated_cell_differs
    assert report.pre_treatment_identical == report.pre_treatment_cells == 6
    assert "MATCHED" in report.summary


def test_a_pre_treatment_divergence_breaks_the_pair(tmp_path: Path) -> None:
    """One flipped token at round 0 desynchronises everything after it."""
    neutral = _write(tmp_path, "n", seed=0, texts=_texts())
    texts = _texts(marker="drift", diverge=(0, ROLES[1]))
    texts[(2, ROLES[0])] += "-shock"
    affective = _write(tmp_path, "a", seed=0, texts=texts)
    report = check_pair(neutral, affective)
    assert not report.matched
    assert not report.pre_treatment_intact
    assert report.pre_treatment_identical == 5
    assert "NOT MATCHED" in report.summary
    assert "SD is inflated" in report.summary


def test_an_untreated_peer_differing_at_the_shock_round_breaks_the_pair(tmp_path: Path) -> None:
    """A peer has read nothing yet and sees the same price, so a difference is nondeterminism."""
    texts = _texts(marker="noise", diverge=(2, ROLES[2]))
    texts[(2, ROLES[0])] += "-shock"
    report = check_pair(
        _write(tmp_path, "n", seed=0, texts=_texts()),
        _write(tmp_path, "a", seed=0, texts=texts),
    )
    assert report.pre_treatment_intact
    assert not report.treatment_round_intact
    assert not report.matched


def test_an_unchanged_treated_cell_is_reported_as_no_manipulation(tmp_path: Path) -> None:
    """If the treated agent wrote the same thing in both arms, nothing entered the channel."""
    report = check_pair(
        _write(tmp_path, "n", seed=0, texts=_texts()),
        _write(tmp_path, "a", seed=0, texts=_texts()),
    )
    assert report.pre_treatment_intact
    assert not report.treated_cell_differs
    assert not report.matched
    assert "NO TREATMENT EFFECT IN THE CHANNEL" in report.summary


def test_arms_must_agree_on_the_treatment(tmp_path: Path) -> None:
    """Comparing runs with different treated roles would silently compare different designs."""
    neutral = _write(tmp_path, "n", seed=0, texts=_texts())
    affective = _write(tmp_path, "a", seed=0, texts=_texts())
    manifest = json.loads((affective / "manifest.json").read_text(encoding="utf-8"))
    manifest["experiment"]["treated_role"] = ROLES[1]
    (affective / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="arms disagree on the treatment"):
        check_pair(neutral, affective)


def test_the_gate_does_not_compare_the_treated_agents_trade_or_the_price(tmp_path: Path) -> None:
    """State the gate's limit: generations only, never trades or prices.

    Two runs whose treated agent traded differently at the shock round, and whose prices then
    differ, still pass, because `check_pair` reads raw generations and nothing else. The gate
    therefore cannot certify that teammates see the same price; E51 counts that directly.
    """
    treated_cell = (2, ROLES[0])
    neutral = _write(tmp_path, "n", seed=0, texts=_texts())
    affective = _write(tmp_path, "a", seed=0, texts=_texts(marker="shock", diverge=treated_cell))
    for run, position, price in ((neutral, -0.5, 99.5), (affective, -0.2, 99.8)):
        (run / "market.jsonl").write_text(
            json.dumps({"round_index": 2, "price": price}) + "\n", encoding="utf-8"
        )
        rows = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
        for row in rows:
            if (row["round_index"], row["role"]) == treated_cell:
                row["position_after"] = position
        (run / "records.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    assert check_pair(neutral, affective).matched
