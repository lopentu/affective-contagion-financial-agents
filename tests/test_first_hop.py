"""Tests for the first-hop manipulation check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from affective_contagion.first_hop import first_hop_separation


def _write_run(
    root: Path, name: str, *, seed: int, arousal: float, treated: str | None = "trader"
) -> Path:
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": {"seed": seed, "treated_role": treated, "shock_round": 4},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"round_index": 3, "role": "trader", "message_affect": {"arousal": 0.1}},
        {"round_index": 4, "role": "other", "message_affect": {"arousal": 0.9}},
        {"round_index": 4, "role": "trader", "message_affect": {"arousal": arousal}},
    ]
    (run / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return run


def test_pairs_by_seed_and_reads_only_the_treated_agent(tmp_path: Path) -> None:
    """The measurement must come from the treated role at the shock round, paired on seed."""
    neutral = [_write_run(tmp_path, f"n{s}", seed=s, arousal=0.40) for s in (0, 1)]
    affective = [_write_run(tmp_path, f"a{s}", seed=s, arousal=0.50) for s in (1, 0)]
    result = first_hop_separation(neutral, affective)
    assert [pair.seed for pair in result.pairs] == [0, 1]
    assert result.mean_difference == pytest.approx(0.10)
    assert result.positive_seeds == 2
    assert result.treated_role == "trader"


def test_power_arithmetic_is_reported(tmp_path: Path) -> None:
    """Sizing a confirmatory run needs the sd of differences, not just the mean."""
    neutral = [_write_run(tmp_path, f"n{s}", seed=s, arousal=0.40) for s in range(4)]
    affective = [
        _write_run(tmp_path, f"a{s}", seed=s, arousal=a)
        for s, a in enumerate((0.42, 0.38, 0.46, 0.44))
    ]
    result = first_hop_separation(neutral, affective)
    assert result.difference_sd > 0
    # More seeds must lower the detectable difference, by the square root of the count.
    twelve = result.minimum_detectable_difference(12)
    assert twelve < result.minimum_detectable_difference(3)
    assert result.minimum_detectable_difference(48) == pytest.approx(twelve / 2, rel=1e-6)
    assert result.seeds_required(0.04) > 0
    with pytest.raises(ValueError, match="assumed_difference must be positive"):
        result.seeds_required(0.0)


def test_broadcast_runs_are_rejected(tmp_path: Path) -> None:
    """A broadcast run has no single treated agent, so it has no first hop."""
    neutral = [_write_run(tmp_path, "n0", seed=0, arousal=0.4, treated=None)]
    affective = [_write_run(tmp_path, "a0", seed=0, arousal=0.5, treated=None)]
    with pytest.raises(ValueError, match="no first hop"):
        first_hop_separation(neutral, affective)


def test_unpairable_arms_are_rejected(tmp_path: Path) -> None:
    """Comparing different seeds across arms would confound seed with condition."""
    neutral = [_write_run(tmp_path, "n0", seed=0, arousal=0.4)]
    affective = [_write_run(tmp_path, "a9", seed=9, arousal=0.5)]
    with pytest.raises(ValueError, match="nothing can be paired"):
        first_hop_separation(neutral, affective)


def test_missing_affect_is_rejected(tmp_path: Path) -> None:
    """A run without message affect cannot support the check and must not score zero."""
    run = _write_run(tmp_path, "n0", seed=0, arousal=0.4)
    rows = [{"round_index": 4, "role": "trader", "message_affect": None}]
    (run / "records.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    other = _write_run(tmp_path, "a0", seed=0, arousal=0.5)
    with pytest.raises(ValueError, match="no message affect"):
        first_hop_separation([run], [other])


def test_single_pair_reports_undefined_spread(tmp_path: Path) -> None:
    """One seed gives a difference but no precision, and must not pretend otherwise."""
    result = first_hop_separation(
        [_write_run(tmp_path, "n0", seed=0, arousal=0.4)],
        [_write_run(tmp_path, "a0", seed=0, arousal=0.5)],
    )
    assert result.mean_difference == pytest.approx(0.1)
    assert math.isnan(result.difference_sd)
    assert math.isnan(result.cohens_dz)
    assert all(math.isnan(bound) for bound in result.mean_difference_ci)


def test_partially_overlapping_arms_are_rejected(tmp_path: Path) -> None:
    """Pairing on the intersection alone would shrink the variance that sizes every later run.

    A seed that crashed in one arm must surface as an error, not silently reduce n.
    """
    neutral = [_write_run(tmp_path, f"n{s}", seed=s, arousal=0.40) for s in (0, 1, 2)]
    affective = [_write_run(tmp_path, f"a{s}", seed=s, arousal=0.45) for s in (1, 2, 3)]
    with pytest.raises(ValueError, match=r"not paired.*\[0\].*\[3\]"):
        first_hop_separation(neutral, affective)


def test_nan_arousal_is_rejected(tmp_path: Path) -> None:
    """NaN would propagate through mean and sd as 'undefined spread' rather than corrupt input."""
    run = _write_run(tmp_path, "n0", seed=0, arousal=0.4)
    rows = [{"round_index": 4, "role": "trader", "message_affect": {"arousal": float("nan")}}]
    (run / "records.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    other = _write_run(tmp_path, "a0", seed=0, arousal=0.5)
    with pytest.raises(ValueError, match="NaN arousal"):
        first_hop_separation([run], [other])


def test_confidence_interval_brackets_the_mean(tmp_path: Path) -> None:
    """The interval decides whether a pilot licenses a conclusion, so it must be reported."""
    neutral = [_write_run(tmp_path, f"n{s}", seed=s, arousal=0.40) for s in range(6)]
    affective = [
        _write_run(tmp_path, f"a{s}", seed=s, arousal=a)
        for s, a in enumerate((0.42, 0.38, 0.46, 0.44, 0.41, 0.43))
    ]
    result = first_hop_separation(neutral, affective)
    low, high = result.mean_difference_ci
    assert low < result.mean_difference < high
    # t(0.975, df=5) = 2.571 is wider than the normal quantile, so the interval must exceed the
    # naive z-based one rather than silently reusing the power constant.
    half_width = high - result.mean_difference
    assert half_width > 1.96 * result.difference_sd / math.sqrt(6)
