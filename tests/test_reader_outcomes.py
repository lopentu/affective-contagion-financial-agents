"""Tests for maintained isolated-reader release analysis."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest
from scipy import stats

from affective_contagion import reader_outcomes
from affective_contagion.reader_outcomes import analyze_reader_outcomes, student_t_estimate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _rows(*, missing: tuple[int, str, str, int] | None = None) -> list[dict[str, object]]:
    """Build a complete two-role paired archive with changing sender labels."""
    rows: list[dict[str, object]] = []
    for seed in (10, 20):
        for role in ("reader_alpha", "reader_beta"):
            for condition in ("neutral", "anxiety_like"):
                for draw in (2, 7):
                    if missing == (seed, role, condition, draw):
                        continue
                    offset = 0.2 if condition == "anxiety_like" else 0.0
                    size = 0.3 + offset + draw / 100
                    rows.append(
                        {
                            "seed": seed,
                            "reader_role": role,
                            "sender_role": "sender_one" if seed == 10 else "sender_two",
                            "condition": condition,
                            "arm": condition,
                            "draw": draw,
                            "action": "HOLD" if role == "reader_alpha" else "SELL",
                            "size": size,
                            "downside_risk": 0.4 + offset,
                            "confidence": 0.7 - offset,
                            "repository": "fixture/model",
                            "revision": "revision-1",
                        }
                    )
    return rows


def _release(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a schema-1 manifest and hashed gzip archive."""
    archive = tmp_path / "readings.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "schema_version": 1,
        "name": "fixture-readers",
        "readings": {
            "file": archive.name,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
        "seeds": [10, 20],
        "repository": "fixture/model",
        "revision": "revision-1",
    }
    path = tmp_path / "fixture.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_student_t_known_answer_uses_df23_quantile() -> None:
    """At n=24 the interval must use t(23), not a fixed 1.98."""
    differences = tuple((index - 11.5) / 10 for index in range(24))
    estimate = student_t_estimate(differences)
    critical = (estimate.interval[1] - estimate.mean) / estimate.standard_error

    assert estimate.degrees_of_freedom == 23
    assert critical == pytest.approx(stats.t.ppf(0.975, 23))
    assert critical > 1.98


def test_known_answer_rejects_historical_1_98_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The known answer fails if a fixed 1.98 critical value is injected."""
    differences = tuple(index / 10 for index in range(24))
    expected = student_t_estimate(differences).interval
    monkeypatch.setattr(reader_outcomes.stats.t, "ppf", lambda *_args: 1.98)

    injected = student_t_estimate(differences).interval
    with pytest.raises(AssertionError):
        assert injected == pytest.approx(expected)


def test_release_analysis_derives_roles_conditions_draws_and_raw_hold_sizes(tmp_path: Path) -> None:
    """No fixed reader or sender roster is needed to validate a balanced archive."""
    report = analyze_reader_outcomes(_release(tmp_path, _rows()))

    assert report.roles == ("reader_alpha", "reader_beta")
    assert report.conditions == ("anxiety_like", "neutral")
    assert report.draws == (2, 7)
    assert report.raw_size_includes_hold
    assert "including HOLD" in report.size_measurement
    assert "do not execute" in report.size_measurement
    assert report.nonzero_hold_sizes == 8
    assert report.action_counts["neutral"] == {"HOLD": 4, "SELL": 4}
    assert report.outcomes["size"][0].estimate.mean == pytest.approx(0.2)


def test_e46_trader_uses_t_p_value_before_holm() -> None:
    """The exact p-value falls above 0.025 where a normal approximation falls below it."""
    report = analyze_reader_outcomes(
        REPOSITORY_ROOT / "data/generations/e46-action-sell.manifest.json"
    )
    trader = next(
        outcome for outcome in report.outcomes["downside_risk"] if outcome.role == "trader"
    )

    old_approximation = 0.023606
    fourth_holm_threshold = 0.025
    assert old_approximation < fourth_holm_threshold
    assert trader.estimate.p_value > fourth_holm_threshold
    assert trader.estimate.p_value == pytest.approx(0.0273634530)
    assert not trader.passes_holm
    assert report.nonzero_hold_sizes == 518


def test_e47_interval_is_not_the_df23_fallback() -> None:
    """The released 24-pair result reproduces the wider Student t interval."""
    report = analyze_reader_outcomes(
        REPOSITORY_ROOT / "data/generations/e47-action-buy.manifest.json"
    )
    bull = next(outcome for outcome in report.outcomes["size"] if outcome.role == "bull_researcher")

    assert bull.estimate.interval == pytest.approx((-0.2420146255, -0.1149298189), abs=1e-9)
    assert bull.passes_holm
    assert report.nonzero_hold_sizes == 100


def test_pr22_actual_buy_mechanism_has_one_discounting_role_and_is_unresolved() -> None:
    """Student t intervals leave one eligible discounting role, below PR22's required two."""
    report = analyze_reader_outcomes(
        REPOSITORY_ROOT / "data/generations/e47-action-buy.manifest.json",
        include_buy_mechanism=True,
    )

    assert report.buy_mechanism is not None
    assert report.buy_mechanism.qualifying_size_roles == ("bull_researcher", "trader")
    assert report.buy_mechanism.discounting_roles == ("trader",)
    assert report.buy_mechanism.risk_aversion_roles == ()
    assert report.buy_mechanism.verdict == "unresolved"


def test_pr22_two_role_known_answer_supports_discounting(tmp_path: Path) -> None:
    """Two qualifying roles with lower downside and higher confidence issue the planned verdict."""
    rows = _rows()
    for row in rows:
        affective = row["condition"] == "anxiety_like"
        row["size"] = 0.2 if affective else 0.6
        row["downside_risk"] = 0.4 if affective else 0.6
        row["confidence"] = 0.7 if affective else 0.5
    report = analyze_reader_outcomes(
        _release(tmp_path, rows),
        include_buy_mechanism=True,
    )

    assert report.buy_mechanism is not None
    assert report.buy_mechanism.discounting_roles == ("reader_alpha", "reader_beta")
    assert report.buy_mechanism.risk_aversion_roles == ()
    assert report.buy_mechanism.verdict == "discounting_supported"


@pytest.mark.parametrize(
    ("name", "nonzero_hold_sizes"),
    [("e48-action-granite", 202), ("e48-action-gemma", 169)],
)
def test_e48_family_archives_have_no_holm_passing_size_role(
    name: str, nonzero_hold_sizes: int
) -> None:
    """The maintained route covers both released family checks and retains their action null."""
    report = analyze_reader_outcomes(REPOSITORY_ROOT / f"data/generations/{name}.manifest.json")

    assert report.rows == 1770
    assert not any(outcome.passes_holm for outcome in report.outcomes["size"])
    assert report.nonzero_hold_sizes == nonzero_hold_sizes


def test_missing_draw_makes_the_paired_archive_invalid(tmp_path: Path) -> None:
    """Every seed-role-condition cell must carry the same draw identifiers."""
    manifest = _release(
        tmp_path,
        _rows(missing=(20, "reader_beta", "anxiety_like", 7)),
    )

    with pytest.raises(ValueError, match="inconsistent draw sets"):
        analyze_reader_outcomes(manifest)


def test_malformed_archive_row_is_rejected_after_hash_verification(tmp_path: Path) -> None:
    """A hash-valid archive still has to contain complete JSON objects."""
    archive = tmp_path / "bad.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as stream:
        stream.write('{"seed": 10}\n')
        stream.write("not json\n")
    manifest = {
        "schema_version": 1,
        "name": "bad",
        "readings": {
            "file": archive.name,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "rows": 2,
        },
        "seeds": [10],
        "repository": "fixture/model",
        "revision": "revision-1",
    }
    path = tmp_path / "bad.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON at line 2"):
        analyze_reader_outcomes(path)


def test_nonfinite_known_answer_is_rejected() -> None:
    """The statistical routine never turns a malformed value into a result."""
    with pytest.raises(ValueError, match="finite"):
        student_t_estimate((0.0, math.nan))
