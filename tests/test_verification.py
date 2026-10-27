"""Tests for recomputing published numbers from source.

The verification module exists to catch a number that drifted from its data, so these tests check
that it *would* catch one, not merely that it runs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from affective_contagion.text_affect import VADLexicon
from affective_contagion.verification import (
    exact_spearman_p,
    peer_response,
    peer_response_by_manifest,
    role_mean,
    treated_self_response,
    verify_release,
)


def _lexicon() -> VADLexicon:
    return VADLexicon({"panic": (0.1, 0.9, 0.2), "steady": (0.6, 0.2, 0.7)})


def _write_pair(root: Path, name: str, *, teammates: int, anxious_word: str) -> None:
    """Write one matched pair whose teammates differ only in the word used.

    The sentiment analyst is the treated sender here, and the manifest says so, as a real run's
    manifest does. This fixture covers only the case in which the fixed-role route (`peer_response`)
    is right. `_write_pair_with_manifest` covers a pair in which another role was treated.
    """
    roles = ["fundamental_analyst", "bull_researcher", "bear_researcher", "trader", "risk_manager"]
    for arm, word in (("affective", anxious_word), ("neutral", "steady")):
        directory = root / f"{name}-{arm}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {"experiment": {"treated_role": "sentiment_analyst", "shock_round": 8, "seed": 1}}
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "round_index": 9,
                "agent_index": index + 1,
                "role": role,
                "decision": {"message": word, "action": "HOLD", "size": 0.0},
            }
            for index, role in enumerate(roles[:teammates])
        ]
        # The treated sender also speaks at round 9; it must never enter the peer sum.
        rows.append(
            {
                "round_index": 9,
                "agent_index": 0,
                "role": "sentiment_analyst",
                "decision": {"message": "panic panic panic", "action": "HOLD", "size": 0.0},
            }
        )
        (directory / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )


def test_the_treated_sender_is_excluded_from_the_peer_sum(tmp_path: Path) -> None:
    """The treated sender speaks at the measured round and must not be counted.

    These pairs treat the sentiment analyst, so both routes must agree. A sender that varies by pair
    is covered by `test_the_manifest_route_excludes_whoever_was_treated`.
    """
    _write_pair(tmp_path, "run-s1", teammates=5, anxious_word="panic")
    _write_pair(tmp_path, "run-s2", teammates=5, anxious_word="panic")
    single = _lexicon().score("panic").arousal - _lexicon().score("steady").arousal
    for route in (peer_response, peer_response_by_manifest):
        got = route(tmp_path, _lexicon(), round_index=9)
        assert got.value / 5 == pytest.approx(single, abs=1e-6), (
            f"{route.__name__}: the per-teammate effect must equal one teammate's shift; a larger "
            "value means the sender was included"
        )


def test_a_pair_with_the_wrong_teammate_count_is_skipped(tmp_path: Path) -> None:
    """A pair with a different teammate count is skipped, so the divisor cannot change silently."""
    _write_pair(tmp_path, "good-s1", teammates=5, anxious_word="panic")
    _write_pair(tmp_path, "good-s2", teammates=5, anxious_word="panic")
    _write_pair(tmp_path, "short-s3", teammates=3, anxious_word="panic")
    got = peer_response(tmp_path, _lexicon(), round_index=9)
    assert got.units == 2, "the three-teammate pair must be skipped, not counted with a smaller sum"


def test_exact_spearman_p_matches_enumeration() -> None:
    """At n=7 the asymptotic formula gives 0.014 where the exact value is 0.024."""
    got = exact_spearman_p(-0.857, 7)
    assert got.value == pytest.approx(0.0238, abs=5e-4)
    assert got.units == 5040


def _release(tmp_path: Path, readings: list[dict], inboxes: list[dict]) -> Path:
    """Write a minimal release bundle and its manifest, the way export_release.py does."""

    def write(name: str, rows: list[dict]) -> tuple[Path, str]:
        path = tmp_path / name
        with gzip.GzipFile(path, "wb", mtime=0) as raw:
            for row in rows:
                raw.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    r_path, r_hash = write("r.jsonl.gz", readings)
    i_path, i_hash = write("i.jsonl.gz", inboxes)
    manifest = tmp_path / "m.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "readings": {"file": r_path.name, "sha256": r_hash, "rows": len(readings)},
                "inboxes": {"file": i_path.name, "sha256": i_hash, "rows": len(inboxes)},
                "repository": "test/model",
                "revision": "0" * 40,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_verify_release_accepts_an_untouched_bundle(tmp_path: Path) -> None:
    """The passing path, so a failure below means something."""
    manifest = _release(tmp_path, [{"a": 1}], [{"b": 2}])
    intact, detail = verify_release(manifest)
    assert intact
    assert "test/model" in detail


def test_verify_release_rejects_an_edited_file(tmp_path: Path) -> None:
    """A release edited after export must fail, or a number could agree only with itself."""
    manifest = _release(tmp_path, [{"a": 1}], [{"b": 2}])
    target = tmp_path / "r.jsonl.gz"
    with gzip.GzipFile(target, "wb", mtime=0) as raw:
        raw.write(b'{"a": 999}\n')
    intact, detail = verify_release(manifest)
    assert not intact
    assert "does not match its manifest hash" in detail


def test_verify_release_rejects_a_missing_file(tmp_path: Path) -> None:
    """A deleted file must not read as an untouched one."""
    manifest = _release(tmp_path, [{"a": 1}], [{"b": 2}])
    (tmp_path / "i.jsonl.gz").unlink()
    intact, detail = verify_release(manifest)
    assert not intact
    assert "missing" in detail


def _reading(seed: int, role: str, arm: str, message: str) -> dict:
    return {"seed": seed, "reader_role": role, "condition": "bare", "arm": arm, "message": message}


def test_role_mean_returns_a_known_answer(tmp_path: Path) -> None:
    """Feed the estimator values chosen in advance and confirm it returns their mean.

    One role responds by exactly 0.7 (panic 0.9 minus steady 0.2), the other by exactly 0.0, so the
    mean must be 0.35. A mean computed over the wrong unit, or a median, fails this.
    """
    readings = [
        _reading(1, "trader", "anxiety_like", "panic"),
        _reading(1, "trader", "neutral", "steady"),
        _reading(1, "risk_manager", "anxiety_like", "steady"),
        _reading(1, "risk_manager", "neutral", "steady"),
    ]
    manifest = _release(tmp_path, readings, [])
    got = role_mean(manifest, _lexicon())
    assert got.value == pytest.approx(0.35)
    assert got.units == 2


def _write_pair_with_manifest(root: Path, name: str, *, treated: str) -> None:
    """Write a pair whose manifest names the treated role.

    Peers say "panic" or "steady"; the treated agent's affective message is "panic steady", a
    different arousal, so a route that counts it instead of a real teammate gives a different sum.
    This is the case the fixed-role estimator gets wrong when the treated role is not the sentiment
    analyst.
    """
    roles = [
        "sentiment_analyst",
        "fundamental_analyst",
        "bull_researcher",
        "bear_researcher",
        "trader",
        "risk_manager",
    ]
    for arm in ("affective", "neutral"):
        directory = root / f"{name}-{arm}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"experiment": {"treated_role": treated, "shock_round": 8, "seed": 1}}),
            encoding="utf-8",
        )
        rows = []
        for index, role in enumerate(roles):
            if role == treated:
                word = "panic steady" if arm == "affective" else "steady"
            else:
                word = "panic" if arm == "affective" else "steady"
            rows.append(
                {
                    "round_index": 9,
                    "agent_index": index,
                    "role": role,
                    "decision": {"message": word, "action": "HOLD", "size": 0.0},
                }
            )
        (directory / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )


def test_the_manifest_route_excludes_whoever_was_treated(tmp_path: Path) -> None:
    """With the trader treated, the trader leaves the sum and the sentiment analyst stays."""
    _write_pair_with_manifest(tmp_path, "run-s1", treated="trader")
    _write_pair_with_manifest(tmp_path, "run-s2", treated="trader")
    single = _lexicon().score("panic").arousal - _lexicon().score("steady").arousal
    got = peer_response_by_manifest(tmp_path, _lexicon(), round_index=9)
    assert got.value / 5 == pytest.approx(single, abs=1e-6)
    # The fixed-role route gets the same data wrong: it counts the treated trader's smaller shift
    # and drops the untreated sentiment analyst, so it does not equal five teammates' shifts.
    old = peer_response(tmp_path, _lexicon(), round_index=9)
    assert old.value / 5 != pytest.approx(single, abs=1e-6)


def test_the_treated_agents_own_shift_is_reported_separately(tmp_path: Path) -> None:
    """The treated agent's own round-9 shift is its own number, not part of any peer sum."""
    _write_pair_with_manifest(tmp_path, "run-s1", treated="risk_manager")
    got = treated_self_response(tmp_path, _lexicon(), round_index=9)
    expected = _lexicon().score("panic steady").arousal - _lexicon().score("steady").arousal
    assert got.value == pytest.approx(expected, abs=1e-6)
    assert got.units == 1
