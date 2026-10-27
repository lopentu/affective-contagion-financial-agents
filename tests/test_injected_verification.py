"""Known-answer tests for independent verification of released injected runs."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from affective_contagion.emotion_probe import EmotionScore
from affective_contagion.injected_verification import InjectedVerification, verify_injected_release

ROLES = (
    "sentiment_analyst",
    "fundamental_analyst",
    "bull_researcher",
    "bear_researcher",
    "trader",
    "risk_manager",
)
MODEL = {
    "key": "fixture-model",
    "repository": "fixture/model",
    "revision": "abc123",
    "activation_layer": None,
    "dtype": "float32",
    "device_map": "cpu",
    "quantization": "none",
    "max_new_tokens": 32,
    "temperature": 0.0,
    "constrained_decision_format": True,
    "trust_remote_code": False,
    "deterministic": True,
}
DISCRIMINANTS = {
    "calm": 0.0,
    "low": 0.2,
    "high": 0.8,
    "medium": 0.5,
    "anxious": 0.7,
    "alarms": 0.6,
    "alarmed": 0.8,
    "frightened": 0.6,
    "dread": 0.5,
    "alarming": 0.4,
}


@dataclass(frozen=True, slots=True)
class PairSpec:
    """All synthetic variations needed for one pair."""

    payload: str
    seed: int
    sender: str
    neutral_id: str
    affective_id: str
    affective_messages: dict[str, str]
    same_request: bool = True
    same_fill: bool = True
    same_price: bool = True
    gate_peer_mismatch: bool = False
    neutral_fallbacks: tuple[tuple[int, str], ...] = ()
    affective_fallbacks: tuple[tuple[int, str], ...] = ()


SPECS = (
    PairSpec(
        payload="Alpha",
        seed=7,
        sender="trader",
        neutral_id="run-04",
        affective_id="run-01",
        affective_messages={
            "sentiment_analyst": "high",
            "fundamental_analyst": "low",
            "bull_researcher": "high",
            "bear_researcher": "calm",
            "trader": "low",
            "risk_manager": "anxious",
        },
        neutral_fallbacks=((3, "bull_researcher"),),
        affective_fallbacks=((3, "bear_researcher"),),
    ),
    PairSpec(
        payload="Beta",
        seed=7,
        sender="risk_manager",
        neutral_id="run-02",
        affective_id="run-03",
        affective_messages={
            "sentiment_analyst": "low",
            "fundamental_analyst": "alarms",
            "bull_researcher": "calm",
            "bear_researcher": "high",
            "trader": "low",
            "risk_manager": "medium",
        },
        same_request=False,
        same_fill=False,
        same_price=False,
        gate_peer_mismatch=True,
        neutral_fallbacks=((3, "sentiment_analyst"),),
    ),
    PairSpec(
        payload="Gamma",
        seed=8,
        sender="sentiment_analyst",
        neutral_id="run-06",
        affective_id="run-05",
        affective_messages={
            "sentiment_analyst": "low",
            "fundamental_analyst": "anxious",
            "bull_researcher": "alarmed",
            "bear_researcher": "frightened",
            "trader": "dread",
            "risk_manager": "alarming",
        },
    ),
)


class StrictScorer:
    """Fake scorer that verifies input order and emits one checked output per input."""

    def __init__(self, expected: tuple[str, ...]) -> None:
        """Record the exact input sequence the verifier must submit."""
        self.expected = expected
        self.calls = 0

    def score_all(self, texts: Sequence[str]) -> tuple[EmotionScore, ...]:
        """Reject changed inputs before returning deterministic discriminants."""
        self.calls += 1
        assert tuple(texts) == self.expected
        assert all(text in DISCRIMINANTS for text in texts)
        scores = tuple(EmotionScore(labels=(("fear", DISCRIMINANTS[text]),)) for text in texts)
        assert len(scores) == len(texts)
        return scores


class ShortScorer:
    """Invalid fake used to prove verifier-side output cardinality checks."""

    def score_all(self, texts: Sequence[str]) -> tuple[EmotionScore, ...]:
        """Return one fewer score than requested."""
        return tuple(EmotionScore(labels=()) for _ in texts[:-1])


def _write_lexicon(path: Path) -> Path:
    """Write a tiny signed lexicon with a deliberate suffix-only match."""
    rows = (
        "term\tvalence\tarousal\tdominance",
        "calm\t0\t-0.8\t0",
        "low\t0\t-0.4\t0",
        "high\t0\t0.8\t0",
        "medium\t0\t0.2\t0",
        "anxious\t0\t0.6\t0",
        "alarm\t0\t0.4\t0",
        "alarmed\t0\t0.8\t0",
        "frightened\t0\t0.4\t0",
        "dread\t0\t0.2\t0",
        "alarming\t0\t0.0\t0",
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _manifest(run_id: str, spec: PairSpec, arm: str) -> dict[str, object]:
    """Return a full saved-run manifest; pair fields vary only where the design permits."""
    return {
        "schema_version": 3,
        "created_at_utc": "2026-09-07T00:00:00+00:00",
        "status": "completed",
        "run_id": run_id,
        "runtime_provenance": {
            "git_commit": "fixture",
            "git_dirty": False,
            "uv_lock_sha256": "lock",
            "python_version": "3.13",
            "operating_system": "fixture",
            "accelerator": None,
            "torch_version": None,
            "transformers_version": None,
        },
        "experiment": {
            "roles": list(ROLES),
            "topology": "full",
            "n_rounds": 4,
            "shock_round": 2,
            "shock_duration": 1,
            "shock_type": arm,
            "treated_role": spec.sender,
            "message_intervention": "inject_message",
            "fundamental_value": 100.0,
            "price_impact": 0.02,
            "mean_reversion": 0.01,
            "noise_sigma": 0.003,
            "seed": spec.seed,
            "var_maxlags": 1,
            "irf_horizon": 2,
        },
        "roster": {
            "schema_version": 1,
            "models": [MODEL],
            "agents": [
                {"role": role, "model_key": MODEL["key"], "system_prompt": f"Be {role}."}
                for role in ROLES
            ],
        },
        "payload": {"asset": spec.payload, "synthetic": True},
        "stimuli": {"neutral": "calm report", "affective": "alarmed report"},
        "stimulus_arousal_delta": 0.4,
        "delivered_stimulus": "calm report" if arm == "neutral" else "alarmed report",
        "intervention_seed": 42,
        "vad_lexicon_source": "fixture.tsv",
        "affect_direction_captured": False,
        "parse_failure_rate": 0.0,
        "run_health": None,
        "gain_definition": "fixture",
    }


def _records(spec: PairSpec, arm: str) -> list[dict[str, object]]:
    """Write every round/role cell, keeping raw gate text separate from response messages."""
    fallbacks = spec.neutral_fallbacks if arm == "neutral" else spec.affective_fallbacks
    rows: list[dict[str, object]] = []
    for round_index in range(4):
        for agent_index, role in enumerate(ROLES):
            raw = f"shared-r{round_index}-{role}"
            if round_index == 2 and role == spec.sender:
                raw = f"treated-{arm}-{role}"
            if (
                spec.gate_peer_mismatch
                and arm == "affective"
                and round_index == 2
                and role == "fundamental_analyst"
            ):
                raw = "nondeterministic-peer"
            action = "HOLD"
            size = 0.0
            if round_index == 2 and role == spec.sender:
                action = "SELL"
                size = 0.5 if arm == "neutral" or spec.same_request else 0.2
            position = -0.5 if round_index >= 2 and role == spec.sender else 0.0
            if (
                arm == "affective"
                and not spec.same_fill
                and round_index >= 2
                and role == spec.sender
            ):
                position = -0.2
            message = "calm"
            if arm == "affective" and round_index == 3:
                message = spec.affective_messages[role]
            rows.append(
                {
                    "round_index": round_index,
                    "agent_index": agent_index,
                    "role": role,
                    "context": {
                        "round_index": round_index,
                        "agent_index": agent_index,
                        "role": role,
                        "price": 100.0,
                        "fundamental_value": 100.0,
                        "news": None,
                        "inbox": [],
                    },
                    "generation": {
                        "raw_output": raw,
                        "model_key": MODEL["key"],
                        "repository": MODEL["repository"],
                        "revision": MODEL["revision"],
                        "generation_seed": spec.seed + round_index + agent_index,
                        "activation_layer": None,
                        "affect_score": None,
                        "prompt_messages": [],
                    },
                    "decision": {"message": message, "action": action, "size": size},
                    "delivery": {
                        "delivered_text": message,
                        "intervention_name": "inject_message",
                        "intervention_seed": 42,
                        "preservation_validated": None,
                    },
                    "parse_succeeded": (round_index, role) not in fallbacks,
                    "position_after": position,
                    "message_affect": None,
                }
            )
    return rows


def _run_row(spec: PairSpec, arm: str) -> dict[str, object]:
    run_id = spec.neutral_id if arm == "neutral" else spec.affective_id
    price_at_shock = 99.0
    if arm == "affective" and not spec.same_price:
        price_at_shock = 98.5
    return {
        "run_id": run_id,
        "manifest": _manifest(run_id, spec, arm),
        "records": _records(spec, arm),
        "market": [
            {"round_index": round_index, "price": price_at_shock if round_index == 2 else 100.0}
            for round_index in range(4)
        ],
        "status": [
            {"at_utc": "2026-09-07T00:00:00+00:00", "status": "running", "detail": None},
            {"at_utc": "2026-09-07T00:01:00+00:00", "status": "completed", "detail": None},
        ],
        "source_sha256": {
            "manifest.json": "1" * 64,
            "records.jsonl": "2" * 64,
            "market.jsonl": "3" * 64,
            "status.jsonl": "4" * 64,
        },
    }


def _write_release(
    root: Path,
    *,
    mutate_rows: Callable[[list[dict[str, object]]], None] | None = None,
    declared_rows: int | None = None,
    outer_models: list[dict[str, str]] | None = None,
) -> tuple[Path, tuple[str, ...]]:
    """Write a release in deliberately shuffled row order and return expected scorer inputs."""
    rows = [_run_row(spec, arm) for spec in SPECS for arm in ("neutral", "affective")]
    rows = [rows[index] for index in (5, 0, 3, 2, 1, 4)]
    if mutate_rows is not None:
        mutate_rows(rows)
    archive = root / "synthetic-runs.jsonl.gz"
    with gzip.GzipFile(filename="", fileobj=archive.open("wb"), mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode())
    release = {
        "schema_version": 2,
        "kind": "complete_synthetic_runs",
        "name": "synthetic",
        "runs": {
            "file": archive.name,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "rows": len(rows) if declared_rows is None else declared_rows,
        },
        "models": outer_models
        if outer_models is not None
        else [{"repository": MODEL["repository"], "revision": MODEL["revision"]}],
        "exported_at_commit": "fixture",
        "exported_from_dirty_tree": False,
        "synthetic": True,
    }
    manifest = root / "synthetic.manifest.json"
    manifest.write_text(json.dumps(release), encoding="utf-8")
    expected_texts: list[str] = []
    for spec in sorted(SPECS, key=lambda item: tuple(sorted((item.neutral_id, item.affective_id)))):
        for arm, _run_id in sorted(
            (("neutral", spec.neutral_id), ("affective", spec.affective_id)),
            key=lambda item: item[1],
        ):
            expected_texts.extend(
                "calm" if arm == "neutral" else spec.affective_messages[role]
                for role in ROLES
                if role != spec.sender
            )
    return manifest, tuple(expected_texts)


def _verification(tmp_path: Path) -> tuple[InjectedVerification, StrictScorer]:
    manifest, expected = _write_release(tmp_path)
    scorer = StrictScorer(expected)
    report = verify_injected_release(manifest, _write_lexicon(tmp_path / "vad.tsv"), scorer)
    return report, scorer


def test_known_answer_counts_and_manifest_pairing(tmp_path: Path) -> None:
    """Payloads at one seed remain distinct; planned counts use both manifests and all rounds."""
    report, scorer = _verification(tmp_path)
    assert scorer.calls == 1
    assert report.archive.runs == 6
    assert report.archive.pairs == 3
    assert report.archive.models == (("fixture/model", "abc123"),)
    assert [row.run_ids for row in report.pairs] == [
        ("run-01", "run-04"),
        ("run-02", "run-03"),
        ("run-05", "run-06"),
    ]
    assert {row.seed for row in report.pairs[:2]} == {7}
    assert len({row.payload_json for row in report.pairs[:2]}) == 2
    pooled = report.identity.pooled
    assert pooled.pairs == 3
    assert pooled.gate_passing == 2
    assert pooled.gate_failures == 1
    assert pooled.same_requested_trade == 2
    assert pooled.same_requested_trade_gate_passing == 2
    assert pooled.same_filled_position == 2
    assert pooled.same_filled_position_gate_passing == 2
    assert pooled.same_observed_price == 2
    assert pooled.same_observed_price_gate_passing == 2
    assert pooled.parser_fallbacks_neutral == 2
    assert pooled.parser_fallbacks_affective == 1
    assert pooled.parser_fallbacks_all_rounds == 3
    assert len(report.identity.by_payload) == 3
    assert report.identity.gate_failure_run_ids == (("run-02", "run-03"),)
    failed = report.pairs[1].gate
    assert failed.pre_treatment_identical == failed.pre_treatment_cells == 12
    assert failed.treatment_round_peers == 5
    assert failed.treatment_round_peers_identical == 4
    assert failed.treated_cell_differs
    assert not failed.passed


def test_manifest_sender_controls_all_response_estimators(tmp_path: Path) -> None:
    """A real non-sender contribution and treated self term expose a fixed-sender route."""
    report, _ = _verification(tmp_path)
    historical = report.outcome_4a_historical_fixed_sentiment
    manifest = report.outcome_4b_by_manifest
    assert historical.full.five_teammate_sum == pytest.approx((1.9 + 1.9 + 3.0) / 3)
    assert manifest.full.five_teammate_sum == pytest.approx((2.5 + 1.6 + 3.0) / 3)
    assert historical.full.five_teammate_sum != pytest.approx(manifest.full.five_teammate_sum)
    assert historical.full.interval == pytest.approx((1.9, 3.0))
    assert manifest.full.interval == pytest.approx((1.6, 3.0))
    assert report.outcome_5_treated_self.value == pytest.approx(0.3)
    assert report.outcome_5_treated_self.interval == pytest.approx((0.2, 0.5))
    assert report.outcome_5_treated_self.pairs == 3


def test_echo_denominators_and_classifier_known_answers(tmp_path: Path) -> None:
    """Echo dropping rescales survivors, counts roles, and names all-echo omissions."""
    report, _ = _verification(tmp_path)
    manifest = report.outcome_4b_by_manifest
    assert manifest.full.retained_pairs == 3
    assert manifest.full.retained_roles == 15
    assert manifest.full.all_echo_pairs_omitted == 0
    assert manifest.echo_dropped.five_teammate_sum == pytest.approx((2.25 + 1.6) / 2)
    assert manifest.echo_dropped.interval == pytest.approx((1.6, 2.25))
    assert manifest.echo_dropped.retained_pairs == 2
    assert manifest.echo_dropped.retained_roles == 9
    assert manifest.echo_dropped.all_echo_pairs_omitted == 1
    historical = report.outcome_4a_historical_fixed_sentiment.echo_dropped
    assert historical.five_teammate_sum == pytest.approx((1.5 + 1.9) / 2)
    assert historical.retained_pairs == 2
    assert historical.retained_roles == 9
    assert historical.all_echo_pairs_omitted == 1

    classifier = report.outcome_7_classifier
    assert classifier is not None
    assert classifier.full.value_per_teammate == pytest.approx((0.5 + 0.36 + 0.6) / 3)
    assert classifier.full.interval_per_teammate == pytest.approx((0.36, 0.6))
    assert classifier.full.retained_pairs == 3
    assert classifier.full.retained_roles == 15
    assert classifier.echo_dropped.value_per_teammate == pytest.approx((0.45 + 0.36) / 2)
    assert classifier.echo_dropped.interval_per_teammate == pytest.approx((0.36, 0.45))
    assert classifier.echo_dropped.retained_pairs == 2
    assert classifier.echo_dropped.retained_roles == 9
    assert classifier.echo_dropped.all_echo_pairs_omitted == 1


def test_per_role_weighting_and_both_vad_match_modes(tmp_path: Path) -> None:
    """Each role uses only non-sender pairs; suffix matching is separate and visible."""
    report, _ = _verification(tmp_path)
    surface = report.outcome_6_surface_only
    suffix = report.outcome_6_surface_then_suffix
    assert surface["fundamental_analyst"].value == pytest.approx((0.2 + 0.4 + 0.7) / 3)
    assert suffix["fundamental_analyst"].value == pytest.approx((0.2 + 0.6 + 0.7) / 3)
    assert surface["fundamental_analyst"].pairs == 3
    assert surface["fundamental_analyst"].positive_pairs == 3
    assert surface["trader"].value == pytest.approx((0.2 + 0.5) / 2)
    assert surface["trader"].pairs == 2
    assert surface["sentiment_analyst"].value == pytest.approx((0.8 + 0.2) / 2)
    assert surface["sentiment_analyst"].pairs == 2
    assert surface["bull_researcher"].interval == pytest.approx((0.0, 0.8))
    assert report.corrected_4b_registered_matching.full.five_teammate_sum == pytest.approx(
        (2.5 + 1.8 + 3.0) / 3
    )
    assert report.corrected_4a_registered_matching.full.five_teammate_sum == pytest.approx(
        (1.9 + 2.1 + 3.0) / 3
    )


def test_missing_classifier_is_explicit(tmp_path: Path) -> None:
    """A core-only verification returns no invented classifier result."""
    manifest, _ = _write_release(tmp_path)
    report = verify_injected_release(manifest, _write_lexicon(tmp_path / "vad.tsv"))
    assert report.outcome_7_classifier is None
    assert report.identity.pooled.pairs == 3


def test_empty_raw_generation_is_counted_as_a_parser_failure(tmp_path: Path) -> None:
    """An empty model reply is valid failure evidence, not an incomplete record."""

    def mutate(rows: list[dict[str, object]]) -> None:
        records = rows[0]["records"]
        assert isinstance(records, list)
        records[0]["generation"]["raw_output"] = ""
        records[0]["parse_succeeded"] = False

    manifest, _ = _write_release(tmp_path, mutate_rows=mutate)
    report = verify_injected_release(manifest, _write_lexicon(tmp_path / "vad.tsv"))
    assert report.identity.pooled.parser_fallbacks_all_rounds == 4


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_missing_or_duplicate_generation_cells_are_rejected(tmp_path: Path, defect: str) -> None:
    """A complete-looking row cannot omit or repeat a round/role cell."""

    def mutate(rows: list[dict[str, object]]) -> None:
        records = rows[0]["records"]
        assert isinstance(records, list)
        if defect == "missing":
            records.pop()
        else:
            records[-1] = records[0]

    manifest, expected = _write_release(tmp_path, mutate_rows=mutate)
    with pytest.raises(ValueError, match=r"generation grid|duplicate generation cell"):
        verify_injected_release(
            manifest, _write_lexicon(tmp_path / "vad.tsv"), StrictScorer(expected)
        )


@pytest.mark.parametrize("defect", ["missing_partner", "duplicate_arm"])
def test_missing_or_duplicate_pair_arms_are_rejected(tmp_path: Path, defect: str) -> None:
    """Every payload and seed must contribute exactly one neutral and one affective run."""

    def mutate(rows: list[dict[str, object]]) -> None:
        target = next(row for row in rows if row["run_id"] == "run-01")
        if defect == "missing_partner":
            rows.remove(target)
            return
        manifest = target["manifest"]
        assert isinstance(manifest, dict)
        experiment = manifest["experiment"]
        assert isinstance(experiment, dict)
        experiment["shock_type"] = "neutral"

    manifest, expected = _write_release(tmp_path, mutate_rows=mutate)
    with pytest.raises(ValueError, match=r"lacks one complete arm|duplicate neutral arm"):
        verify_injected_release(
            manifest, _write_lexicon(tmp_path / "vad.tsv"), StrictScorer(expected)
        )


@pytest.mark.parametrize("defect", ["treated_role", "config", "roster", "outer_model"])
def test_inconsistent_manifests_and_model_identity_are_rejected(
    tmp_path: Path, defect: str
) -> None:
    """Pair invariants and roster-derived model identity fail before outcomes are computed."""

    def mutate(rows: list[dict[str, object]]) -> None:
        target = next(row for row in rows if row["run_id"] == "run-01")
        manifest = target["manifest"]
        assert isinstance(manifest, dict)
        experiment = manifest["experiment"]
        roster = manifest["roster"]
        assert isinstance(experiment, dict)
        assert isinstance(roster, dict)
        if defect == "treated_role":
            experiment["treated_role"] = "risk_manager"
        elif defect == "config":
            experiment["topology"] = "chain"
        elif defect == "roster":
            models = roster["models"]
            assert isinstance(models, list)
            model = models[0]
            assert isinstance(model, dict)
            model["revision"] = "different"

    outer = None
    if defect == "outer_model":
        outer = [{"repository": "filename/guess", "revision": "wrong"}]
    manifest, expected = _write_release(tmp_path, mutate_rows=mutate, outer_models=outer)
    with pytest.raises(ValueError, match=r"config|roster|model identit"):
        verify_injected_release(
            manifest, _write_lexicon(tmp_path / "vad.tsv"), StrictScorer(expected)
        )


def test_archive_hash_and_cardinality_are_verified(tmp_path: Path) -> None:
    """Both byte tampering and a false row declaration fail before archive analysis."""
    manifest, expected = _write_release(tmp_path)
    release = json.loads(manifest.read_text(encoding="utf-8"))
    archive = tmp_path / release["runs"]["file"]
    with archive.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="manifest hash"):
        verify_injected_release(
            manifest, _write_lexicon(tmp_path / "vad.tsv"), StrictScorer(expected)
        )

    other = tmp_path / "cardinality"
    other.mkdir()
    false_manifest, false_expected = _write_release(other, declared_rows=7)
    with pytest.raises(ValueError, match="declares 7 rows but contains 6"):
        verify_injected_release(
            false_manifest, _write_lexicon(other / "vad.tsv"), StrictScorer(false_expected)
        )


def test_emotion_scorer_output_cardinality_is_strict(tmp_path: Path) -> None:
    """A scorer must return exactly one result for every input in the same order."""
    manifest, _ = _write_release(tmp_path)
    with pytest.raises(ValueError, match="scores for"):
        verify_injected_release(manifest, _write_lexicon(tmp_path / "vad.tsv"), ShortScorer())
