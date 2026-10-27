"""Independently verify released complete runs from the injected-message design.

This module reads only a schema-version-2 run archive. It does not use the E51 runners or the
pairing and trade-identity modules. Pair identity, treatment timing, sender identity, and model
identity all come from archived manifests.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from affective_contagion.emotion_probe import EmotionScorer
from affective_contagion.scrub import scrub_decision_vocabulary
from affective_contagion.text_affect import VADLexicon

# Comparator route that assumes the sentiment analyst was the sender in every pair (PR26).
HISTORICAL_FIXED_SENDER = "sentiment_analyst"

# Literal terms fixed in the E51 analysis plan. Matching is case-insensitive substring matching,
# including inside a longer word, to match the E51 runners.
PLAN_ECHO_TERMS = ("alarmed", "anxious", "frightened", "dread", "alarming", "anxiety")

_ARMS = ("neutral", "affective")
_EXPECTED_TEAMMATES = 5
_EXPECTED_ROLES = _EXPECTED_TEAMMATES + 1
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_LOW_INDEX = 50
_BOOTSTRAP_HIGH_INDEX = 1949
_HEX_DIGITS = frozenset("0123456789abcdef")

JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class ArchiveSummary:
    """Evidence that the outer manifest and archive agreed before analysis."""

    archive_file: str
    sha256: str
    runs: int
    pairs: int
    models: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class GateCheck:
    """Raw-generation equality counts for one pair."""

    pre_treatment_cells: int
    pre_treatment_identical: int
    treatment_round_peers: int
    treatment_round_peers_identical: int
    treated_cell_differs: bool

    @property
    def passed(self) -> bool:
        """Whether only the treated cell changed through the treatment round."""
        return (
            self.pre_treatment_identical == self.pre_treatment_cells
            and self.treatment_round_peers_identical == self.treatment_round_peers
            and self.treated_cell_differs
        )


@dataclass(frozen=True, slots=True)
class PairVerification:
    """Identity, gate, and parser facts for one complete matched pair."""

    run_ids: tuple[str, str]
    payload_json: str
    seed: int
    treated_role: str
    shock_round: int
    same_requested_trade: bool
    same_filled_position: bool
    same_observed_price: bool
    gate: GateCheck
    parser_fallbacks_neutral: int
    parser_fallbacks_affective: int

    @property
    def parser_fallbacks(self) -> int:
        """Return fallbacks across both arms and every round."""
        return self.parser_fallbacks_neutral + self.parser_fallbacks_affective


@dataclass(frozen=True, slots=True)
class IdentityCounts:
    """Planned trade, price, gate, and all-round parser counts."""

    pairs: int
    gate_passing: int
    gate_failures: int
    same_requested_trade: int
    same_requested_trade_gate_passing: int
    same_filled_position: int
    same_filled_position_gate_passing: int
    same_observed_price: int
    same_observed_price_gate_passing: int
    parser_fallbacks_neutral: int
    parser_fallbacks_affective: int
    parser_fallbacks_all_rounds: int


@dataclass(frozen=True, slots=True)
class PayloadIdentityCounts:
    """Identity counts for one exact payload rather than a filename-derived scenario."""

    payload_json: str
    counts: IdentityCounts


@dataclass(frozen=True, slots=True)
class IdentityReport:
    """Per-payload and pooled planned counts, with every failed gate retained."""

    by_payload: tuple[PayloadIdentityCounts, ...]
    pooled: IdentityCounts
    gate_failure_run_ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    """A paired mean and the frozen 2,000-resample percentile interval."""

    value: float
    interval: tuple[float, float]
    pairs: int


@dataclass(frozen=True, slots=True)
class TeammateEstimate:
    """A five-teammate sum, including its echo-drop denominator audit."""

    five_teammate_sum: float
    interval: tuple[float, float]
    retained_pairs: int
    retained_roles: int
    all_echo_pairs_omitted: int


@dataclass(frozen=True, slots=True)
class EchoComparison:
    """Full and echo-dropped versions of one teammate estimator."""

    full: TeammateEstimate
    echo_dropped: TeammateEstimate


@dataclass(frozen=True, slots=True)
class RoleEstimate:
    """One role's response over pairs where that role was not the sender."""

    value: float
    interval: tuple[float, float]
    pairs: int
    positive_pairs: int


@dataclass(frozen=True, slots=True)
class ClassifierEstimate:
    """Classifier discriminant per teammate and its echo-drop denominator audit."""

    value_per_teammate: float
    interval_per_teammate: tuple[float, float]
    retained_pairs: int
    retained_roles: int
    all_echo_pairs_omitted: int


@dataclass(frozen=True, slots=True)
class ClassifierComparison:
    """Full and echo-dropped classifier discriminants."""

    full: ClassifierEstimate
    echo_dropped: ClassifierEstimate


@dataclass(frozen=True, slots=True)
class InjectedVerification:
    """All independent checks and PR26 outcomes derivable from the release."""

    archive: ArchiveSummary
    pairs: tuple[PairVerification, ...]
    identity: IdentityReport
    outcome_4a_historical_fixed_sentiment: EchoComparison
    outcome_4b_by_manifest: EchoComparison
    outcome_5_treated_self: MeanEstimate
    corrected_4a_registered_matching: EchoComparison
    corrected_4b_registered_matching: EchoComparison
    corrected_5_registered_matching: MeanEstimate
    outcome_6_surface_only: dict[str, RoleEstimate]
    outcome_6_surface_then_suffix: dict[str, RoleEstimate]
    outcome_7_classifier: ClassifierComparison | None


@dataclass(frozen=True, slots=True)
class _Run:
    run_id: str
    manifest: JsonObject
    experiment: JsonObject
    payload_json: str
    records: tuple[JsonObject, ...]
    cells: dict[tuple[int, str], JsonObject]
    market: dict[int, JsonObject]
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Pair:
    neutral: _Run
    affective: _Run
    payload_json: str
    seed: int
    treated_role: str
    shock_round: int

    @property
    def run_ids(self) -> tuple[str, str]:
        ordered = sorted((self.neutral.run_id, self.affective.run_id))
        return ordered[0], ordered[1]


def verify_injected_release(
    archive_manifest: Path,
    vad_lexicon: Path,
    emotion_scorer: EmotionScorer | None = None,
) -> InjectedVerification:
    """Verify one complete-run release and independently derive every PR26 outcome.

    ``vad_lexicon`` is the signed NRC-VAD TSV. It is loaded twice so outcomes 4 and 5 use exact
    surface matching while outcome 6 is exposed under both surface-only and surface-then-suffix
    matching. Separate corrected fields apply the registered suffix matching to outcomes 4 and 5;
    they never replace the frozen surface-only results. Without an emotion scorer, outcome 7 is
    explicitly None. The command-line checker must report that omission.
    """
    surface = VADLexicon.from_tsv(vad_lexicon, scale="signed", match="surface")
    suffix = VADLexicon.from_tsv(vad_lexicon, scale="signed", match="surface_then_suffix")
    outer, runs, archive_file, digest, models = _load_archive(archive_manifest)
    del outer
    pairs = _make_pairs(runs)
    pair_rows = tuple(_verify_pair(pair) for pair in pairs)
    return InjectedVerification(
        archive=ArchiveSummary(
            archive_file=archive_file,
            sha256=digest,
            runs=len(runs),
            pairs=len(pairs),
            models=models,
        ),
        pairs=pair_rows,
        identity=_identity_report(pair_rows),
        outcome_4a_historical_fixed_sentiment=_teammate_comparison(
            pairs, surface, historical_fixed_sender=True
        ),
        outcome_4b_by_manifest=_teammate_comparison(pairs, surface, historical_fixed_sender=False),
        outcome_5_treated_self=_treated_self(pairs, surface),
        corrected_4a_registered_matching=_teammate_comparison(
            pairs, suffix, historical_fixed_sender=True
        ),
        corrected_4b_registered_matching=_teammate_comparison(
            pairs, suffix, historical_fixed_sender=False
        ),
        corrected_5_registered_matching=_treated_self(pairs, suffix),
        outcome_6_surface_only=_per_role(pairs, surface),
        outcome_6_surface_then_suffix=_per_role(pairs, suffix),
        outcome_7_classifier=(
            None if emotion_scorer is None else _classifier_comparison(pairs, emotion_scorer)
        ),
    )


def _load_archive(
    manifest_path: Path,
) -> tuple[JsonObject, tuple[_Run, ...], str, str, tuple[tuple[str, str], ...]]:
    outer = _read_json_object(manifest_path, "archive manifest")
    if outer.get("schema_version") != 2 or outer.get("kind") != "complete_synthetic_runs":
        msg = "archive manifest must be schema version 2 and kind complete_synthetic_runs"
        raise ValueError(msg)
    runs_entry = _object(outer.get("runs"), "archive manifest runs")
    archive_file = _string(runs_entry.get("file"), "archive filename")
    if Path(archive_file).name != archive_file or archive_file in {".", ".."}:
        msg = "archive filename must be one safe path component"
        raise ValueError(msg)
    expected_digest = _sha256(runs_entry.get("sha256"), "archive sha256")
    expected_rows = _integer(runs_entry.get("rows"), "archive row count", minimum=1)
    archive_path = manifest_path.parent / archive_file
    try:
        archive_bytes = archive_path.read_bytes()
    except OSError as error:
        msg = f"cannot read archive {archive_file}"
        raise ValueError(msg) from error
    digest = hashlib.sha256(archive_bytes).hexdigest()
    if digest != expected_digest:
        msg = f"archive {archive_file} does not match its manifest hash"
        raise ValueError(msg)
    rows = _read_gzip_rows(archive_path)
    if len(rows) != expected_rows:
        msg = f"archive declares {expected_rows} rows but contains {len(rows)}"
        raise ValueError(msg)
    runs = tuple(_validate_run(row, number) for number, row in enumerate(rows, start=1))
    if len({run.run_id for run in runs}) != len(runs):
        msg = "archive contains duplicate run IDs"
        raise ValueError(msg)
    outer_models = _outer_model_identities(outer.get("models"))
    roster_models = tuple(
        sorted(
            {
                identity
                for run in runs
                for identity in _roster_model_identities(run.manifest, run.run_id)
            }
        )
    )
    if outer_models != roster_models:
        msg = "outer model identities do not equal the archived roster identities"
        raise ValueError(msg)
    return outer, runs, archive_file, digest, outer_models


def _read_gzip_rows(path: Path) -> list[JsonObject]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            values = [json.loads(line) for line in stream if line.strip()]
    except (OSError, UnicodeDecodeError, JSONDecodeError) as error:
        msg = f"archive {path.name} is not valid gzipped JSONL"
        raise ValueError(msg) from error
    return [_object(value, f"archive row {index}") for index, value in enumerate(values, start=1)]


def _validate_run(row: JsonObject, row_number: int) -> _Run:
    where = f"archive row {row_number}"
    run_id = _string(row.get("run_id"), f"{where} run_id")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        msg = f"{where} has an unsafe run_id"
        raise ValueError(msg)
    manifest = _object(row.get("manifest"), f"{run_id} manifest")
    experiment, payload_json, roles, n_rounds = _validate_run_manifest(run_id, manifest)
    records = tuple(
        _object(value, f"{run_id} record")
        for value in _list(row.get("records"), f"{run_id} records")
    )
    cells = _validate_records(run_id, records, roles, n_rounds, manifest)
    market = _validate_market(
        run_id,
        _list(row.get("market"), f"{run_id} market"),
        n_rounds,
    )
    _validate_status_and_source_hashes(run_id, row)
    return _Run(run_id, manifest, experiment, payload_json, records, cells, market, roles)


def _validate_run_manifest(
    run_id: str, manifest: JsonObject
) -> tuple[JsonObject, str, tuple[str, ...], int]:
    if manifest.get("run_id") != run_id or manifest.get("status") != "completed":
        msg = f"{run_id}: incomplete run or inconsistent manifest run ID"
        raise ValueError(msg)
    _integer(manifest.get("schema_version"), f"{run_id} manifest schema", minimum=1)
    payload = _object(manifest.get("payload"), f"{run_id} payload")
    if payload.get("synthetic") is not True:
        msg = f"{run_id}: payload is not explicitly synthetic"
        raise ValueError(msg)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    experiment = _object(manifest.get("experiment"), f"{run_id} experiment")
    roles = tuple(
        _string(role, f"{run_id} experiment role")
        for role in _list(experiment.get("roles"), f"{run_id} experiment roles")
    )
    if len(roles) != _EXPECTED_ROLES or len(set(roles)) != len(roles):
        msg = f"{run_id}: injected verification requires six unique roles"
        raise ValueError(msg)
    if HISTORICAL_FIXED_SENDER not in roles:
        msg = f"{run_id}: historical comparator role is absent"
        raise ValueError(msg)
    treated = _string(experiment.get("treated_role"), f"{run_id} treated role")
    if treated not in roles:
        msg = f"{run_id}: treated role is not in experiment roles"
        raise ValueError(msg)
    n_rounds = _integer(experiment.get("n_rounds"), f"{run_id} n_rounds", minimum=1)
    shock_round = _integer(experiment.get("shock_round"), f"{run_id} shock_round", minimum=1)
    if shock_round + 1 >= n_rounds:
        msg = f"{run_id}: shock needs a preceding and response round"
        raise ValueError(msg)
    _integer(experiment.get("seed"), f"{run_id} seed")
    arm = _string(experiment.get("shock_type"), f"{run_id} shock type")
    if arm not in _ARMS:
        msg = f"{run_id}: arm must be neutral or affective"
        raise ValueError(msg)
    _object(manifest.get("stimuli"), f"{run_id} stimuli")
    if "intervention_seed" not in manifest:
        msg = f"{run_id}: manifest lacks intervention_seed"
        raise ValueError(msg)
    return experiment, payload_json, roles, n_rounds


def _validate_status_and_source_hashes(run_id: str, row: JsonObject) -> None:
    statuses = [
        _object(value, f"{run_id} status row")
        for value in _list(row.get("status"), f"{run_id} status")
    ]
    if not statuses or statuses[-1].get("status") != "completed":
        msg = f"{run_id}: lifecycle does not end in completed"
        raise ValueError(msg)
    source_hashes = _object(row.get("source_sha256"), f"{run_id} source_sha256")
    required_sources = {"manifest.json", "records.jsonl", "market.jsonl", "status.jsonl"}
    if set(source_hashes) != required_sources:
        msg = f"{run_id}: source hashes do not cover the four archived source files"
        raise ValueError(msg)
    for name, value in source_hashes.items():
        _sha256(value, f"{run_id} source hash for {name}")


def _validate_records(
    run_id: str,
    records: tuple[JsonObject, ...],
    roles: tuple[str, ...],
    n_rounds: int,
    manifest: JsonObject,
) -> dict[tuple[int, str], JsonObject]:
    expected = {(round_index, role) for round_index in range(n_rounds) for role in roles}
    cells: dict[tuple[int, str], JsonObject] = {}
    model_for_role = _roster_role_models(manifest, roles, run_id)
    for record in records:
        round_index = _integer(record.get("round_index"), f"{run_id} record round", minimum=0)
        role = _string(record.get("role"), f"{run_id} record role")
        cell = (round_index, role)
        if cell in cells:
            msg = f"{run_id}: duplicate generation cell {cell}"
            raise ValueError(msg)
        agent_index = _integer(record.get("agent_index"), f"{run_id} agent index", minimum=0)
        if agent_index >= len(roles) or roles[agent_index] != role:
            msg = f"{run_id}: role and agent index disagree at {cell}"
            raise ValueError(msg)
        generation = _object(record.get("generation"), f"{run_id} generation at {cell}")
        _string(
            generation.get("raw_output"), f"{run_id} raw generation at {cell}", allow_empty=True
        )
        model_key, repository, revision = model_for_role[role]
        if (
            generation.get("model_key") != model_key
            or generation.get("repository") != repository
            or generation.get("revision") != revision
        ):
            msg = f"{run_id}: generation model identity disagrees with roster at {cell}"
            raise ValueError(msg)
        decision = _object(record.get("decision"), f"{run_id} decision at {cell}")
        _string(decision.get("message"), f"{run_id} decision message at {cell}", allow_empty=True)
        action = _string(decision.get("action"), f"{run_id} decision action at {cell}")
        if action not in {"BUY", "SELL", "HOLD"}:
            msg = f"{run_id}: invalid action at {cell}"
            raise ValueError(msg)
        _number(decision.get("size"), f"{run_id} decision size at {cell}")
        _number(record.get("position_after"), f"{run_id} position at {cell}")
        if not isinstance(record.get("parse_succeeded"), bool):
            msg = f"{run_id}: parse_succeeded is not Boolean at {cell}"
            raise TypeError(msg)
        cells[cell] = record
    if set(cells) != expected:
        missing = len(expected - set(cells))
        unexpected = len(set(cells) - expected)
        msg = f"{run_id}: incomplete generation grid ({missing} missing, {unexpected} unexpected)"
        raise ValueError(msg)
    return cells


def _validate_market(run_id: str, values: list[object], n_rounds: int) -> dict[int, JsonObject]:
    market: dict[int, JsonObject] = {}
    for value in values:
        row = _object(value, f"{run_id} market row")
        round_index = _integer(row.get("round_index"), f"{run_id} market round", minimum=0)
        if round_index in market:
            msg = f"{run_id}: duplicate market round {round_index}"
            raise ValueError(msg)
        _number(row.get("price"), f"{run_id} market price at round {round_index}")
        market[round_index] = row
    if set(market) != set(range(n_rounds)):
        msg = f"{run_id}: incomplete market rounds"
        raise ValueError(msg)
    return market


def _make_pairs(runs: tuple[_Run, ...]) -> tuple[_Pair, ...]:
    grouped: dict[tuple[str, int], dict[str, _Run]] = defaultdict(dict)
    for run in sorted(runs, key=lambda item: item.run_id):
        seed = _integer(run.experiment.get("seed"), f"{run.run_id} seed")
        arm = _string(run.experiment.get("shock_type"), f"{run.run_id} arm")
        key = (run.payload_json, seed)
        if arm in grouped[key]:
            msg = f"{run.run_id}: duplicate {arm} arm for one payload and seed"
            raise ValueError(msg)
        grouped[key][arm] = run
    pairs: list[_Pair] = []
    for (payload_json, seed), arms in grouped.items():
        if set(arms) != set(_ARMS):
            msg = f"payload/seed pair {payload_json}/{seed} lacks one complete arm"
            raise ValueError(msg)
        neutral, affective = arms["neutral"], arms["affective"]
        _validate_pair_manifests(neutral, affective)
        pairs.append(
            _Pair(
                neutral=neutral,
                affective=affective,
                payload_json=payload_json,
                seed=seed,
                treated_role=_string(neutral.experiment.get("treated_role"), "treated role"),
                shock_round=_integer(neutral.experiment.get("shock_round"), "shock round"),
            )
        )
    return tuple(sorted(pairs, key=lambda pair: pair.run_ids))


def _validate_pair_manifests(neutral: _Run, affective: _Run) -> None:
    configs = []
    for experiment in (neutral.experiment, affective.experiment):
        config = dict(experiment)
        config.pop("shock_type", None)
        configs.append(config)
    if configs[0] != configs[1]:
        msg = f"{neutral.run_id}/{affective.run_id}: configs differ beyond shock_type"
        raise ValueError(msg)
    for field in ("payload", "roster", "stimuli", "intervention_seed"):
        if neutral.manifest.get(field) != affective.manifest.get(field):
            msg = f"{neutral.run_id}/{affective.run_id}: paired {field} manifests differ"
            raise ValueError(msg)


def _verify_pair(pair: _Pair) -> PairVerification:
    neutral, affective = pair.neutral, pair.affective
    shock = pair.shock_round
    before = shock - 1
    role = pair.treated_role
    n_now = neutral.cells[(shock, role)]
    a_now = affective.cells[(shock, role)]
    n_before = neutral.cells[(before, role)]
    a_before = affective.cells[(before, role)]
    gate = _gate(pair)
    return PairVerification(
        run_ids=pair.run_ids,
        payload_json=pair.payload_json,
        seed=pair.seed,
        treated_role=role,
        shock_round=shock,
        same_requested_trade=_requested_trade(n_now) == _requested_trade(a_now),
        same_filled_position=(
            _position(n_now) == _position(a_now) and _position(n_before) == _position(a_before)
        ),
        same_observed_price=_price(neutral, shock) == _price(affective, shock),
        gate=gate,
        parser_fallbacks_neutral=sum(
            record.get("parse_succeeded") is False for record in neutral.records
        ),
        parser_fallbacks_affective=sum(
            record.get("parse_succeeded") is False for record in affective.records
        ),
    )


def _gate(pair: _Pair) -> GateCheck:
    shock = pair.shock_round
    treated = pair.treated_role
    cells = tuple(pair.neutral.cells)
    pre = tuple(cell for cell in cells if cell[0] < shock)
    peers = tuple(cell for cell in cells if cell[0] == shock and cell[1] != treated)
    return GateCheck(
        pre_treatment_cells=len(pre),
        pre_treatment_identical=sum(
            _raw(pair.neutral.cells[cell]) == _raw(pair.affective.cells[cell]) for cell in pre
        ),
        treatment_round_peers=len(peers),
        treatment_round_peers_identical=sum(
            _raw(pair.neutral.cells[cell]) == _raw(pair.affective.cells[cell]) for cell in peers
        ),
        treated_cell_differs=(
            _raw(pair.neutral.cells[(shock, treated)])
            != _raw(pair.affective.cells[(shock, treated)])
        ),
    )


def _identity_report(rows: tuple[PairVerification, ...]) -> IdentityReport:
    grouped: dict[str, list[PairVerification]] = defaultdict(list)
    for row in rows:
        grouped[row.payload_json].append(row)
    return IdentityReport(
        by_payload=tuple(
            PayloadIdentityCounts(payload_json, _identity_counts(tuple(group)))
            for payload_json, group in sorted(grouped.items())
        ),
        pooled=_identity_counts(rows),
        gate_failure_run_ids=tuple(row.run_ids for row in rows if not row.gate.passed),
    )


def _identity_counts(rows: tuple[PairVerification, ...]) -> IdentityCounts:
    passing = tuple(row for row in rows if row.gate.passed)
    return IdentityCounts(
        pairs=len(rows),
        gate_passing=len(passing),
        gate_failures=len(rows) - len(passing),
        same_requested_trade=sum(row.same_requested_trade for row in rows),
        same_requested_trade_gate_passing=sum(row.same_requested_trade for row in passing),
        same_filled_position=sum(row.same_filled_position for row in rows),
        same_filled_position_gate_passing=sum(row.same_filled_position for row in passing),
        same_observed_price=sum(row.same_observed_price for row in rows),
        same_observed_price_gate_passing=sum(row.same_observed_price for row in passing),
        parser_fallbacks_neutral=sum(row.parser_fallbacks_neutral for row in rows),
        parser_fallbacks_affective=sum(row.parser_fallbacks_affective for row in rows),
        parser_fallbacks_all_rounds=sum(row.parser_fallbacks for row in rows),
    )


def _teammate_comparison(
    pairs: tuple[_Pair, ...],
    lexicon: VADLexicon,
    *,
    historical_fixed_sender: bool,
) -> EchoComparison:
    return EchoComparison(
        full=_teammate_estimate(
            pairs,
            lexicon,
            historical_fixed_sender=historical_fixed_sender,
            drop_echo=False,
        ),
        echo_dropped=_teammate_estimate(
            pairs,
            lexicon,
            historical_fixed_sender=historical_fixed_sender,
            drop_echo=True,
        ),
    )


def _teammate_estimate(
    pairs: tuple[_Pair, ...],
    lexicon: VADLexicon,
    *,
    historical_fixed_sender: bool,
    drop_echo: bool,
) -> TeammateEstimate:
    values: list[float] = []
    retained_roles = 0
    omitted = 0
    for pair in pairs:
        excluded = HISTORICAL_FIXED_SENDER if historical_fixed_sender else pair.treated_role
        roles = [role for role in pair.neutral.roles if role != excluded]
        if drop_echo:
            roles = [
                role
                for role in roles
                if not _contains_echo(_message(pair.affective, pair.shock_round + 1, role))
            ]
        if not roles:
            omitted += 1
            continue
        total = sum(
            _arousal(lexicon, _message(pair.affective, pair.shock_round + 1, role))
            - _arousal(lexicon, _message(pair.neutral, pair.shock_round + 1, role))
            for role in roles
        )
        values.append(total * _EXPECTED_TEAMMATES / len(roles))
        retained_roles += len(roles)
    mean, interval = _mean_interval(values, "teammate response")
    return TeammateEstimate(mean, interval, len(values), retained_roles, omitted)


def _treated_self(pairs: tuple[_Pair, ...], lexicon: VADLexicon) -> MeanEstimate:
    values = [
        _arousal(lexicon, _message(pair.affective, pair.shock_round + 1, pair.treated_role))
        - _arousal(lexicon, _message(pair.neutral, pair.shock_round + 1, pair.treated_role))
        for pair in pairs
    ]
    mean, interval = _mean_interval(values, "treated self response")
    return MeanEstimate(mean, interval, len(values))


def _per_role(pairs: tuple[_Pair, ...], lexicon: VADLexicon) -> dict[str, RoleEstimate]:
    values: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        for role in pair.neutral.roles:
            if role == pair.treated_role:
                continue
            values[role].append(
                _arousal(lexicon, _message(pair.affective, pair.shock_round + 1, role))
                - _arousal(lexicon, _message(pair.neutral, pair.shock_round + 1, role))
            )
    out: dict[str, RoleEstimate] = {}
    for role, role_values in sorted(values.items()):
        mean, interval = _mean_interval(role_values, f"per-role response for {role}")
        out[role] = RoleEstimate(
            value=mean,
            interval=interval,
            pairs=len(role_values),
            positive_pairs=sum(value > 0 for value in role_values),
        )
    return out


def _classifier_comparison(pairs: tuple[_Pair, ...], scorer: EmotionScorer) -> ClassifierComparison:
    texts: list[str] = []
    keys: list[tuple[int, str, str]] = []
    for pair_index, pair in enumerate(pairs):
        for run in sorted((pair.neutral, pair.affective), key=lambda item: item.run_id):
            arm = _string(run.experiment.get("shock_type"), f"{run.run_id} arm")
            for role in pair.neutral.roles:
                if role != pair.treated_role:
                    texts.append(_message(run, pair.shock_round + 1, role))
                    keys.append((pair_index, arm, role))
    scored = scorer.score_all(tuple(texts))
    if len(scored) != len(texts):
        msg = f"emotion scorer returned {len(scored)} scores for {len(texts)} texts"
        raise ValueError(msg)
    scores: dict[tuple[int, str, str], float] = {}
    for key, score in zip(keys, scored, strict=True):
        value = score.discriminant
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            msg = "emotion scorer returned a non-finite numeric discriminant"
            raise ValueError(msg)
        scores[key] = float(value)
    return ClassifierComparison(
        full=_classifier_estimate(pairs, scores, drop_echo=False),
        echo_dropped=_classifier_estimate(pairs, scores, drop_echo=True),
    )


def _classifier_estimate(
    pairs: tuple[_Pair, ...],
    scores: dict[tuple[int, str, str], float],
    *,
    drop_echo: bool,
) -> ClassifierEstimate:
    five_role_sums: list[float] = []
    retained_roles = 0
    omitted = 0
    for pair_index, pair in enumerate(pairs):
        roles = [role for role in pair.neutral.roles if role != pair.treated_role]
        if drop_echo:
            roles = [
                role
                for role in roles
                if not _contains_echo(_message(pair.affective, pair.shock_round + 1, role))
            ]
        if not roles:
            omitted += 1
            continue
        total = sum(
            scores[(pair_index, "affective", role)] - scores[(pair_index, "neutral", role)]
            for role in roles
        )
        five_role_sums.append(total * _EXPECTED_TEAMMATES / len(roles))
        retained_roles += len(roles)
    mean, interval = _mean_interval(five_role_sums, "classifier response")
    return ClassifierEstimate(
        value_per_teammate=mean / _EXPECTED_TEAMMATES,
        interval_per_teammate=(
            interval[0] / _EXPECTED_TEAMMATES,
            interval[1] / _EXPECTED_TEAMMATES,
        ),
        retained_pairs=len(five_role_sums),
        retained_roles=retained_roles,
        all_echo_pairs_omitted=omitted,
    )


def _mean_interval(values: Sequence[float], label: str) -> tuple[float, tuple[float, float]]:
    if not values:
        msg = f"no pairs remain for {label}"
        raise ValueError(msg)
    return statistics.fmean(values), _bootstrap(values)


def _bootstrap(values: Sequence[float]) -> tuple[float, float]:
    """Reproduce the frozen random.Random(0) mean bootstrap independently."""
    if not values:
        msg = "bootstrap needs at least one value"
        raise ValueError(msg)
    rng = random.Random(0)  # noqa: S311
    draws = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    return draws[_BOOTSTRAP_LOW_INDEX], draws[_BOOTSTRAP_HIGH_INDEX]


def _roster_model_identities(manifest: JsonObject, run_id: str) -> set[tuple[str, str]]:
    roster = _object(manifest.get("roster"), f"{run_id} roster")
    models = _list(roster.get("models"), f"{run_id} roster models")
    identities: set[tuple[str, str]] = set()
    keys: set[str] = set()
    for value in models:
        model = _object(value, f"{run_id} roster model")
        key = _string(model.get("key"), f"{run_id} model key")
        repository = _string(model.get("repository"), f"{run_id} model repository")
        revision = _string(model.get("revision"), f"{run_id} model revision")
        if key in keys:
            msg = f"{run_id}: duplicate roster model key"
            raise ValueError(msg)
        keys.add(key)
        identities.add((repository, revision))
    if not identities:
        msg = f"{run_id}: roster has no models"
        raise ValueError(msg)
    return identities


def _roster_role_models(
    manifest: JsonObject, roles: tuple[str, ...], run_id: str
) -> dict[str, tuple[str, str, str]]:
    roster = _object(manifest.get("roster"), f"{run_id} roster")
    models: dict[str, tuple[str, str]] = {}
    for value in _list(roster.get("models"), f"{run_id} roster models"):
        model = _object(value, f"{run_id} roster model")
        key = _string(model.get("key"), f"{run_id} model key")
        if key in models:
            msg = f"{run_id}: duplicate roster model key"
            raise ValueError(msg)
        models[key] = (
            _string(model.get("repository"), f"{run_id} model repository"),
            _string(model.get("revision"), f"{run_id} model revision"),
        )
    assigned: dict[str, tuple[str, str, str]] = {}
    for value in _list(roster.get("agents"), f"{run_id} roster agents"):
        agent = _object(value, f"{run_id} roster agent")
        role = _string(agent.get("role"), f"{run_id} roster agent role")
        key = _string(agent.get("model_key"), f"{run_id} roster agent model key")
        if role in assigned or key not in models:
            msg = f"{run_id}: invalid duplicate or missing roster assignment"
            raise ValueError(msg)
        repository, revision = models[key]
        assigned[role] = (key, repository, revision)
    if set(assigned) != set(roles):
        msg = f"{run_id}: roster roles do not equal experiment roles"
        raise ValueError(msg)
    return assigned


def _outer_model_identities(value: object) -> tuple[tuple[str, str], ...]:
    identities = []
    for item in _list(value, "outer manifest models"):
        model = _object(item, "outer manifest model")
        identities.append(
            (
                _string(model.get("repository"), "outer model repository"),
                _string(model.get("revision"), "outer model revision"),
            )
        )
    if not identities or len(set(identities)) != len(identities):
        msg = "outer manifest models must be non-empty and unique"
        raise ValueError(msg)
    return tuple(sorted(identities))


def _requested_trade(record: JsonObject) -> tuple[str, float]:
    decision = _object(record.get("decision"), "decision")
    return (
        _string(decision.get("action"), "decision action"),
        _number(decision.get("size"), "decision size"),
    )


def _position(record: JsonObject) -> float:
    return _number(record.get("position_after"), "position_after")


def _price(run: _Run, round_index: int) -> float:
    return _number(run.market[round_index].get("price"), "market price")


def _raw(record: JsonObject) -> str:
    generation = _object(record.get("generation"), "generation")
    return _string(generation.get("raw_output"), "raw_output", allow_empty=True)


def _message(run: _Run, round_index: int, role: str) -> str:
    decision = _object(run.cells[(round_index, role)].get("decision"), "decision")
    return _string(decision.get("message"), "decision message", allow_empty=True)


def _arousal(lexicon: VADLexicon, text: str) -> float:
    return float(lexicon.score(scrub_decision_vocabulary(text)).arousal)


def _contains_echo(text: str) -> bool:
    folded = text.casefold()
    return any(term in folded for term in PLAN_ECHO_TERMS)


def _read_json_object(path: Path, label: str) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, JSONDecodeError) as error:
        msg = f"{label} is not readable JSON"
        raise ValueError(msg) from error
    return _object(value, label)


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"{label} must be a JSON array"
        raise TypeError(msg)
    return value


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        msg = f"{label} must be a string"
        raise ValueError(msg)
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    if minimum is not None and value < minimum:
        msg = f"{label} must be at least {minimum}"
        raise ValueError(msg)
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = f"{label} must be a finite number"
        raise ValueError(msg)
    return float(value)


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label).lower()
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        msg = f"{label} must be a 64-character hexadecimal digest"
        raise ValueError(msg)
    return digest
