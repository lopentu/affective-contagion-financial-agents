"""Verify and analyse released isolated-reader outcomes with paired Student t tests."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

from scipy import stats

DEFAULT_FIELDS = ("size", "downside_risk", "confidence")
NEUTRAL_CONDITION = "neutral"
AFFECTIVE_CONDITION = "anxiety_like"
ALPHA = 0.05


@dataclass(frozen=True, slots=True)
class StudentTEstimate:
    """One mean difference with a two-sided Student t interval and test."""

    mean: float
    standard_deviation: float
    standard_error: float
    degrees_of_freedom: int
    t_statistic: float
    p_value: float
    interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RoleOutcome:
    """One role's paired outcome and its across-role Holm result."""

    role: str
    pairs: int
    estimate: StudentTEstimate
    holm_adjusted_p: float = 1.0
    passes_holm: bool = False


@dataclass(frozen=True, slots=True)
class BuyMechanismDiagnostic:
    """PR22's BUY-only diagnostic, evaluated after the reported-size Holm gate."""

    scope: str
    qualifying_size_roles: tuple[str, ...]
    discounting_roles: tuple[str, ...]
    risk_aversion_roles: tuple[str, ...]
    required_agreeing_roles: int
    requires_no_opposed_roles: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class ReaderOutcomeReport:
    """Verified release metadata, diagnostics, and outcomes."""

    schema_version: int
    name: str
    manifest_file: str
    archive_file: str
    archive_sha256: str
    rows: int
    seeds: tuple[int, ...]
    roles: tuple[str, ...]
    conditions: tuple[str, ...]
    draws: tuple[int, ...]
    repository: str
    revision: str
    fields: tuple[str, ...]
    effect_direction: str
    raw_size_includes_hold: bool
    size_measurement: str
    nonzero_hold_sizes: int
    action_counts: dict[str, dict[str, int]]
    outcomes: dict[str, tuple[RoleOutcome, ...]]
    buy_mechanism: BuyMechanismDiagnostic | None


@dataclass(frozen=True, slots=True)
class _Reading:
    seed: int
    role: str
    condition: str
    draw: int
    action: str
    size: float
    values: dict[str, float]


def student_t_estimate(
    differences: Sequence[float], *, confidence: float = 0.95
) -> StudentTEstimate:
    """Return an exact two-sided Student t interval and test for paired differences."""
    values = tuple(float(value) for value in differences)
    if len(values) < 2:
        msg = "a Student t estimate requires at least two paired differences"
        raise ValueError(msg)
    if not 0 < confidence < 1:
        msg = "confidence must be between zero and one"
        raise ValueError(msg)
    if any(not math.isfinite(value) for value in values):
        msg = "paired differences must all be finite"
        raise ValueError(msg)

    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = float(stats.sem(values))
    degrees_of_freedom = len(values) - 1
    if standard_error == 0.0:
        t_statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p_value = 1.0 if mean == 0.0 else 0.0
        interval = (mean, mean)
    else:
        t_statistic = mean / standard_error
        p_value = float(2 * stats.t.sf(abs(t_statistic), degrees_of_freedom))
        critical = float(stats.t.ppf(1 - (1 - confidence) / 2, degrees_of_freedom))
        half_width = critical * standard_error
        interval = (mean - half_width, mean + half_width)
    return StudentTEstimate(
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        t_statistic=t_statistic,
        p_value=p_value,
        interval=interval,
    )


def _require_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    return value


def _require_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer"
        raise TypeError(msg)
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{label} must be numeric"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{label} must be finite"
        raise ValueError(msg)
    return number


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"could not read release manifest {path}: {error}"
        raise ValueError(msg) from error
    manifest = _require_object(parsed, "release manifest")
    if manifest.get("schema_version") != 1:
        msg = f"unsupported release manifest schema: {manifest.get('schema_version')!r}"
        raise ValueError(msg)
    return manifest


def _load_rows(archive: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    msg = f"{archive} contains a blank row at line {line_number}"
                    raise ValueError(msg)
                try:
                    parsed: object = json.loads(line)
                except json.JSONDecodeError as error:
                    msg = f"{archive} has invalid JSON at line {line_number}: {error}"
                    raise ValueError(msg) from error
                rows.append(_require_object(parsed, f"archive row {line_number}"))
    except OSError as error:
        msg = f"could not read release archive {archive}: {error}"
        raise ValueError(msg) from error
    return rows


def _manifest_seeds(manifest: dict[str, object]) -> tuple[int, ...]:
    raw = manifest.get("seeds")
    if not isinstance(raw, list) or not raw:
        msg = "manifest seeds must be a non-empty list"
        raise ValueError(msg)
    seeds = tuple(_require_integer(value, "manifest seed") for value in raw)
    if len(set(seeds)) != len(seeds):
        msg = "manifest seeds must be unique"
        raise ValueError(msg)
    return seeds


def _normalize_reading(
    row: dict[str, object],
    *,
    index: int,
    repository: str,
    revision: str,
    fields: tuple[str, ...],
) -> _Reading:
    """Validate and normalize one archived reading."""
    prefix = f"row {index}"
    seed = _require_integer(row.get("seed"), f"{prefix} seed")
    role = _require_string(row.get("reader_role"), f"{prefix} reader_role")
    condition = _require_string(row.get("condition"), f"{prefix} condition")
    arm = _require_string(row.get("arm"), f"{prefix} arm")
    if arm != condition:
        msg = f"{prefix} condition {condition!r} does not match arm {arm!r}"
        raise ValueError(msg)
    if row.get("repository") != repository or row.get("revision") != revision:
        msg = f"{prefix} model repository or revision does not match the manifest"
        raise ValueError(msg)
    draw = _require_integer(row.get("draw"), f"{prefix} draw")
    action = _require_string(row.get("action"), f"{prefix} action")
    if action not in {"BUY", "SELL", "HOLD"}:
        msg = f"{prefix} action must be BUY, SELL, or HOLD"
        raise ValueError(msg)
    size = _require_number(row.get("size"), f"{prefix} size")
    values = {field: _require_number(row.get(field), f"{prefix} {field}") for field in fields}
    return _Reading(seed, role, condition, draw, action, size, values)


def _validate_rows(
    raw_rows: Sequence[dict[str, object]],
    *,
    seeds: tuple[int, ...],
    repository: str,
    revision: str,
    fields: tuple[str, ...],
) -> tuple[tuple[_Reading, ...], tuple[str, ...], tuple[int, ...]]:
    readings = [
        _normalize_reading(
            row,
            index=index,
            repository=repository,
            revision=revision,
            fields=fields,
        )
        for index, row in enumerate(raw_rows, start=1)
    ]

    actual_seeds = {reading.seed for reading in readings}
    if actual_seeds != set(seeds):
        msg = "archive seeds do not match the manifest seeds"
        raise ValueError(msg)
    conditions = tuple(sorted({reading.condition for reading in readings}))
    required_conditions = {NEUTRAL_CONDITION, AFFECTIVE_CONDITION}
    if set(conditions) != required_conditions:
        msg = f"archive conditions must be {sorted(required_conditions)}, got {list(conditions)}"
        raise ValueError(msg)
    roles = tuple(sorted({reading.role for reading in readings}))
    if not roles:
        msg = "archive contains no reader roles"
        raise ValueError(msg)

    cells: dict[tuple[int, str, str], dict[int, _Reading]] = defaultdict(dict)
    for reading in readings:
        cell = cells[(reading.seed, reading.role, reading.condition)]
        if reading.draw in cell:
            msg = (
                f"duplicate draw {reading.draw} for seed {reading.seed}, role {reading.role!r}, "
                f"condition {reading.condition!r}"
            )
            raise ValueError(msg)
        cell[reading.draw] = reading
    expected_cells = set(product(seeds, roles, conditions))
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        extra = sorted(set(cells) - expected_cells)
        msg = f"archive has incomplete paired cells; missing={missing[:3]}, extra={extra[:3]}"
        raise ValueError(msg)
    draw_sets = {tuple(sorted(cell)) for cell in cells.values()}
    if len(draw_sets) != 1:
        msg = f"paired cells use inconsistent draw sets: {sorted(draw_sets)}"
        raise ValueError(msg)
    draws = next(iter(draw_sets))
    if not draws:
        msg = "paired cells contain no draws"
        raise ValueError(msg)
    return tuple(readings), roles, draws


def _select_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Require a non-empty unique list of numeric outcome fields."""
    selected = tuple(fields)
    if not selected or any(not field for field in selected):
        msg = "fields must contain at least one non-empty field name"
        raise ValueError(msg)
    if len(set(selected)) != len(selected):
        msg = "fields must be unique"
        raise ValueError(msg)
    return selected


def _holm(outcomes: Sequence[RoleOutcome], *, alpha: float = ALPHA) -> tuple[RoleOutcome, ...]:
    ordered = sorted(outcomes, key=lambda outcome: (outcome.estimate.p_value, outcome.role))
    count = len(ordered)
    running_adjusted = 0.0
    still_rejecting = True
    corrected: dict[str, tuple[float, bool]] = {}
    for index, outcome in enumerate(ordered):
        remaining = count - index
        adjusted = min(1.0, max(running_adjusted, remaining * outcome.estimate.p_value))
        running_adjusted = adjusted
        passes = still_rejecting and outcome.estimate.p_value <= alpha / remaining
        still_rejecting = still_rejecting and passes
        corrected[outcome.role] = (adjusted, passes)
    return tuple(
        replace(
            outcome,
            holm_adjusted_p=corrected[outcome.role][0],
            passes_holm=corrected[outcome.role][1],
        )
        for outcome in sorted(outcomes, key=lambda item: item.role)
    )


def _buy_mechanism(outcomes: dict[str, tuple[RoleOutcome, ...]]) -> BuyMechanismDiagnostic:
    """Apply PR22's BUY-event explanation criterion without correcting belief intervals."""
    required = set(DEFAULT_FIELDS)
    if not required.issubset(outcomes):
        msg = f"BUY mechanism diagnostic requires fields {sorted(required)}"
        raise ValueError(msg)
    indexed = {
        field: {outcome.role: outcome for outcome in outcomes[field]} for field in DEFAULT_FIELDS
    }
    qualifying = tuple(
        sorted(
            role
            for role, outcome in indexed["size"].items()
            if outcome.passes_holm and outcome.estimate.mean < 0
        )
    )
    discounting = tuple(
        role
        for role in qualifying
        if indexed["downside_risk"][role].estimate.interval[1] < 0
        and indexed["confidence"][role].estimate.interval[0] > 0
    )
    risk_aversion = tuple(
        role
        for role in qualifying
        if indexed["downside_risk"][role].estimate.interval[0] > 0
        and indexed["confidence"][role].estimate.interval[1] < 0
    )
    if len(discounting) >= 2 and not risk_aversion:
        verdict = "discounting_supported"
    elif len(risk_aversion) >= 2 and not discounting:
        verdict = "risk_aversion_supported"
    else:
        verdict = "unresolved"
    return BuyMechanismDiagnostic(
        scope=(
            "PR22 BUY-event diagnostic: belief directions use ordinary 95% intervals only after "
            "a role's reported-size decrease passes Holm"
        ),
        qualifying_size_roles=qualifying,
        discounting_roles=discounting,
        risk_aversion_roles=risk_aversion,
        required_agreeing_roles=2,
        requires_no_opposed_roles=True,
        verdict=verdict,
    )


def analyze_reader_outcomes(
    manifest_path: Path,
    *,
    fields: Sequence[str] = DEFAULT_FIELDS,
    include_buy_mechanism: bool = False,
) -> ReaderOutcomeReport:
    """Verify one schema-1 release and analyse affective-minus-neutral reader outcomes."""
    selected_fields = _select_fields(fields)

    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    name = _require_string(manifest.get("name"), "manifest name")
    readings = _require_object(manifest.get("readings"), "manifest readings")
    archive_name = _require_string(readings.get("file"), "readings file")
    archive = (manifest_path.parent / archive_name).resolve()
    expected_hash = _require_string(readings.get("sha256"), "readings sha256")
    expected_rows = _require_integer(readings.get("rows"), "readings rows")
    try:
        archive_bytes = archive.read_bytes()
    except OSError as error:
        msg = f"could not read release archive {archive}: {error}"
        raise ValueError(msg) from error
    actual_hash = hashlib.sha256(archive_bytes).hexdigest()
    if actual_hash != expected_hash:
        msg = f"archive SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        raise ValueError(msg)

    raw_rows = _load_rows(archive)
    if len(raw_rows) != expected_rows:
        msg = f"archive row count mismatch: expected {expected_rows}, got {len(raw_rows)}"
        raise ValueError(msg)
    seeds = _manifest_seeds(manifest)
    repository = _require_string(manifest.get("repository"), "manifest repository")
    revision = _require_string(manifest.get("revision"), "manifest revision")
    normalized, roles, draws = _validate_rows(
        raw_rows,
        seeds=seeds,
        repository=repository,
        revision=revision,
        fields=selected_fields,
    )

    grouped: dict[tuple[int, str, str], list[_Reading]] = defaultdict(list)
    for reading in normalized:
        grouped[(reading.seed, reading.role, reading.condition)].append(reading)
    outcomes: dict[str, tuple[RoleOutcome, ...]] = {}
    for field in selected_fields:
        role_outcomes: list[RoleOutcome] = []
        for role in roles:
            differences = []
            for seed in seeds:
                neutral = grouped[(seed, role, NEUTRAL_CONDITION)]
                affective = grouped[(seed, role, AFFECTIVE_CONDITION)]
                differences.append(
                    statistics.mean(row.values[field] for row in affective)
                    - statistics.mean(row.values[field] for row in neutral)
                )
            role_outcomes.append(
                RoleOutcome(
                    role=role,
                    pairs=len(differences),
                    estimate=student_t_estimate(differences),
                )
            )
        outcomes[field] = _holm(role_outcomes)

    counts: dict[str, Counter[str]] = {
        condition: Counter() for condition in sorted({r.condition for r in normalized})
    }
    for reading in normalized:
        counts[reading.condition][reading.action] += 1
    return ReaderOutcomeReport(
        schema_version=1,
        name=name,
        manifest_file=str(manifest_path),
        archive_file=str(archive),
        archive_sha256=actual_hash,
        rows=len(normalized),
        seeds=seeds,
        roles=roles,
        conditions=tuple(sorted(counts)),
        draws=draws,
        repository=repository,
        revision=revision,
        fields=selected_fields,
        effect_direction=f"{AFFECTIVE_CONDITION}_minus_{NEUTRAL_CONDITION}",
        raw_size_includes_hold=True,
        size_measurement=(
            "raw model-emitted size over all replies, including HOLD; isolated readers do not "
            "execute or fill trades"
        ),
        nonzero_hold_sizes=sum(
            reading.action == "HOLD" and reading.size != 0.0 for reading in normalized
        ),
        action_counts={
            condition: dict(sorted(value.items())) for condition, value in counts.items()
        },
        outcomes=outcomes,
        buy_mechanism=_buy_mechanism(outcomes) if include_buy_mechanism else None,
    )
