"""Independently verify reported reader fields, never treating HOLD size as a filled trade.

This route uses standard-library cell means and the Student distribution's special functions.
It does not import the analysis module or its loader, aggregation, interval or Holm code.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.special import stdtr, stdtrit

from affective_contagion.verification import verify_release


def _rows(manifest_path: Path) -> list[dict]:
    intact, detail = verify_release(manifest_path)
    if not intact:
        raise ValueError(detail)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent / manifest["readings"]["file"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != manifest["readings"]["rows"]:
        msg = "Reader archive cardinality differs from its manifest"
        raise ValueError(msg)
    if {row["seed"] for row in rows} != set(manifest["seeds"]):
        msg = "Reader seeds differ from their manifest"
        raise ValueError(msg)
    for row in rows:
        if any(row[key] != manifest[key] for key in ("repository", "revision")):
            msg = "Reader model identity differs from its manifest"
            raise ValueError(msg)
        if row["arm"] != row["condition"]:
            msg = "This estimator requires one context per treatment condition"
            raise ValueError(msg)
    if {row["arm"] for row in rows} != {"neutral", "anxiety_like"}:
        msg = "Expected exactly the calm and high-arousal conditions"
        raise ValueError(msg)
    return rows


def _differences(rows: list[dict], field: str) -> dict[str, list[float]]:
    cells: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (row["reader_role"], row["seed"], row["arm"])
        if row["draw"] in cells[key] or not math.isfinite(float(row[field])):
            msg = "Duplicate draw or nonfinite reader outcome"
            raise ValueError(msg)
        cells[key][row["draw"]] = float(row[field])
    roles = {row["reader_role"] for row in rows}
    seeds = {row["seed"] for row in rows}
    draws = {row["draw"] for row in rows}
    out: dict[str, list[float]] = {}
    for role in sorted(roles):
        values = []
        for seed in sorted(seeds):
            calm, high = cells[role, seed, "neutral"], cells[role, seed, "anxiety_like"]
            if set(calm) != draws or set(high) != draws:
                msg = "Missing role, seed, condition or draw in paired reader data"
                raise ValueError(msg)
            values.append(statistics.fmean(high.values()) - statistics.fmean(calm.values()))
        out[role] = values
    return out


def student_summary(values: list[float]) -> dict:
    """Compute the two-sided paired t result without a lookup table or normal approximation."""
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        msg = "A t interval requires at least two finite paired differences"
        raise ValueError(msg)
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / math.sqrt(len(values))
    degrees = len(values) - 1
    half = float(stdtrit(degrees, 0.975)) * error
    p_value = float(2 * stdtr(degrees, -abs(mean / error))) if error else float(mean == 0)
    return {"value": mean, "ci": [mean - half, mean + half], "p": p_value}


def _field_results(rows: list[dict], field: str) -> dict[str, dict]:
    results = {role: student_summary(vals) for role, vals in _differences(rows, field).items()}
    ordered = sorted(results, key=lambda role: results[role]["p"])
    # Holm-adjusted p-values via the cumulative maximum, rather than the analysis's step-down test.
    maximum = 0.0
    for rank, role in enumerate(ordered):
        maximum = max(maximum, (len(ordered) - rank) * results[role]["p"])
        results[role]["holm_p"] = min(1.0, maximum)
    return results


def reader_numbers(manifest_path: Path) -> dict:
    """Return independently rebuilt effects, action counts and HOLD-size diagnostics."""
    rows = _rows(manifest_path)
    result: dict = {
        field: _field_results(rows, field) for field in ("size", "downside_risk", "confidence")
    }
    result["counts"] = {
        f"{arm}_{action}": sum(row["arm"] == arm and row["action"] == action for row in rows)
        for arm in ("neutral", "anxiety_like")
        for action in ("BUY", "SELL", "HOLD")
    }
    result["counts"]["nonzero_hold_size"] = sum(
        row["action"] == "HOLD" and row["size"] != 0 for row in rows
    )
    result["counts"]["rows"] = len(rows)
    result["counts"]["buy_less_weight_pattern"] = sum(
        result["size"][role]["value"] < 0
        and result["size"][role]["holm_p"] <= 0.05
        and result["downside_risk"][role]["ci"][1] < 0
        and result["confidence"][role]["ci"][0] > 0
        for role in result["size"]
    )
    return result
