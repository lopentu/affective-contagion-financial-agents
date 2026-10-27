"""E51: compare the treated agent's trade and the observed price across arms in the injected design.

For every matched pair of run directories this compares, at the shock round, the treated agent's
requested decision (action and size), its filled position, and the market price after the round
clears, which is the price every agent observes at the next round. It also reports the pairing gate.
Counts are the result; there is no threshold. The plan is docs/plans/pr26.

The command-line entry point is `scripts/e51_trade_identity.py`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from affective_contagion.pairing import check_pair

SCENARIOS = (
    "meridian_deposit_outflow",
    "harborline_capacity_award",
    "pallasridge_covenant_note",
    "calderon_mixed_guidance",
)


@dataclass(frozen=True, slots=True)
class PairIdentity:
    """What was identical across the two arms of one pair at the shock round."""

    scenario: str
    seed: int
    treated_role: str
    shock_round: int
    same_decision: bool
    same_filled_position: bool
    same_next_price: bool
    gate_matched: bool
    fallbacks: int


def _records(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _treated_at(records: list[dict], role: str, round_index: int) -> dict:
    for record in records:
        if record["role"] == role and record["round_index"] == round_index:
            return record
    msg = f"no record for {role!r} at round {round_index}"
    raise ValueError(msg)


def _price_after(run: Path, round_index: int) -> float:
    for line in (run / "market.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["round_index"] == round_index:
            return float(entry["price"])
    msg = f"no market entry for round {round_index} in {run}"
    raise ValueError(msg)


def compare_pair(neutral: Path, affective: Path, scenario: str) -> PairIdentity:
    """Compare one pair at the shock round; both runs must exist and be complete."""
    manifest = json.loads((neutral / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    role, shock, seed = experiment["treated_role"], experiment["shock_round"], experiment["seed"]
    left, right = _records(neutral), _records(affective)
    n_now, a_now = _treated_at(left, role, shock), _treated_at(right, role, shock)
    n_before, a_before = _treated_at(left, role, shock - 1), _treated_at(right, role, shock - 1)
    same_decision = (n_now["decision"]["action"], n_now["decision"]["size"]) == (
        a_now["decision"]["action"],
        a_now["decision"]["size"],
    )
    same_filled = n_now["position_after"] == a_now["position_after"] and (
        n_before["position_after"] == a_before["position_after"]
    )
    same_price = _price_after(neutral, shock) == _price_after(affective, shock)
    fallbacks = sum(
        1
        for record in (*left, *right)
        if record["round_index"] <= shock and not record["parse_succeeded"]
    )
    return PairIdentity(
        scenario=scenario,
        seed=int(seed),
        treated_role=role,
        shock_round=int(shock),
        same_decision=same_decision,
        same_filled_position=same_filled,
        same_next_price=same_price,
        gate_matched=check_pair(neutral, affective).matched,
        fallbacks=fallbacks,
    )


def summarise(pairs: list[PairIdentity]) -> dict:
    """Count identities per scenario and pooled, over all pairs and over gate-passing pairs."""
    out: dict = {"pairs": len(pairs), "by_scenario": {}, "pooled": {}}

    def counts(rows: list[PairIdentity]) -> dict:
        gate = [row for row in rows if row.gate_matched]
        return {
            "pairs": len(rows),
            "gate_passing": len(gate),
            "same_decision": sum(row.same_decision for row in rows),
            "same_filled_position": sum(row.same_filled_position for row in rows),
            "same_next_price": sum(row.same_next_price for row in rows),
            "same_filled_position_gate_passing": sum(row.same_filled_position for row in gate),
            "same_next_price_gate_passing": sum(row.same_next_price for row in gate),
            "parser_fallbacks_through_shock": sum(row.fallbacks for row in rows),
        }

    grouped: dict[str, list[PairIdentity]] = defaultdict(list)
    for row in pairs:
        grouped[row.scenario].append(row)
    out["by_scenario"] = {name: counts(rows) for name, rows in sorted(grouped.items())}
    out["pooled"] = counts(pairs)
    return out
