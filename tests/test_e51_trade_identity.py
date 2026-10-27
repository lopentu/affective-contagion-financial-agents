"""Known-answer test for the E51 trade-identity comparison."""

from __future__ import annotations

import json
from pathlib import Path

from affective_contagion.trade_identity import compare_pair, summarise

ROLES = ("sentiment_analyst", "fundamental_analyst", "trader")


def _write_run(
    root: Path,
    name: str,
    *,
    treated_action: tuple[str, float],
    treated_position: float,
    price_after_shock: float,
    treated_text: str,
) -> Path:
    run = root / name
    run.mkdir()
    shock = 2
    manifest = {
        "experiment": {"treated_role": "sentiment_analyst", "shock_round": shock, "seed": 7}
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = []
    for round_index in range(4):
        for index, role in enumerate(ROLES):
            treated = role == "sentiment_analyst" and round_index == shock
            action, size = treated_action if treated else ("HOLD", 0.0)
            text = treated_text if treated else f"r{round_index} {role}"
            records.append(
                {
                    "round_index": round_index,
                    "agent_index": index,
                    "role": role,
                    "decision": {"message": text, "action": action, "size": size},
                    "generation": {"raw_output": text},
                    "position_after": treated_position if treated else 0.0,
                    "parse_succeeded": True,
                }
            )
    (run / "records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    market = [
        {"round_index": r, "price": price_after_shock if r == shock else 100.0} for r in range(4)
    ]
    (run / "market.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in market) + "\n", encoding="utf-8"
    )
    return run


def test_identical_trade_and_price_are_counted(tmp_path: Path) -> None:
    """Same requested trade, same position, same price: every identity is true."""
    neutral = _write_run(
        tmp_path,
        "n",
        treated_action=("SELL", 0.5),
        treated_position=-0.5,
        price_after_shock=99.5,
        treated_text="calm",
    )
    affective = _write_run(
        tmp_path,
        "a",
        treated_action=("SELL", 0.5),
        treated_position=-0.5,
        price_after_shock=99.5,
        treated_text="alarmed",
    )
    row = compare_pair(neutral, affective, "meridian_deposit_outflow")
    assert row.same_decision
    assert row.same_filled_position
    assert row.same_next_price
    assert row.gate_matched  # pre-treatment identical, peers identical, treated text differs
    assert summarise([row])["pooled"]["same_next_price_gate_passing"] == 1


def test_a_different_size_is_a_different_trade(tmp_path: Path) -> None:
    """A smaller position under the alarmed arm is not identical, and the price moves with it."""
    neutral = _write_run(
        tmp_path,
        "n",
        treated_action=("SELL", 0.5),
        treated_position=-0.5,
        price_after_shock=99.5,
        treated_text="calm",
    )
    affective = _write_run(
        tmp_path,
        "a",
        treated_action=("SELL", 0.2),
        treated_position=-0.2,
        price_after_shock=99.8,
        treated_text="alarmed",
    )
    row = compare_pair(neutral, affective, "meridian_deposit_outflow")
    assert not row.same_decision
    assert not row.same_filled_position
    assert not row.same_next_price
    pooled = summarise([row])["pooled"]
    assert pooled["same_filled_position"] == 0
    assert pooled["gate_passing"] == 1


def test_same_position_but_different_price_is_reported_separately(tmp_path: Path) -> None:
    """Position identity and price identity are two counts, not one."""
    neutral = _write_run(
        tmp_path,
        "n",
        treated_action=("SELL", 0.5),
        treated_position=-0.5,
        price_after_shock=99.5,
        treated_text="calm",
    )
    affective = _write_run(
        tmp_path,
        "a",
        treated_action=("SELL", 0.5),
        treated_position=-0.5,
        price_after_shock=99.4,
        treated_text="alarmed",
    )
    row = compare_pair(neutral, affective, "meridian_deposit_outflow")
    assert row.same_filled_position
    assert not row.same_next_price
