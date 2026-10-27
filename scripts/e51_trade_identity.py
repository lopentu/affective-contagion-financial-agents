"""E51 command line: compare every matched pair under --runs and write the counts.

The comparison itself lives in `affective_contagion.trade_identity`; the plan is PR26.

Usage:
    uv run python scripts/e51_trade_identity.py --runs outputs/runs/e51 \
        --output outputs/e51_trade_identity.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from affective_contagion.trade_identity import SCENARIOS, PairIdentity, compare_pair, summarise


def main() -> int:
    """Compare every pair under --runs and write the counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 336)))
    arguments = parser.parse_args()
    pairs: list[PairIdentity] = []
    missing: list[str] = []
    for scenario in SCENARIOS:
        for seed in arguments.seeds:
            neutral = arguments.runs / f"e51-{scenario}-neutral-s{seed}"
            affective = arguments.runs / f"e51-{scenario}-affective-s{seed}"
            if (
                not (neutral / "records.jsonl").exists()
                or not (affective / "records.jsonl").exists()
            ):
                missing.append(f"{scenario} s{seed}")
                continue
            pairs.append(compare_pair(neutral, affective, scenario))
    summary = summarise(pairs)
    summary["missing_pairs"] = missing
    summary["rows"] = [asdict(row) for row in pairs]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pooled = summary["pooled"]
    print(
        f"{pooled['pairs']} pairs ({len(missing)} missing); gate passing {pooled['gate_passing']}; "
        f"same decision {pooled['same_decision']}; same filled position "
        f"{pooled['same_filled_position']}; same next price {pooled['same_next_price']}"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
