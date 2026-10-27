"""Verify E51's exact planned inputs and complete source files before running its frozen analyses.

This checks inventory and provenance only. It computes no treatment outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affective_contagion.config import ExperimentConfig
from affective_contagion.configuration import load_yaml_model
from affective_contagion.payload import FinancialPayload
from affective_contagion.roster import ModelRoster
from affective_contagion.run_release import validate_source_inventory


def main() -> int:
    """Compare every source manifest with the PR26 scenario and PR5 configuration catalogue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roster = load_yaml_model(Path("configs/models/qwen3_4b_shared.yaml"), ModelRoster)
    expected = []
    # The four scenarios and the seed range named in PR26.
    for scenario in (
        "meridian_deposit_outflow",
        "harborline_capacity_award",
        "pallasridge_covenant_note",
        "calderon_mixed_guidance",
    ):
        payload = load_yaml_model(Path(f"configs/scenarios/{scenario}.yaml"), FinancialPayload)
        for seed in range(300, 336):
            for arm in ("neutral", "affective"):
                experiment = load_yaml_model(
                    Path(f"configs/experiments/pr5/{arm}_s{seed}.yaml"), ExperimentConfig
                )
                expected.append(
                    {
                        "payload": payload.model_dump(mode="json"),
                        "experiment": experiment.model_dump(mode="json"),
                        "roster": roster.model_dump(mode="json"),
                    }
                )
    result = validate_source_inventory(args.runs, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"E51 inventory verified: {result['runs']} complete runs match the planned inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
