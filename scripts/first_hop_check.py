"""Report whether the treatment survived into the treated agent's own message.

Thin CLI over `affective_contagion.first_hop`. Run this before believing any propagation estimate:
if the arms do not separate here, peers never received a manipulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affective_contagion.first_hop import first_hop_separation


def parse_arguments() -> argparse.Namespace:
    """Parse the two matched arms and an optional report destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral", type=Path, nargs="+", required=True)
    parser.add_argument("--affective", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="Write the report as JSON as well.")
    parser.add_argument(
        "--planned-seeds",
        type=int,
        default=12,
        help="Seed count of the confirmatory matrix, for the detectability line.",
    )
    return parser.parse_args()


def main() -> None:
    """Print the paired first-hop separation and what it implies for a confirmatory run."""
    arguments = parse_arguments()
    result = first_hop_separation(list(arguments.neutral), list(arguments.affective))
    print(f"treated role {result.treated_role!r}, shock round {result.shock_round}")
    print(f"{'seed':>6}  {'neutral':>9}  {'anxiety':>9}  {'difference':>11}")
    for pair in result.pairs:
        print(
            f"{pair.seed:>6}  {pair.neutral:9.4f}  {pair.anxiety_like:9.4f}  "
            f"{pair.difference:+11.4f}"
        )
    n = len(result.pairs)
    low, high = result.mean_difference_ci
    print(f"\npaired mean difference : {result.mean_difference:+.4f}  (n={n})")
    print(f"95% t interval         : [{low:+.4f}, {high:+.4f}]")
    print(f"sd of differences      : {result.difference_sd:.4f}")
    print(f"cohen's dz             : {result.cohens_dz:+.3f}")
    print(f"seeds with a rise      : {result.positive_seeds}/{n}")
    planned = arguments.planned_seeds
    print(f"\ndetectable at {planned} seeds  : {result.minimum_detectable_difference(planned):.4f}")
    print("  (large-sample z approximation; the exact paired t calculation needs more seeds)")
    report = {
        "schema_version": 2,
        "treated_role": result.treated_role,
        "shock_round": result.shock_round,
        "pairs": [
            {"seed": p.seed, "neutral": p.neutral, "anxiety_like": p.anxiety_like}
            for p in result.pairs
        ],
        "mean_difference": result.mean_difference,
        "mean_difference_ci_95": [low, high],
        "difference_sd": result.difference_sd,
        "cohens_dz": result.cohens_dz,
        "positive_seeds": result.positive_seeds,
        "minimum_detectable_difference_at_planned_seeds": (
            result.minimum_detectable_difference(planned)
        ),
        "power_method": (
            "two-sided z approximation, alpha 0.05, power 0.80, pilot SD as known sigma"
        ),
        "planned_seeds": planned,
    }
    if arguments.output:
        with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
            # allow_nan=False: a single-pair run has undefined spread, and emitting bare `NaN`
            # produces a file that is not valid JSON for any strict reader.
            stream.write(json.dumps(report, indent=2, allow_nan=False))
        print(f"\nsaved {arguments.output}")


if __name__ == "__main__":
    main()
