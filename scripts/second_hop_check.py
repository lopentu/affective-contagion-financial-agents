"""Report PR1's second-hop outcome with every gate the plan requires.

Thin CLI over `affective_contagion.second_hop`. Prints the plan's verdict first, then the numbers
behind it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from affective_contagion.paired import PairedComparison
from affective_contagion.second_hop import (
    INFORMATIVE_NULL_CEILING,
    SecondHopReport,
    second_hop_separation,
)


def _line(label: str, comparison: PairedComparison) -> str:
    """Format one paired comparison with its interval."""
    low, high = comparison.mean_difference_ci
    return (
        f"{label:<22} {comparison.mean_difference:+.4f}  "
        f"CI [{low:+.4f}, {high:+.4f}]  sd {comparison.difference_sd:.4f}  "
        f"dz {comparison.cohens_dz:+.3f}  signs {comparison.positive_seeds}/{comparison.n}"
    )


def _report_dict(report: SecondHopReport) -> dict[str, object]:
    """Serialise the report, keeping every interval attached to its point estimate."""

    def block(comparison: PairedComparison) -> dict[str, object]:
        low, high = comparison.mean_difference_ci
        return {
            "mean_difference": comparison.mean_difference,
            "ci_95": [low, high],
            "difference_sd": comparison.difference_sd,
            "cohens_dz": comparison.cohens_dz,
            "positive_seeds": comparison.positive_seeds,
            "n": comparison.n,
        }

    return {
        "schema_version": 1,
        "plan": "docs/plans/pr1-second-hop.md",
        "verdict": report.verdict,
        "treated_role": report.treated_role,
        "shock_round": report.shock_round,
        "peer_round": report.peer_round,
        "peers": block(report.peers),
        "first_hop": block(report.first_hop),
        "order_imbalance": block(report.order_imbalance),
        "price": block(report.price),
        "treated_placebo": block(report.treated_placebo),
        "clean_subset": None if report.clean_subset is None else block(report.clean_subset),
        "clean_seeds": list(report.clean_seeds),
        "seeds_with_price_difference": report.seeds_with_price_difference,
        "excluded_seeds": list(report.excluded_seeds),
        "minimum_coverage": report.minimum_coverage,
        "first_hop_replicated": report.first_hop_replicated,
        "price_channel_active": report.price_channel_active,
        "informative_if_null": report.informative_if_null,
        "supported": report.supported,
        "informative_null_ceiling": INFORMATIVE_NULL_CEILING,
    }


def main() -> None:
    """Print PR1's verdict and the numbers behind it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral", type=Path, nargs="+", required=True)
    parser.add_argument("--affective", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="Write the report as JSON as well.")
    parser.add_argument(
        "--peer-round-offset",
        type=int,
        default=1,
        help="1 is the single-hop round and PR1's primary. Any other value is exploratory.",
    )
    arguments = parser.parse_args()
    report = second_hop_separation(
        list(arguments.neutral),
        list(arguments.affective),
        peer_round_offset=arguments.peer_round_offset,
    )
    if arguments.peer_round_offset != 1:
        print("EXPLORATORY: not PR1's primary window.\n")
    print(report.verdict)
    print()
    print(f"treated role {report.treated_role!r}, shock round {report.shock_round}, ")
    print(f"peer round {report.peer_round}\n")
    print(_line("peers (PRIMARY)", report.peers))
    print(_line("first hop (gate)", report.first_hop))
    print(_line("placebo: treated", report.treated_placebo))
    print(_line("shock imbalance", report.order_imbalance))
    print(_line("shock price", report.price))
    if report.clean_subset is not None:
        print(_line("peers, clean subset", report.clean_subset))
    print()
    print(f"seeds whose shock-round trade matched : {list(report.clean_seeds)}")
    print(f"seeds with a price difference         : {report.seeds_with_price_difference}")
    print(f"seeds excluded (parse)                : {list(report.excluded_seeds)}")
    print(f"minimum lexicon coverage              : {report.minimum_coverage:.3f}")
    floor = report.peers.minimum_detectable_difference(report.peers.n)
    print(
        f"detectable at n={report.peers.n}: {floor:.4f} "
        f"(informative-null ceiling {INFORMATIVE_NULL_CEILING:.4f}) -> "
        f"{'informative' if report.informative_if_null else 'UNDERPOWERED'}"
    )
    if arguments.output:
        with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_report_dict(report), indent=2, allow_nan=False))
        print(f"\nsaved {arguments.output}")


if __name__ == "__main__":
    main()
