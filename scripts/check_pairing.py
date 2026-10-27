"""Verify that matched arms stayed byte-identical until the treatment landed.

Run this before any paired analysis. A pair that fails here still produces numbers, but its
intervals understate precision, because the paired difference then carries generation nondeterminism
as well as the treatment. Exits non-zero when any pair fails, so it can gate a confirmatory run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from affective_contagion.pairing import check_pair


def main() -> None:
    """Report pairing health for every matched pair and exit non-zero if any is broken."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral", type=Path, nargs="+", required=True)
    parser.add_argument("--affective", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    if len(arguments.neutral) != len(arguments.affective):
        print("the two arms must have the same number of runs", file=sys.stderr)
        raise SystemExit(2)
    reports = [
        check_pair(neutral, affective)
        for neutral, affective in zip(arguments.neutral, arguments.affective, strict=True)
    ]
    for report in reports:
        print(report.summary)
    matched = sum(report.matched for report in reports)
    total = len(reports)
    cells = sum(report.pre_treatment_cells for report in reports)
    identical = sum(report.pre_treatment_identical for report in reports)
    print(
        f"\n{matched}/{total} pairs matched; "
        f"{identical}/{cells} pre-treatment generations byte-identical "
        f"({identical / cells:.1%})"
    )
    if matched != total:
        print(
            "\nFAIL: at least one pair diverged before its treatment. Paired SDs from these runs "
            "are inflated by generation nondeterminism, so any interval from them is wider than "
            "the design warrants and any 'seeds required' figure from it is too large.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("\nOK: every pair is a valid matched pair.")


if __name__ == "__main__":
    main()
