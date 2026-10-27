"""E51 outcomes 4a, 4b and 5: the peer response three ways on the rerun of the injected design.

4a is the registered E27 route (`peer_response`, sentiment analyst assumed to be the sender), kept
for comparison with the published figures. 4b is `peer_response_by_manifest`, which excludes the
role each pair's manifest names as treated. 5 is the treated agent's own round-9 shift. All three
are descriptive replication outcomes under PR26 Amendment A and are never pooled with E27.

Usage:
    uv run python scripts/e51_replication.py --runs outputs/runs/e51
        --vad-lexicon data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt
        --output outputs/e51_replication.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from affective_contagion.text_affect import VADLexicon
from affective_contagion.verification import (
    peer_response,
    peer_response_by_manifest,
    treated_self_response,
)


def main() -> int:
    """Compute the three outcomes at round 9 and write them as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--vad-lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=9)
    # Added after the frozen run (PR26 Amendment D): the frozen default is exact surface matching;
    # the registered E27 route matched word endings too. Both are reported.
    parser.add_argument(
        "--vad-match", default="surface", choices=("surface", "surface_then_suffix")
    )
    arguments = parser.parse_args()
    lexicon = VADLexicon.from_tsv(arguments.vad_lexicon, scale="signed", match=arguments.vad_match)
    results = {
        "4a_fixed_role": peer_response(arguments.runs, lexicon, round_index=arguments.round),
        "4a_fixed_role_echo_dropped": peer_response(
            arguments.runs, lexicon, round_index=arguments.round, drop_echo=True
        ),
        "4b_by_manifest": peer_response_by_manifest(
            arguments.runs, lexicon, round_index=arguments.round
        ),
        "4b_by_manifest_echo_dropped": peer_response_by_manifest(
            arguments.runs, lexicon, round_index=arguments.round, drop_echo=True
        ),
        "5_treated_self": treated_self_response(
            arguments.runs, lexicon, round_index=arguments.round
        ),
    }
    payload = {name: asdict(value) for name, value in results.items()}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, value in results.items():
        low, high = value.interval if value.interval else (float("nan"), float("nan"))
        print(f"{name:32s} {value.value:+.4f} [{low:+.4f}, {high:+.4f}]  {value.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
