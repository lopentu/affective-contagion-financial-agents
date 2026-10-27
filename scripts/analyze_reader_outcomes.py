"""Verify and analyse released E46/E47/E48 isolated-reader outcome archives.

Usage:
    uv run python scripts/analyze_reader_outcomes.py data/generations/e46-action-sell.manifest.json
    uv run python scripts/analyze_reader_outcomes.py \
        data/generations/e46-action-sell.manifest.json \
        data/generations/e47-action-buy.manifest.json \
        data/generations/e48-action-*.manifest.json \
        --output outputs/reader-outcomes.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from affective_contagion.reader_outcomes import DEFAULT_FIELDS, analyze_reader_outcomes

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPOSITORY_ROOT / "outputs"


def _write_new_output(path: Path, text: str) -> Path:
    """Write below the repository outputs directory without replacing an artefact."""
    destination = path.resolve()
    outputs = OUTPUTS_ROOT.resolve()
    if not destination.is_relative_to(outputs) or destination == outputs:
        msg = f"--output must name a new file below {outputs}"
        raise ValueError(msg)
    if destination.exists():
        msg = f"refusing to overwrite existing output: {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Analyse each supplied release manifest and emit one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument(
        "--buy-mechanism",
        action="store_true",
        help="Apply PR22's BUY-only explanation criterion; requires exactly one manifest.",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.buy_mechanism and len(arguments.manifests) != 1:
        parser.error("--buy-mechanism requires exactly one BUY-event manifest")
    analyses = []
    for path in arguments.manifests:
        report = analyze_reader_outcomes(
            path,
            fields=arguments.fields,
            include_buy_mechanism=arguments.buy_mechanism,
        )
        serialized = asdict(report)
        if report.buy_mechanism is None:
            del serialized["buy_mechanism"]
        analyses.append(serialized)
    payload = {
        "schema_version": 1,
        "analyses": analyses,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        destination = _write_new_output(arguments.output, rendered)
        print(f"wrote {len(payload['analyses'])} analysis report(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
