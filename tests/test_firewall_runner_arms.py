"""The firewall runner's optional fourth arm must be the true no-closing rendering.

The within-run mechanism test of PR24 compares the filtered text against the ablation arm of E45,
which is the frame plus the verbatim fact block and nothing else. The added arm must be
byte-identical to that rendering, or the comparison uses the wrong baseline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from affective_contagion.configuration import load_yaml_model
from affective_contagion.payload import FinancialPayload

_SPEC = importlib.util.spec_from_file_location(
    "run_firewall_readers", Path(__file__).parent.parent / "scripts" / "run_firewall_readers.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_firewall_readers"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_the_runner_exposes_the_no_closing_flag() -> None:
    """The flag must exist and default off, so existing invocations are unchanged."""
    parser = _MODULE._parser()  # noqa: SLF001 - the parser is the behaviour under test
    arguments = parser.parse_args(
        [
            "--pairs-root",
            "x",
            "--seeds",
            "1",
            "--scenario",
            "s",
            "--roster",
            "r",
            "--vad-lexicon",
            "l",
            "--vad-lexicon-scale",
            "signed",
            "--output",
            "o",
            "--rewrite-log",
            "w",
        ]
    )
    assert arguments.include_no_closing is False
    with_flag = parser.parse_args(
        [
            "--pairs-root",
            "x",
            "--seeds",
            "1",
            "--scenario",
            "s",
            "--roster",
            "r",
            "--vad-lexicon",
            "l",
            "--vad-lexicon-scale",
            "signed",
            "--output",
            "o",
            "--rewrite-log",
            "w",
            "--include-no-closing",
        ]
    )
    assert with_flag.include_no_closing is True


def test_no_closing_arm_matches_the_ablation_rendering() -> None:
    """The fourth arm is payload.realize('no_closing'): a strict prefix of both closing arms."""
    payload = load_yaml_model(
        Path(__file__).parent.parent / "configs" / "scenarios" / "meridian_deposit_outflow.yaml",
        FinancialPayload,
    )
    bare = payload.realize("no_closing")
    assert payload.fact_block() in bare
    assert payload.realize("neutral").startswith(bare)
    assert payload.realize("anxiety_like").startswith(bare)
