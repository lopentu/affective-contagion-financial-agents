"""Cross affect with urgency and show readers all four cells (PR19).

The arousal lexicon scores *anxious* and *urgent* wording almost identically. This design varies the
two separately, so the manipulation, not the instrument, tells them apart.

Urgency is a **payload slot** here rather than part of the treatment.
`FinancialPayload.fact_block` puts review timing inside the block both arms embed verbatim, so the
two factors are not collinear. Two payloads differing only in that slot, each rendered in both
affective registers, give the 2x2 design.

**The two factors are dose-matched on stimulus arousal, the measure that cannot separate them.** The
urgency levels below move it by +0.0309 against affect's +0.0353, within 13% of each other, while
the emotion classifier separates them completely (+0.0014 against +0.8966). PR19 Amendment A
records these levels.

Each level is a statement about *when to review*. Neither describes the speaker's state and neither
asserts an outcome, so the existing residue gate applies unchanged and both levels pass it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from affective_contagion.backends.huggingface import HuggingFaceCausalLMBackend
from affective_contagion.configuration import load_yaml_model
from affective_contagion.generation import GenerationBackend, RoutedDecisionGenerator
from affective_contagion.isolated_reader import StimulusInbox, read_in_isolation
from affective_contagion.pairing import check_pair
from affective_contagion.payload import FinancialPayload, matched_realizations
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.roster import ModelRoster
from affective_contagion.text_affect import VADLexicon

# Base seed for this design's reader draws; each reader design has its own base.
FACTORIAL_SEED_BASE = 500_000

URGENCY_LEVELS = {
    "routine": "review at the next scheduled committee meeting, at the usual time",
    "immediate": "urgent immediate review demanded now, before the deadline expires this session",
}


def build_cells(payload: FinancialPayload) -> dict[tuple[str, str], str]:
    """Return the four renderings, keyed by (affect, urgency).

    `matched_realizations` raises when a pair states a number outside the fact block, restates a
    payload value in the affect residue, or uses an outcome-asserting term. Calling it once per
    urgency level applies that gate to both levels.
    """
    cells: dict[tuple[str, str], str] = {}
    for urgency, wording in URGENCY_LEVELS.items():
        arms = matched_realizations(payload.model_copy(update={"urgency": wording}))
        for affect, text in arms.items():
            cells[(affect, urgency)] = text
    return cells


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--pair-prefix", default="s30")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--vad-lexicon", type=Path, required=True)
    parser.add_argument("--vad-lexicon-scale", choices=("unit", "signed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--treated-role", default="sentiment_analyst")
    parser.add_argument(
        "--reader-roles",
        nargs="+",
        default=[
            "fundamental_analyst",
            "bull_researcher",
            "bear_researcher",
            "trader",
            "risk_manager",
        ],
    )
    return parser


def main() -> None:
    """Generate every reading the square needs, one row per generation."""
    arguments = _parser().parse_args()
    if arguments.output.exists():
        msg = f"{arguments.output} exists; refusing to overwrite recorded readings"
        raise FileExistsError(msg)

    roster = load_yaml_model(arguments.roster, ModelRoster)
    payload = load_yaml_model(arguments.scenario, FinancialPayload)
    cells = build_cells(payload)
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon, scale=arguments.vad_lexicon_scale, match="surface_then_suffix"
    )
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    role_index = {agent.role: index for index, agent in enumerate(roster.agents)}
    treated_index = role_index[arguments.treated_role]

    # Print the stimulus doses before any generation. The design requires the two factors to move
    # arousal by comparable amounts.
    score = {key: lexicon.score(text).arousal for key, text in cells.items()}
    affect = (
        score[("anxiety_like", "routine")]
        + score[("anxiety_like", "immediate")]
        - score[("neutral", "routine")]
        - score[("neutral", "immediate")]
    ) / 2
    urgency = (
        score[("neutral", "immediate")]
        + score[("anxiety_like", "immediate")]
        - score[("neutral", "routine")]
        - score[("anxiety_like", "routine")]
    ) / 2
    print(
        f"stimulus dose  affect {affect:+.4f}  urgency {urgency:+.4f}  "
        f"ratio {urgency / affect:.2f}",
        flush=True,
    )

    # Run the pairing gate over every seed before any generation.
    for seed in arguments.seeds:
        report = check_pair(
            arguments.pairs_root / f"{arguments.pair_prefix}-neutral-s{seed}",
            arguments.pairs_root / f"{arguments.pair_prefix}-affective-s{seed}",
        )
        if not report.matched:
            msg = f"seed {seed} did not pass the pairing gate: {report.summary}"
            raise ValueError(msg)

    written = 0
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in arguments.seeds:
            for (affect_level, urgency_level), text in cells.items():
                # Bare context only: the treated message alone, price pinned to the fundamental
                # value. Context is varied in E36's design, not here.
                stimulus = StimulusInbox(
                    seed=seed,
                    arm=f"{affect_level}|{urgency_level}",
                    treated_role=arguments.treated_role,
                    treated_text=text,
                    peer_messages=(),
                    treated_index=treated_index,
                )
                for reader_role in arguments.reader_roles:
                    generator = RoutedDecisionGenerator(
                        roster=roster,
                        backends=backends,
                        prompt_builder=FinancialPromptBuilder(),
                        run_seed=FACTORIAL_SEED_BASE + seed,
                        affect_directions={},
                    )
                    for reading in read_in_isolation(
                        generator,
                        stimulus,
                        reader_role=reader_role,
                        reader_index=role_index[reader_role],
                        price=100.0,
                        fundamental_value=100.0,
                        draws=arguments.draws,
                        vad_lexicon=lexicon,
                        round_index=1,
                    ):
                        row: dict[str, Any] = asdict(reading)
                        row["affect"] = asdict(reading.affect)
                        row["affect_level"] = affect_level
                        row["urgency_level"] = urgency_level
                        stream.write(json.dumps(row) + "\n")
                        written += 1
            print(f"seed {seed}: {written} readings so far", flush=True)
    print(f"wrote {written} readings to {arguments.output}")


if __name__ == "__main__":
    main()
