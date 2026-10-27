"""Show readers three renderings of one payload: calm, high-arousal, and no closing at all.

PR20 tests whether the response left after E41's payload-preserving rewrite comes from the *calm
closing's absence* rather than the alarm's presence. The third arm is not a rewrite. It has the
same frame and verbatim fact block with no closing sentence, so the three arms differ only in which
self-referential sentence, if any, ends the message.

Two contrasts, held apart because each answers a different half of the decomposition:

* **What does removing the calm do?** ``no_closing`` against ``neutral``. Positive means the calm
  closing was pulling the reader down, and its absence alone raises the reply.
* **What does adding the alarm do?** ``anxiety_like`` against ``no_closing``. Positive means the
  alarm adds something beyond the missing reassurance.

Their sum is the reference effect (``anxiety_like`` against ``neutral``), which must reproduce
E36/E41 before either component is read.
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

# Base seed for this design's reader draws. The other bases are 500k factorial, 600k rewrite,
# 700k firewall, 800k context comparison (plus 100k per extra replicate) and 900k isolated reader.
CLOSING_SEED_BASE = 1_000_000


def _shock_round(run: Path) -> int:
    """Return the round at which the stimulus was delivered."""
    return int(
        json.loads((run / "manifest.json").read_text(encoding="utf-8"))["experiment"]["shock_round"]
    )


def _peers_and_price(
    run: Path, treated_role: str
) -> tuple[tuple[tuple[int, str, str], ...], float]:
    """Return the untreated peers' shock-round messages and the price one round later."""
    rows = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines() if x.strip()]
    shock = _shock_round(run)
    peers = tuple(
        (int(r["agent_index"]), str(r["role"]), str(r["decision"]["message"]))
        for r in rows
        if r["round_index"] == shock and r["role"] != treated_role and r["decision"]["message"]
    )
    after = [r for r in rows if r["round_index"] == shock + 1]
    price = float(after[0]["context"]["price"]) if after else 100.0
    return peers, price


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
    parser.add_argument(
        "--conditions", nargs="+", default=["bare", "embedded"], choices=["bare", "embedded"]
    )
    return parser


def _rows_for_seed(
    seed: int,
    texts: dict[str, str],
    *,
    arguments: argparse.Namespace,
    roster: ModelRoster,
    backends: dict[str, GenerationBackend],
    lexicon: VADLexicon,
    role_index: dict[str, int],
    treated_index: int,
) -> list[dict[str, Any]]:
    """Every reading one matched pair produces, across contexts, arms and reader roles.

    The three arms share a seed base, which is what makes them common random numbers rather than
    independent draws.
    """
    neutral = arguments.pairs_root / f"{arguments.pair_prefix}-neutral-s{seed}"
    peers, price = _peers_and_price(neutral, arguments.treated_role)
    shock = _shock_round(neutral)
    rows: list[dict[str, Any]] = []
    for condition in arguments.conditions:
        use_peers = peers if condition == "embedded" else ()
        use_price = price if condition == "embedded" else 100.0
        use_round = shock if condition == "embedded" else 1
        for arm, text in texts.items():
            stimulus = StimulusInbox(
                seed=seed,
                arm=arm,
                treated_role=arguments.treated_role,
                treated_text=text,
                peer_messages=use_peers,
                treated_index=treated_index,
            )
            for reader_role in arguments.reader_roles:
                generator = RoutedDecisionGenerator(
                    roster=roster,
                    backends=backends,
                    prompt_builder=FinancialPromptBuilder(),
                    run_seed=CLOSING_SEED_BASE + seed,
                    affect_directions={},
                )
                for reading in read_in_isolation(
                    generator,
                    stimulus,
                    reader_role=reader_role,
                    reader_index=role_index[reader_role],
                    price=use_price,
                    fundamental_value=100.0,
                    draws=arguments.draws,
                    vad_lexicon=lexicon,
                    round_index=use_round,
                ):
                    row: dict[str, Any] = asdict(reading)
                    row["affect"] = asdict(reading.affect)
                    row["condition"] = condition
                    row["n_peers"] = len(use_peers)
                    row["price"] = use_price
                    rows.append(row)
    return rows


def main() -> None:
    """Run every reading the three arms need."""
    arguments = _parser().parse_args()
    if arguments.output.exists():
        msg = f"{arguments.output} exists; refusing to overwrite recorded readings"
        raise FileExistsError(msg)

    roster = load_yaml_model(arguments.roster, ModelRoster)
    payload = load_yaml_model(arguments.scenario, FinancialPayload)
    arms = matched_realizations(payload)
    bare = payload.realize("no_closing")
    # The ablation arm must be a strict prefix of both closing arms: same frame, same verbatim
    # fact block, nothing after it. Otherwise the arm is not an ablation.
    for name, text in arms.items():
        if not text.startswith(bare):
            msg = f"no_closing is not a prefix of the {name} arm; the ablation is broken"
            raise ValueError(msg)
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon, scale=arguments.vad_lexicon_scale, match="surface_then_suffix"
    )
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    role_index = {agent.role: index for index, agent in enumerate(roster.agents)}
    treated_index = role_index[arguments.treated_role]

    # Run the pairing gate over every seed before any reading.
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
            texts = {
                "neutral": arms["neutral"],
                "anxiety_like": arms["anxiety_like"],
                "no_closing": bare,
            }
            for row in _rows_for_seed(
                seed,
                texts,
                arguments=arguments,
                roster=roster,
                backends=backends,
                lexicon=lexicon,
                role_index=role_index,
                treated_index=treated_index,
            ):
                stream.write(json.dumps(row) + "\n")
                written += 1
            print(f"seed {seed}: {written} readings so far", flush=True)
    print(f"wrote {written} readings to {arguments.output}")


if __name__ == "__main__":
    main()
