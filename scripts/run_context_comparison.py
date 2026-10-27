"""Run PR12's role-by-context reader design, also used by PR15 and PR25.

Each reader sees one treated message and the design varies only what surrounds it.

`bare` gives the reader the treated message alone, with price pinned to the fundamental value.
`embedded` also gives the reader the four other peers' real shock-round messages, at the pair's own
post-shock price. Those four are byte-identical across arms in a verified pair, so within a pair the
two arms still differ in exactly one message.
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

# Base generation seed for this design. Replicate r reads with seed
# CONTEXT_SEED_BASE + 100,000 * (r - 1) + the pair's seed.
CONTEXT_SEED_BASE = 800_000


def _shock_round(run: Path) -> int:
    return int(
        json.loads((run / "manifest.json").read_text(encoding="utf-8"))["experiment"]["shock_round"]
    )


def _peers_and_price(
    run: Path, treated_role: str
) -> tuple[tuple[tuple[int, str, str], ...], float]:
    """Return the untreated peers' shock-round messages and the price one round later."""
    shock = _shock_round(run)
    rows = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines() if x.strip()]
    peers = tuple(
        (int(r["agent_index"]), str(r["role"]), str(r["decision"]["message"]))
        for r in rows
        if r["round_index"] == shock and r["role"] != treated_role and r["decision"]["message"]
    )
    after = [r for r in rows if r["round_index"] == shock + 1]
    price = float(after[0]["context"]["price"]) if after else 100.0
    return peers, price


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser. Split out of main() to keep the run loop readable."""
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
        default=["fundamental_analyst"],
        help="Every role that reads. PR12 crosses all five non-sender roles.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["bare", "embedded"],
        choices=["bare", "embedded"],
        help="Which contexts to run: the report alone at the fundamental price (bare), or with "
        "four teammates' messages at the pair's own price (embedded).",
    )
    parser.add_argument(
        "--rung",
        default="",
        help="Label stamped into every row, so runs with different rosters can be concatenated "
        "and told apart.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Independent repeats of the whole design, differing only in the generation seed base. "
        "PR12 runs two, to check that the role differences are stable.",
    )
    return parser


def main() -> None:
    """Generate every reading the design specifies and write them as one JSONL file."""
    arguments = _parser().parse_args()

    if arguments.output.exists():
        msg = f"{arguments.output} exists; refusing to overwrite recorded readings"
        raise FileExistsError(msg)

    roster = load_yaml_model(arguments.roster, ModelRoster)
    arms = matched_realizations(load_yaml_model(arguments.scenario, FinancialPayload))
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon, scale=arguments.vad_lexicon_scale, match="surface_then_suffix"
    )
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    role_index = {agent.role: index for index, agent in enumerate(roster.agents)}

    written = 0
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in arguments.seeds:
            neutral = arguments.pairs_root / f"{arguments.pair_prefix}-neutral-s{seed}"
            affective = arguments.pairs_root / f"{arguments.pair_prefix}-affective-s{seed}"
            report = check_pair(neutral, affective)
            if not report.matched:
                msg = f"seed {seed} did not pass the pairing gate: {report.summary}"
                raise ValueError(msg)
            peers, price = _peers_and_price(neutral, arguments.treated_role)
            shock = _shock_round(neutral)

            # Within each seed, every replicate runs before the next seed starts.
            for replicate in range(1, arguments.replicates + 1):
                for condition in arguments.conditions:
                    use_peers = peers if condition == "embedded" else ()
                    use_price = price if condition == "embedded" else 100.0
                    use_round = shock if condition == "embedded" else 1
                    for arm, text in arms.items():
                        stimulus = StimulusInbox(
                            seed=seed,
                            arm=arm,
                            treated_role=arguments.treated_role,
                            treated_text=text,
                            peer_messages=use_peers,
                            treated_index=role_index[arguments.treated_role],
                        )
                        for reader_role in arguments.reader_roles:
                            # A distinct base per replicate, so the two passes are independent
                            # generations of one design rather than the same numbers twice.
                            generator = RoutedDecisionGenerator(
                                roster=roster,
                                backends=backends,
                                prompt_builder=FinancialPromptBuilder(),
                                run_seed=(CONTEXT_SEED_BASE + 100_000 * (replicate - 1) + seed),
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
                                row["replicate"] = replicate
                                row["rung"] = arguments.rung
                                row["price"] = use_price
                                row["n_peers"] = len(use_peers)
                                stream.write(json.dumps(row) + "\n")
                                written += 1
            print(f"seed {seed}: {written} readings so far", flush=True)
    print(f"wrote {written} readings to {arguments.output}")


if __name__ == "__main__":
    main()
