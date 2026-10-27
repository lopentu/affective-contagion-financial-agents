"""Show readers three renderings of one payload: calm, high-arousal, and high-arousal filtered.

This runs the firewall arm of PR17, and with `--include-no-closing` the four-arm design of PR24. It
is a separate script rather than a flag on `run_context_comparison.py` because the filtered arm
needs a generation step of its own, with its own seed stream and its own gate, before any reader
sees it.

The design is E36's with one arm added: the same matched pairs, five reader roles, two contexts,
lexicon and scrub. A third text is produced for each pair by passing the high-arousal rendering
through the model with the neutralising instruction.

Two contrasts are kept apart:

* **Does the filter reduce what the reader writes?** `filtered` against `anxiety_like`.
* **Does it reduce it to the calm baseline, or past it?** `filtered` against `neutral`. A filter
  that lands well below `neutral` has flattened the message rather than neutralised it, which is a
  cost.

Every rewrite is checked against the payload before any reader sees it. A rewrite that lost a
number or a decision slot is written to the rewrite log, and its pair is excluded from the reading
pass and counted. Without this check, a filter that deleted the bad news would score as one that
works.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from affective_contagion.backends.huggingface import HuggingFaceCausalLMBackend
from affective_contagion.configuration import load_yaml_model
from affective_contagion.firewall import filter_message
from affective_contagion.generation import GenerationBackend, RoutedDecisionGenerator
from affective_contagion.isolated_reader import StimulusInbox, read_in_isolation
from affective_contagion.pairing import check_pair
from affective_contagion.payload import FinancialPayload, matched_realizations
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.roster import ModelRoster
from affective_contagion.text_affect import VADLexicon

# Base seed for this design's reader draws; each reader design has its own base.
FIREWALL_SEED_BASE = 700_000
# The rewrite stream is separate again, so adding or removing a rewrite cannot shift a
# reader's seed.
REWRITE_SEED_BASE = 600_000


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
    parser.add_argument(
        "--rewrite-log",
        type=Path,
        required=True,
        help="Every rewrite with its preservation report, kept whether or not it passed. Without "
        "this the filter's failure rate is unmeasurable and the arm cannot be interpreted.",
    )
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
    parser.add_argument(
        "--filter-model-key",
        default="",
        help="Roster key of the model performing the rewrite. Defaults to the treated agent's own "
        "model, so the filter is part of the system being studied.",
    )
    parser.add_argument(
        "--include-no-closing",
        action="store_true",
        help="Add a fourth arm: the payload's no-closing rendering (frame plus verbatim fact "
        "block, no self-referential sentence). Puts the filtered and no-closing texts in ONE "
        "design, which is the within-run version of the cross-experiment comparison E45 made.",
    )
    return parser


def _rewrite_all(
    seeds: list[int],
    payload: FinancialPayload,
    anxious_text: str,
    backend: GenerationBackend,
    log: Path,
) -> dict[int, str]:
    """Filter the high-arousal rendering once per seed and return the usable rewrites.

    One rewrite per seed rather than one overall, because the rewrite is a generation and its
    variability is part of the filter. A single rewrite reused across every pair would understate
    the spread of the filtered arm and make the comparison look more precise than it is.
    """
    usable: dict[int, str] = {}
    with log.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in seeds:
            result = filter_message(backend, payload, anxious_text, seed=REWRITE_SEED_BASE + seed)
            stream.write(
                json.dumps(
                    {
                        "seed": seed,
                        "original": result.original,
                        "filtered": result.filtered,
                        "generation_seed": result.generation_seed,
                        "preserved": result.report.preserved,
                        "empty": result.report.empty,
                        "missing_numbers": list(result.report.missing_numbers),
                        "missing_slots": list(result.report.missing_slots),
                        "summary": result.report.summary,
                    }
                )
                + "\n"
            )
            if result.usable:
                usable[seed] = result.filtered
            else:
                print(f"seed {seed}: rewrite REJECTED - {result.report.summary}", flush=True)
    kept, total = len(usable), len(seeds)
    print(f"rewrites: {kept} of {total} preserved the payload ({kept / total:.1%})", flush=True)
    return usable


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

    The arms share a seed base, so they use common random numbers rather than independent draws.
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
                    run_seed=FIREWALL_SEED_BASE + seed,
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
    """Generate the rewrites, gate them, then run every reading the arms need."""
    arguments = _parser().parse_args()
    if arguments.output.exists():
        msg = f"{arguments.output} exists; refusing to overwrite recorded readings"
        raise FileExistsError(msg)

    roster = load_yaml_model(arguments.roster, ModelRoster)
    payload = load_yaml_model(arguments.scenario, FinancialPayload)
    arms = matched_realizations(payload)
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon, scale=arguments.vad_lexicon_scale, match="surface_then_suffix"
    )
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    role_index = {agent.role: index for index, agent in enumerate(roster.agents)}
    treated_index = role_index[arguments.treated_role]
    filter_key = arguments.filter_model_key or roster.agents[treated_index].model_key
    if filter_key not in backends:
        msg = f"filter model key {filter_key!r} is not in the roster"
        raise ValueError(msg)

    # Run the pairing gate over every seed before any rewriting.
    for seed in arguments.seeds:
        report = check_pair(
            arguments.pairs_root / f"{arguments.pair_prefix}-neutral-s{seed}",
            arguments.pairs_root / f"{arguments.pair_prefix}-affective-s{seed}",
        )
        if not report.matched:
            msg = f"seed {seed} did not pass the pairing gate: {report.summary}"
            raise ValueError(msg)

    filtered = _rewrite_all(
        arguments.seeds, payload, arms["anxiety_like"], backends[filter_key], arguments.rewrite_log
    )
    if not filtered:
        msg = "no rewrite preserved the payload; there is no filtered arm to read"
        raise ValueError(msg)

    written = 0
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in arguments.seeds:
            if seed not in filtered:
                # Excluded from every arm, not only the filtered one: dropping a pair from one arm
                # would leave the remaining arms comparing different sets of pairs.
                continue
            texts = {
                "neutral": arms["neutral"],
                "anxiety_like": arms["anxiety_like"],
                "filtered": filtered[seed],
            }
            if arguments.include_no_closing:
                texts["no_closing"] = payload.realize("no_closing")
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
    print(f"pairs used: {len(filtered)} of {len(arguments.seeds)}")


if __name__ == "__main__":
    main()
