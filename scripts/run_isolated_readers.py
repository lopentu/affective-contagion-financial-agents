"""Run PR2's isolated-reader experiment over a set of matched stimulus pairs.

Three arms, all declared in the plan:

* **real**: the treated agent's actual delivered message, rebuilt into the full five-message inbox a
  peer would have read. PR2's primary. It measures what an anxious agent's message does to a reader,
  wording and content together, because E15 showed the sender changes both.
* **matched**: a payload-rendered pair in place of the agent's message, so the arms differ only in
  wording. This isolates tone. It is a bench test: nothing here scripts an agent's output inside a
  simulation, so it cannot establish transmission through an agent.
* **silent**: an empty inbox, the no-communication baseline. Without it, a difference cannot be read
  as the anxious arm raising arousal rather than the neutral arm lowering it.

Each pair gives one reading per (reader role, draw, arm). Price is the neutral run's post-shock
value and is used for every arm, so the price channel is closed.
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
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.roster import ModelRoster
from affective_contagion.text_affect import VADLexicon

# Base seed for reader draws, far above every simulation run seed.
READER_SEED_BASE = 900_000


def _shock_round(run: Path) -> int:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["experiment"]["shock_round"])


def _messages_at_shock(
    run: Path, treated_role: str
) -> tuple[str, tuple[tuple[int, str, str], ...]]:
    """Return the treated agent's delivered text and every peer's, at the shock round."""
    shock = _shock_round(run)
    treated = ""
    peers: list[tuple[int, str, str]] = []
    for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["round_index"] != shock:
            continue
        # Use the delivered text, which is what a peer reads. It equals the raw decision under the
        # identity intervention and is the injected report under injection.
        text = record["delivery"]["delivered_text"]
        if record["role"] == treated_role:
            treated = text
        else:
            peers.append((record["agent_index"], record["role"], text))
    return treated, tuple(sorted(peers))


def _price_after_shock(run: Path) -> float:
    """Return the price a peer observes in the round after the shock."""
    shock = _shock_round(run)
    for line in (run / "market.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step = json.loads(line)
        if step["round_index"] == shock:
            return float(step["price"])
    msg = f"{run.name} has no market step at the shock round"
    raise ValueError(msg)


def _build_arms(
    neutral: Path,
    affective: Path,
    treated_role: str,
    treated_index: int,
    arm: str,
    matched: dict[str, str] | None,
) -> dict[str, StimulusInbox]:
    """Build the two inboxes for one pair under one arm."""
    seed = json.loads((neutral / "manifest.json").read_text(encoding="utf-8"))["experiment"]["seed"]
    n_treated, n_peers = _messages_at_shock(neutral, treated_role)
    a_treated, a_peers = _messages_at_shock(affective, treated_role)
    # The peers' shock-round messages must be byte-identical across arms: they have read nothing yet
    # and see the same price, so anything else means the pairing gate let a broken pair through and
    # the "only the treated message differs" property no longer holds.
    if n_peers != a_peers:
        differing = [
            role
            for (_, role, text), (_, _, other) in zip(n_peers, a_peers, strict=True)
            if text != other
        ]
        msg = f"seed {seed}: peer messages differ across arms at the shock round ({differing})"
        raise ValueError(msg)
    if arm == "matched" and matched is not None:
        n_treated, a_treated = matched["neutral"], matched["anxiety_like"]
    if arm == "silent":
        # One condition only. With no message there is nothing for the arms to differ in, and
        # emitting neutral and anxiety_like would generate byte-identical duplicates.
        return {"silent": StimulusInbox(seed, "silent", treated_role, "", (), treated_index)}
    treated_texts = {"neutral": n_treated, "anxiety_like": a_treated}
    peers = n_peers
    return {
        name: StimulusInbox(
            seed=seed,
            arm=name,
            treated_role=treated_role,
            treated_text=text,
            peer_messages=peers,
            treated_index=treated_index,
        )
        for name, text in treated_texts.items()
    }


def main() -> None:
    """Generate every reading PR2 specifies and write them as one JSONL file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--vad-lexicon", type=Path, required=True)
    parser.add_argument("--vad-lexicon-scale", choices=("unit", "signed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=["real", "matched", "silent"])
    parser.add_argument("--matched-stimuli", type=Path, help="JSON with neutral/anxiety_like text.")
    parser.add_argument("--treated-role", default="sentiment_analyst")
    parser.add_argument(
        "--pair-prefix",
        default="s30",
        help="Run-id prefix of the stimulus pairs, e.g. s30 for the PR1/PR3 batch or pr7 for the "
        "belief pilot.",
    )
    parser.add_argument(
        "--cognition",
        action="store_true",
        help="Ask readers for expected return, confidence and downside risk. "
        "Changes the output grammar, so runs with and without it are not comparable.",
    )
    arguments = parser.parse_args()

    roster = load_yaml_model(arguments.roster, ModelRoster)
    lexicon: VADLexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon,
        scale=arguments.vad_lexicon_scale,
        match="surface_then_suffix",
    )
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    matched = (
        json.loads(arguments.matched_stimuli.read_text(encoding="utf-8"))
        if arguments.matched_stimuli
        else None
    )
    reader_roles = [a.role for a in roster.agents if a.role != arguments.treated_role]
    role_index = {a.role: i for i, a in enumerate(roster.agents)}

    if arguments.output.exists():
        msg = f"{arguments.output} exists; refusing to overwrite recorded readings"
        raise FileExistsError(msg)
    written = 0
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        for seed in arguments.seeds:
            neutral = arguments.pairs_root / f"{arguments.pair_prefix}-neutral-s{seed}"
            affective = arguments.pairs_root / f"{arguments.pair_prefix}-affective-s{seed}"
            # Refuse a pair that failed the pairing gate rather than quietly including it.
            report = check_pair(neutral, affective)
            if not report.matched:
                msg = f"seed {seed} did not pass the pairing gate: {report.summary}"
                raise ValueError(msg)
            price = _price_after_shock(neutral)
            for arm in arguments.arms:
                inboxes = _build_arms(
                    neutral,
                    affective,
                    arguments.treated_role,
                    role_index[arguments.treated_role],
                    arm,
                    matched,
                )
                for condition, stimulus in inboxes.items():
                    # One reader seed per pair, shared across conditions so both draw the same
                    # generation seeds. These common random numbers reduce variance.
                    generator = RoutedDecisionGenerator(
                        roster=roster,
                        backends=backends,
                        prompt_builder=FinancialPromptBuilder(
                            cognition_outcomes=arguments.cognition
                        ),
                        run_seed=READER_SEED_BASE + seed,
                        affect_directions={},
                    )
                    for role in reader_roles:
                        for reading in read_in_isolation(
                            generator,
                            stimulus,
                            reader_role=role,
                            reader_index=role_index[role],
                            price=price,
                            fundamental_value=100.0,
                            draws=arguments.draws,
                            vad_lexicon=lexicon,
                        ):
                            row: dict[str, Any] = asdict(reading)
                            row["affect"] = asdict(reading.affect)
                            row["stimulus_arm"] = arm
                            row["condition"] = condition
                            row["price"] = price
                            stream.write(json.dumps(row) + "\n")
                            written += 1
            print(f"seed {seed}: {written} readings so far", flush=True)
    print(f"wrote {written} readings to {arguments.output}")


if __name__ == "__main__":
    main()
