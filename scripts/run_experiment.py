"""Run a configured local Hugging Face simulation with immutable outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from affective_contagion.activation import load_affect_direction
from affective_contagion.backends import HuggingFaceCausalLMBackend
from affective_contagion.config import ExperimentConfig
from affective_contagion.configuration import load_yaml_model
from affective_contagion.diagnostics import evaluate_run_health
from affective_contagion.generation import (
    GenerationBackend,
    RoutedDecisionGenerator,
)
from affective_contagion.interventions import InjectedMessageIntervention
from affective_contagion.market import MarketParameters, SimpleMarket
from affective_contagion.payload import FinancialPayload, matched_realizations
from affective_contagion.prompts import FinancialPromptBuilder
from affective_contagion.provenance import collect_runtime_provenance
from affective_contagion.roster import ModelRoster
from affective_contagion.simulation import StageTrace, run_simulation
from affective_contagion.storage import (
    append_run_status,
    create_run_directory,
    save_simulation_result,
)
from affective_contagion.text_affect import VADLexicon, validate_affect_separation

FIREWALL_SEED_OFFSET = 10_000
"""Offset for the intervention seed recorded in each run manifest."""


class NumPyNormalSource:
    """Independent market-noise stream."""

    def __init__(self, seed: int) -> None:
        """Create a dedicated NumPy random stream."""
        self._random = np.random.default_rng(seed)

    def normal(self) -> float:
        """Draw one standard-normal innovation."""
        return float(self._random.normal())


def parse_arguments() -> argparse.Namespace:
    """Parse run configuration and immutable destination identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Canonical FinancialPayload YAML. Both arms are rendered from it.",
    )
    parser.add_argument("--directions", type=Path, nargs="*", default=[])
    parser.add_argument("--outputs", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--shock-keywords",
        nargs="*",
        default=[],
        help=(
            "Prespecified terms counted in shock-window messages as a treatment-uptake "
            "check. Freeze these before confirmatory runs; empty records window size only."
        ),
    )
    parser.add_argument(
        "--vad-match",
        choices=("surface", "surface_then_suffix"),
        default="surface_then_suffix",
        help=(
            "Token matching. surface uses exact forms only; surface_then_suffix falls back to a "
            "small suffix strip when a form is absent, raising coverage without lemmatising."
        ),
    )
    parser.add_argument(
        "--vad-lexicon",
        type=Path,
        required=True,
        help=("NRC-style VAD lexicon. Required; there is no default."),
    )
    parser.add_argument(
        "--vad-lexicon-scale",
        choices=("unit", "signed"),
        default="unit",
        help="Source range of the lexicon: unit for NRC VAD v1.0, signed for v2.x.",
    )
    return parser.parse_args()


def main() -> None:
    """Construct backends, run one condition, and persist its evidence."""
    arguments = parse_arguments()
    config = load_yaml_model(arguments.experiment, ExperimentConfig)
    roster = load_yaml_model(arguments.roster, ModelRoster)
    if tuple(agent.role for agent in roster.agents) != config.roles:
        msg = "experiment roles and roster agent order must match exactly"
        raise ValueError(msg)
    backends: dict[str, GenerationBackend] = {
        model.key: HuggingFaceCausalLMBackend(model) for model in roster.models
    }
    loaded_directions = [load_affect_direction(path) for path in arguments.directions]
    directions = {item.model_key: item for item in loaded_directions}
    generator = RoutedDecisionGenerator(
        roster=roster,
        backends=backends,
        prompt_builder=FinancialPromptBuilder(),
        run_seed=config.seed,
        affect_directions=directions,
    )
    market = SimpleMarket(
        MarketParameters(
            config.fundamental_value, config.price_impact, config.mean_reversion, config.noise_sigma
        ),
        NumPyNormalSource(config.seed),
    )
    # Both arms come from one payload, so they cannot drift apart. `matched_realizations` refuses
    # to return a pair whose financial content differs, which is the precondition for attributing
    # any outcome difference to wording rather than to information.
    payload = load_yaml_model(arguments.scenario, FinancialPayload)
    arms = matched_realizations(payload)
    stimuli = {"neutral": arms["neutral"], "affective": arms["anxiety_like"]}
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon,
        scale=arguments.vad_lexicon_scale,
        match=arguments.vad_match,
    )
    # Fail before any generation if the manipulation is not measurable in the intended direction.
    arousal_delta = validate_affect_separation(lexicon, arms)
    intervention_seed = config.seed + FIREWALL_SEED_OFFSET
    intervention = None
    if config.message_intervention == "inject_message":
        # PR5's treatment. The delivered text is one arm's rendering of the same payload, so the two
        # arms differ in exactly one delivered message at the treatment round. The treated agent
        # decides and trades before delivery, having read its arm's rendering, so its trade is not
        # held fixed.
        if config.treated_role is None:
            msg = "inject_message requires a treated_role: there is no sender to replace"
            raise ValueError(msg)
        if config.treated_role not in config.roles:
            msg = f"treated_role {config.treated_role!r} is not in roles"
            raise ValueError(msg)
        injected = {"neutral": arms["neutral"], "affective": arms["anxiety_like"]}[
            config.shock_type
        ]
        intervention = InjectedMessageIntervention(
            text=injected,
            sender_index=config.roles.index(config.treated_role),
            round_index=config.shock_round,
        )
    trace = StageTrace()
    destination = create_run_directory(arguments.outputs, arguments.run_id)
    append_run_status(destination, "started")
    try:
        result = run_simulation(
            config,
            generator,
            market,
            shocks=stimuli,
            intervention=intervention,
            vad_lexicon=lexicon,
            trace=trace,
        )
    except Exception as error:
        append_run_status(destination, "failed", f"{type(error).__name__}: {error}")
        raise
    health = evaluate_run_health(
        result,
        shock_round=config.shock_round,
        shock_duration=config.shock_duration,
        shock_keywords=tuple(arguments.shock_keywords),
        treated_role=config.treated_role,
    )
    provenance = collect_runtime_provenance(Path.cwd(), Path("uv.lock"))
    save_simulation_result(
        destination,
        config=config,
        roster=roster,
        result=result,
        provenance=provenance,
        stimuli=stimuli,
        payload=payload.to_dict(),
        stimulus_arousal_delta=arousal_delta,
        trace=trace,
        intervention_seed=intervention_seed if intervention is not None else None,
        run_health=health,
        vad_lexicon_source=(
            f"{arguments.vad_lexicon}"
            f"#scale={arguments.vad_lexicon_scale}#match={arguments.vad_match}"
        ),
    )
    append_run_status(destination, "completed")
    print(f"saved completed run to {destination}")


if __name__ == "__main__":
    main()
