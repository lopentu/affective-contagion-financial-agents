"""The PR1 cells must differ only in seed and arm.

If any other field differed between the arms, a measured difference could come from that field
rather than the treatment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from affective_contagion.config import ExperimentConfig
from affective_contagion.configuration import load_yaml_model
from affective_contagion.seeds import GENERATED_SEEDS

PR1 = Path("configs/experiments/pr1")


def _body_digest(path: Path) -> str:
    """Hash a config with its comments, seed, and arm removed."""
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("#", "seed:", "shock_type:"))
    ]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def test_all_cells_are_present() -> None:
    """Every generated seed has one config per arm: 110 seeds, 220 configs."""
    assert sorted(path.name for path in PR1.glob("*.yaml")) == sorted(
        f"{arm}_s{seed}.yaml" for arm in ("neutral", "affective") for seed in GENERATED_SEEDS
    )


def test_cells_differ_only_in_seed_and_arm() -> None:
    """One distinct body across every file, or the arms are not matched."""
    assert len({_body_digest(path) for path in PR1.glob("*.yaml")}) == 1


def test_every_cell_loads_and_holds_the_design_pr1_froze() -> None:
    """A config that parses but drifts from the frozen design must fail."""
    for path in sorted(PR1.glob("*.yaml")):
        config = load_yaml_model(path, ExperimentConfig)
        assert config.n_rounds == 7
        assert config.shock_round == 4
        # A unit impulse. Anything else makes the response one to sustained input.
        assert config.shock_duration == 1
        # A named treated role is what makes a peer effect transmission rather than shared exposure.
        assert config.treated_role == "sentiment_analyst"
        assert config.topology == "full"
        assert config.message_intervention == "identity"
        expected_arm = "neutral" if path.name.startswith("neutral") else "affective"
        assert config.shock_type == expected_arm
        assert config.seed == int(path.stem.split("_s")[1])
        # Three blocks share this generator: 6-67 for PR1/PR3, 400-423 for PR7 and 500-523 for PR8.
        assert config.seed in set(GENERATED_SEEDS)
        # PR1's primary outcome lives at shock_round + 1, so the run must be long enough to have it.
        assert config.n_rounds > config.shock_round + 1
