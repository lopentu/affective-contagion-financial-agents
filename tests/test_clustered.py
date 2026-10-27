"""Tests for clustered estimation of the isolated-reader effect.

Draws over one stimulus pair are not independent. Counting 780 draws over 12 stimulus pairs as 780
observations would shrink every interval by roughly the square root of the cluster size.
"""

from __future__ import annotations

import pytest

from affective_contagion.clustered import Observation, clustered_effect

ROLES = ("fundamental_analyst", "bull_researcher", "trader")


def _observations(
    effect_by_seed: dict[int, float], *, draws: int = 5, spread: float = 0.0
) -> list[Observation]:
    """Build readings where each seed carries a known effect, replicated over roles and draws."""
    out: list[Observation] = []
    for seed, effect in effect_by_seed.items():
        for role_index, role in enumerate(ROLES):
            for draw in range(draws):
                # Within-cell jitter: extra draws must average this away without adding confidence.
                jitter = spread * ((draw % 2) * 2 - 1)
                out.append(Observation(seed=seed, role=role, arm="neutral", value=0.40 + jitter))
                out.append(
                    Observation(
                        seed=seed,
                        role=role,
                        arm="affective",
                        value=0.40 + effect + jitter + role_index * 0.0,
                    )
                )
    return out


def test_the_effect_is_recovered_at_both_levels(tmp_path: object) -> None:
    """A constant effect must appear at both clustering levels, with equal point estimates."""
    del tmp_path
    result = clustered_effect(_observations({0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}))
    assert result.by_seed.mean_difference == pytest.approx(0.02)
    assert result.by_cell.mean_difference == pytest.approx(0.02)
    assert result.seeds == 4
    assert result.cells == 4 * len(ROLES)


def test_the_by_seed_interval_is_the_wider_one(tmp_path: object) -> None:
    """Roles within a seed share a stimulus, so treating them as independent is over-confident.

    With variation between seeds but none between roles, the by-cell estimator sees 12 observations
    where only 4 independent stimuli exist, and its interval is correspondingly too narrow.
    """
    del tmp_path
    result = clustered_effect(_observations({0: 0.05, 1: -0.01, 2: 0.03, 3: 0.00}))
    seed_low, seed_high = result.by_seed.mean_difference_ci
    cell_low, cell_high = result.by_cell.mean_difference_ci
    assert (seed_high - seed_low) > (cell_high - cell_low)
    assert result.variance_inflation > 1.0


def test_extra_draws_average_noise_without_narrowing_the_primary_interval(
    tmp_path: object,
) -> None:
    """Draws buy measurement precision, not degrees of freedom.

    Within-cell jitter cancels in the cell mean, so the by-seed interval must be unchanged whether
    each cell holds two draws or twenty. If more draws narrowed it, the estimator would be inventing
    independent information.
    """
    del tmp_path
    effects = {0: 0.05, 1: -0.01, 2: 0.03, 3: 0.00}
    few = clustered_effect(_observations(effects, draws=2, spread=0.02))
    many = clustered_effect(_observations(effects, draws=20, spread=0.02))
    assert few.by_seed.mean_difference == pytest.approx(many.by_seed.mean_difference)
    assert few.by_seed.difference_sd == pytest.approx(many.by_seed.difference_sd)
    assert few.by_seed.n == many.by_seed.n == 4


def test_a_cell_missing_an_arm_is_rejected(tmp_path: object) -> None:
    """A half-populated cell cannot be differenced, and must not silently vanish from the design."""
    del tmp_path
    observations = _observations({0: 0.02, 1: 0.02})
    kept = [
        item
        for item in observations
        if not (item.seed == 1 and item.role == ROLES[0] and item.arm == "affective")
    ]
    with pytest.raises(ValueError, match="missing an arm"):
        clustered_effect(kept)


def test_empty_input_is_rejected(tmp_path: object) -> None:
    """Empty input raises rather than returning a vacuous result."""
    del tmp_path
    with pytest.raises(ValueError, match="no observations"):
        clustered_effect([])
