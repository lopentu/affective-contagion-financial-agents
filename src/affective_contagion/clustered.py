"""Estimate an effect from observations sharing stimuli, without treating them as independent.

`isolated_reader` produces many readings per stimulus message. Many draws over a few message pairs
are not that many independent observations, and an interval that treats them so understates
uncertainty by roughly the square root of the cluster size.

The structure is nested. A draw sits inside a (seed, role) cell, and a cell's two arms read the
*same pair* of messages. The design is paired at the cell level and replicated within it. The
estimator:

1. averages the draws within each (seed, role) cell, per arm, which shrinks measurement noise
   without adding degrees of freedom;
2. differences the two arms within each cell, which removes any cell-level effect (a role's
   baseline verbosity, a seed's particular message) exactly rather than by adjustment;
3. treats the resulting cell differences as the units of analysis.

Cells sharing a *seed* still share the stimulus message pair, so they are not independent either.
Two intervals are reported:

- **by cell** (n = seeds x roles): narrower, valid only if role responses to the same message are
  independent, which they are not entirely.
- **by seed** (n = seeds): wider, valid if different seeds' message pairs are independent, which
  holds because each comes from a separately seeded run.

The by-seed interval is primary because its assumption holds. The gap between the two intervals
shows how much apparent precision comes from double-counting.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from affective_contagion.paired import PairedComparison, SeedPair


@dataclass(frozen=True, slots=True)
class Observation:
    """One measured value, tagged with the cluster it belongs to."""

    seed: int
    role: str
    arm: str
    value: float


@dataclass(frozen=True, slots=True)
class ClusteredEffect:
    """A paired effect reported at both levels of clustering, so neither can be quoted alone."""

    by_seed: PairedComparison
    by_cell: PairedComparison
    draws_per_cell: tuple[int, ...]
    cells: int
    seeds: int

    @property
    def variance_inflation(self) -> float:
        """Return the by-seed SD of paired differences divided by the by-cell SD.

        This compares spread only. Interval widths also depend on the two sample sizes and their t
        critical values. A large value means the cells sharing a seed are far from independent.
        """
        cell_sd = self.by_cell.difference_sd
        if not cell_sd:
            return float("nan")
        return self.by_seed.difference_sd / cell_sd

    @property
    def summary(self) -> str:
        """State the primary estimate, then the naive one, so the gap is visible."""
        low, high = self.by_seed.mean_difference_ci
        naive_low, naive_high = self.by_cell.mean_difference_ci
        return (
            f"by seed (PRIMARY, n={self.by_seed.n}): {self.by_seed.mean_difference:+.4f} "
            f"CI [{low:+.4f}, {high:+.4f}] dz {self.by_seed.cohens_dz:+.3f} "
            f"signs {self.by_seed.positive_seeds}/{self.by_seed.n}\n"
            f"by cell (naive,   n={self.by_cell.n}): {self.by_cell.mean_difference:+.4f} "
            f"CI [{naive_low:+.4f}, {naive_high:+.4f}] - treats roles reading the same message as "
            f"independent, so this interval is too narrow"
        )


def clustered_effect(
    observations: list[Observation],
    *,
    neutral_arm: str = "neutral",
    affective_arm: str = "affective",
) -> ClusteredEffect:
    """Difference two arms within each (seed, role) cell, then aggregate at both levels."""
    if not observations:
        msg = "no observations"
        raise ValueError(msg)
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for item in observations:
        grouped[(item.seed, item.role, item.arm)].append(item.value)

    cells = sorted({(seed, role) for seed, role, _ in grouped})
    cell_differences: dict[tuple[int, str], float] = {}
    draws: list[int] = []
    for seed, role in cells:
        left = grouped.get((seed, role, neutral_arm))
        right = grouped.get((seed, role, affective_arm))
        if not left or not right:
            msg = (
                f"cell (seed {seed}, role {role!r}) is missing an arm, so cannot be differenced; "
                f"drop it deliberately or rerun it"
            )
            raise ValueError(msg)
        draws.extend((len(left), len(right)))
        cell_differences[(seed, role)] = statistics.mean(right) - statistics.mean(left)

    # By cell: every cell difference is one observation. Narrower and over-confident.
    by_cell = PairedComparison(
        tuple(
            # PairedComparison takes (neutral, affective) pairs, so encode an already-computed
            # difference as a pair against zero. The arithmetic is identical and the interval
            # logic, including the t quantile, is shared rather than reimplemented.
            SeedPair(seed=index, neutral=0.0, anxiety_like=value)
            for index, value in enumerate(cell_differences[cell] for cell in cells)
        )
    )
    # By seed: average the roles within a seed first, so each independently seeded message pair
    # contributes exactly one number.
    per_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _role), value in cell_differences.items():
        per_seed[seed].append(value)
    by_seed = PairedComparison(
        tuple(
            SeedPair(seed=seed, neutral=0.0, anxiety_like=statistics.mean(values))
            for seed, values in sorted(per_seed.items())
        )
    )
    return ClusteredEffect(
        by_seed=by_seed,
        by_cell=by_cell,
        draws_per_cell=tuple(sorted(set(draws))),
        cells=len(cells),
        seeds=len(per_seed),
    )
