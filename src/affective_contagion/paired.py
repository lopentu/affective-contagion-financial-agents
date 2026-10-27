"""Paired-by-seed statistics shared by every matched comparison in this project.

`first_hop` and `second_hop` use these statistics and differ only in *which number* they pull
from a run, so interval reporting is consistent between them.

Everything pairs on seed, because seed dominates run-to-run variance far more than condition does.
Pairing also yields the standard deviation of *differences*, which is the quantity that sizes a
confirmatory run.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Two-sided alpha 0.05 with 80% power: z(0.975) + z(0.80).
#
# This is the large-sample approximation, which treats the pilot's SD as the known population sigma.
# It is *optimistic* at pilot sizes. The exact paired t calculation needs more seeds, because both
# the critical value and the SD estimate carry finite-sample penalties; at df=5 the two t quantiles
# sum to 3.49 against 2.80 here. Read every number it produces as a floor, not as an exact value.
Z_TWO_SIDED_80_POWER = 1.959964 + 0.841621

# Two-sided 95% t quantiles by degrees of freedom, for interval estimates at pilot sizes where the
# normal approximation is visibly wrong. Above 30 df the normal quantile is close enough.
T_QUANTILE_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class SeedPair:
    """One seed's measurement under each arm."""

    seed: int
    neutral: float
    anxiety_like: float

    @property
    def difference(self) -> float:
        """Return how much higher the treated arm's measurement was for this seed."""
        return self.anxiety_like - self.neutral


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """A matched difference reported with the precision needed to act on it."""

    pairs: tuple[SeedPair, ...]

    @property
    def n(self) -> int:
        """Return the number of paired seeds."""
        return len(self.pairs)

    @property
    def mean_difference(self) -> float:
        """Return the paired mean difference."""
        return statistics.mean(pair.difference for pair in self.pairs)

    @property
    def difference_sd(self) -> float:
        """Return the sample standard deviation of paired differences."""
        if len(self.pairs) < 2:
            return float("nan")
        return statistics.stdev(pair.difference for pair in self.pairs)

    @property
    def cohens_dz(self) -> float:
        """Return Cohen's d_z: the paired mean difference over the SD *of differences*.

        Named explicitly because d_z is not comparable to the between-group d that the same word
        usually denotes. d_z uses the SD of within-pair differences, so it grows when seeds are
        correlated across arms and cannot be read against between-group benchmarks.
        """
        sd = self.difference_sd
        if math.isnan(sd) or sd == 0.0:
            return float("nan")
        return self.mean_difference / sd

    @property
    def mean_difference_ci(self) -> tuple[float, float]:
        """Return the two-sided 95% t interval for the paired mean difference.

        A comparison whose interval spans both zero and the effect of interest has not
        distinguished those hypotheses, whatever its mean is.
        """
        n = len(self.pairs)
        sd = self.difference_sd
        if n < 2 or math.isnan(sd):
            return (float("nan"), float("nan"))
        quantile = T_QUANTILE_975.get(n - 1, 1.959964)
        half_width = quantile * sd / math.sqrt(n)
        return (self.mean_difference - half_width, self.mean_difference + half_width)

    @property
    def excludes_zero(self) -> bool:
        """Report whether the 95% interval lies wholly on one side of zero."""
        low, high = self.mean_difference_ci
        if math.isnan(low) or math.isnan(high):
            return False
        return low > 0.0 or high < 0.0

    @property
    def positive_seeds(self) -> int:
        """Count seeds where the treated arm measured higher.

        A split near half means the sign is not stable across seeds, which no amount of extra
        precision on the mean will fix.
        """
        return sum(pair.difference > 0 for pair in self.pairs)

    def minimum_detectable_difference(self, seeds: int) -> float:
        """Return the smallest paired difference this design could detect with ``seeds`` seeds.

        Optimistic at pilot sizes; see the note on `Z_TWO_SIDED_80_POWER`.
        Read it as "no smaller than this", not as an exact threshold.
        """
        if seeds < 1:
            msg = "seeds must be positive"
            raise ValueError(msg)
        return Z_TWO_SIDED_80_POWER * self.difference_sd / math.sqrt(seeds)

    def seeds_required(self, assumed_difference: float) -> float:
        """Return the seeds needed to detect a *hypothesised* difference at conventional power.

        ``assumed_difference`` must be a difference worth detecting, such as a theoretical quantity
        like complete transmission, chosen before looking at the outcome. Passing this comparison's
        own `mean_difference` produces a number that describes sampling noise rather than an
        effect. The point estimate is one draw from an interval that `mean_difference_ci` usually
        shows to span zero, so the "seeds needed" figure inherits that instability and can move by
        orders of magnitude between pilots of the same design.
        """
        if assumed_difference <= 0:
            msg = "assumed_difference must be positive"
            raise ValueError(msg)
        return (Z_TWO_SIDED_80_POWER * self.difference_sd / assumed_difference) ** 2


def sign_test_threshold(n: int, alpha: float = 0.05) -> int:
    """Return the seeds that must share a sign for a one-sided binomial sign test at ``alpha``.

    A fixed-proportion rule such as `ceil(0.75 n)` demands a true effect of at least
    `Phi^-1(0.75) = 0.674` standard deviations, a floor no sample size removes. Below that floor it
    becomes less likely to pass as the sample grows, because the observed positive rate converges
    to a value under 0.75.

    The binomial threshold scales with n instead: 10 of 12, 16 of 22, 20 of 30, 37 of 60.
    """
    if n < 1:
        msg = "n must be positive"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie strictly between 0 and 1"
        raise ValueError(msg)
    # Smallest k whose upper tail P(X >= k) under a fair coin does not exceed alpha.
    for k in range(n, -1, -1):
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
        if tail > alpha:
            return k + 1
    return 0


def pair_by_seed(neutral: dict[int, float], affective: dict[int, float]) -> tuple[SeedPair, ...]:
    """Pair two arms on seed, refusing anything but an exact match of seed sets.

    The intersection is refused. Dropping a seed present in only one arm changes both the mean and
    the SD of differences, and the SD sizes later runs, so a run that crashed in one arm would
    silently shrink the design's apparent variance. Failing here forces the operator to either
    rerun the missing cell or exclude the seed on purpose.
    """
    if not neutral or not affective:
        msg = "both arms need at least one run"
        raise ValueError(msg)
    if set(neutral) != set(affective):
        neutral_only = sorted(set(neutral) - set(affective))
        affective_only = sorted(set(affective) - set(neutral))
        msg = (
            f"arms are not paired: seeds {neutral_only} appear only in the neutral arm and "
            f"{affective_only} only in the affective arm. Rerun the missing cells, or pass matched "
            f"subsets explicitly if the exclusion is deliberate."
        )
        raise ValueError(msg)
    return tuple(SeedPair(seed, neutral[seed], affective[seed]) for seed in sorted(neutral))
