"""Manipulation check for whether the treatment reaches the message channel at all.

This check runs between a matched pair of conditions and any propagation estimate from them. The
stimulus arms separate on arousal, but the treated agent paraphrases the stimulus into its own
message, and the affect need not survive that step. If it does not, peers never receive a
manipulation and a null propagation result says nothing about affect.

The comparison is paired on seed, because seed dominates run-to-run variance far more than condition
does. Pairing also yields the standard deviation of differences, which is what determines how many
seeds a confirmatory run would need.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# Two-sided alpha 0.05 with 80% power: z(0.975) + z(0.80).
#
# This is the large-sample approximation, which treats the pilot's SD as the known population sigma.
# It is optimistic at pilot sizes. The exact paired t calculation needs more seeds, because both the
# critical value and the SD estimate carry finite-sample penalties; at df=5 the two t quantiles sum
# to 3.49 against 2.80 here. Read every number it produces as a floor, not as exact.
_Z_TWO_SIDED_80_POWER = 1.959964 + 0.841621

# Two-sided 95% t quantiles by degrees of freedom, for interval estimates at pilot sizes where the
# normal approximation is visibly wrong. Above 30 df the normal quantile is close enough.
_T_QUANTILE_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class SeedPair:
    """One seed's treated-agent arousal under each arm."""

    seed: int
    neutral: float
    anxiety_like: float

    @property
    def difference(self) -> float:
        """Return how much more aroused the treated agent's own message was under treatment."""
        return self.anxiety_like - self.neutral


@dataclass(frozen=True, slots=True)
class FirstHopResult:
    """Paired first-hop separation with the precision needed to size a confirmatory run."""

    pairs: tuple[SeedPair, ...]
    treated_role: str
    shock_round: int

    @property
    def mean_difference(self) -> float:
        """Return the paired mean difference in arousal."""
        return statistics.mean(pair.difference for pair in self.pairs)

    @property
    def difference_sd(self) -> float:
        """Return the standard deviation of paired differences."""
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

        A pilot whose interval spans both zero and complete transmission has not distinguished
        those hypotheses, whatever its mean.
        """
        n = len(self.pairs)
        sd = self.difference_sd
        if n < 2 or math.isnan(sd):
            return (float("nan"), float("nan"))
        quantile = _T_QUANTILE_975.get(n - 1, 1.959964)
        half_width = quantile * sd / math.sqrt(n)
        return (self.mean_difference - half_width, self.mean_difference + half_width)

    @property
    def positive_seeds(self) -> int:
        """Count seeds where treatment raised arousal.

        A split near half means the sign is not stable across seeds, which no amount of extra
        precision on the mean will fix.
        """
        return sum(pair.difference > 0 for pair in self.pairs)

    def minimum_detectable_difference(self, seeds: int) -> float:
        """Return the smallest paired difference this design could detect with ``seeds`` seeds.

        Optimistic at pilot sizes; see the note on `_Z_TWO_SIDED_80_POWER`.
        Read it as "no smaller than this", not as an exact threshold.
        """
        if seeds < 1:
            msg = "seeds must be positive"
            raise ValueError(msg)
        return _Z_TWO_SIDED_80_POWER * self.difference_sd / math.sqrt(seeds)

    def seeds_required(self, assumed_difference: float) -> float:
        """Return the seeds needed to detect a *hypothesised* difference at conventional power.

        ``assumed_difference`` must be a difference worth detecting, a theoretical quantity such as
        complete transmission, chosen before looking at the outcome. Passing this pilot's own
        `mean_difference` produces a number that describes sampling noise rather than any effect.
        The point estimate is one draw from an interval that may span zero, so the "seeds needed"
        figure inherits that instability and can move by orders of magnitude between pilots of the
        same design.
        """
        if assumed_difference <= 0:
            msg = "assumed_difference must be positive"
            raise ValueError(msg)
        return (_Z_TWO_SIDED_80_POWER * self.difference_sd / assumed_difference) ** 2


def _treated_arousal(run: Path) -> tuple[int, str, int, float]:
    """Return seed, treated role, shock round, and the treated agent's shock-round arousal."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    treated_role = experiment["treated_role"]
    if treated_role is None:
        msg = f"{run.name} broadcast the stimulus, so it has no first hop to measure"
        raise ValueError(msg)
    shock_round = experiment["shock_round"]
    for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["round_index"] == shock_round and record["role"] == treated_role:
            affect = record["message_affect"]
            if affect is None:
                msg = f"{run.name} recorded no message affect; rerun with --vad-lexicon"
                raise ValueError(msg)
            arousal = float(affect["arousal"])
            if math.isnan(arousal):
                # A NaN here would propagate through mean and SD without raising, producing a NaN
                # effect size that reads as "undefined spread" rather than "corrupt input".
                msg = f"{run.name} recorded NaN arousal for {treated_role} at round {shock_round}"
                raise ValueError(msg)
            return experiment["seed"], treated_role, shock_round, arousal
    msg = f"{run.name} has no record for {treated_role} at round {shock_round}"
    raise ValueError(msg)


def first_hop_separation(neutral_runs: list[Path], affective_runs: list[Path]) -> FirstHopResult:
    """Pair matched runs by seed and measure whether the treatment survived the first hop."""
    if not neutral_runs or not affective_runs:
        msg = "both arms need at least one run"
        raise ValueError(msg)
    neutral = {}
    affective = {}
    roles: set[str] = set()
    rounds: set[int] = set()
    for runs, target in ((neutral_runs, neutral), (affective_runs, affective)):
        for run in runs:
            seed, role, shock_round, arousal = _treated_arousal(run)
            if seed in target:
                msg = f"seed {seed} appears twice in one arm"
                raise ValueError(msg)
            target[seed] = arousal
            roles.add(role)
            rounds.add(shock_round)
    if len(roles) != 1 or len(rounds) != 1:
        msg = f"arms must share one treated role and shock round; saw {roles} and {rounds}"
        raise ValueError(msg)
    shared = sorted(set(neutral) & set(affective))
    if not shared:
        msg = "no seed is present in both arms, so nothing can be paired"
        raise ValueError(msg)
    # Refuse to pair on the intersection alone. Dropping a seed present in one arm changes both the
    # mean and the SD of differences, and the SD sizes later runs, so a run that crashed in one arm
    # would silently shrink the apparent variance. The caller must rerun the missing cell or
    # exclude the seed on purpose.
    unmatched = set(neutral) ^ set(affective)
    if unmatched:
        neutral_only = sorted(set(neutral) - set(affective))
        affective_only = sorted(set(affective) - set(neutral))
        msg = (
            f"arms are not paired: seeds {neutral_only} appear only in the neutral arm and "
            f"{affective_only} only in the affective arm. Rerun the missing cells, or pass matched "
            f"subsets explicitly if the exclusion is deliberate."
        )
        raise ValueError(msg)
    return FirstHopResult(
        pairs=tuple(SeedPair(seed, neutral[seed], affective[seed]) for seed in shared),
        treated_role=roles.pop(),
        shock_round=rounds.pop(),
    )
