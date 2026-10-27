"""Tests for the shared paired statistics, especially the sign criterion.

The sign threshold scales with the sample size, as PR2 amendment B2.1 specifies. A fixed 75%
proportion would demand a true effect of at least 0.674 standard deviations whatever the sample
size, and would get *harder* to satisfy as evidence accumulated.
"""

from __future__ import annotations

import math
import random
from math import comb

import pytest

from affective_contagion.paired import sign_test_threshold


def test_the_threshold_matches_an_exact_one_sided_binomial_test() -> None:
    """The threshold must be the smallest count whose upper tail does not exceed alpha."""
    for n in range(1, 61):
        k = sign_test_threshold(n)
        tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
        assert tail <= 0.05
        if k > 0:
            looser = sum(comb(n, i) for i in range(k - 1, n + 1)) / 2**n
            assert looser > 0.05, f"n={n} threshold {k} is stricter than necessary"


def test_the_required_proportion_falls_as_the_sample_grows() -> None:
    """The required proportion must fall as the sample grows.

    A fixed `ceil(0.75 n)` holds the proportion at 0.75 for every n, which creates a detection floor
    no sample size can remove.
    """
    proportions = [sign_test_threshold(n) / n for n in (12, 22, 30, 60)]
    assert proportions == sorted(proportions, reverse=True)
    assert proportions[0] > 0.8
    assert proportions[-1] < 0.65


def test_it_is_stricter_at_twelve_and_looser_at_thirty_than_the_old_rule() -> None:
    """The binomial threshold is stricter than `ceil(0.75 n)` at twelve and looser at thirty.

    PR1's "9 of 12" does not clear 0.05 as a sign test, so at twelve seeds the threshold is higher.
    """
    assert sign_test_threshold(12) == 10 > math.ceil(0.75 * 12)
    assert sign_test_threshold(30) == 20 < math.ceil(0.75 * 30)


def test_power_against_the_benchmark_effect_rises_with_n() -> None:
    """Detection probability must rise with n, which a fixed proportion fails to do.

    At the benchmark effect of 0.0103 and spread of 0.0164, the detection probability of
    `ceil(0.75 n)` falls from 0.605 at twelve seeds to 0.151 at a thousand.
    """
    rng = random.Random(0)  # noqa: S311  # a power simulation, not cryptography

    def pass_rate(n: int, *, threshold: int) -> float:
        wins = 0
        trials = 4000
        for _ in range(trials):
            draws = [rng.gauss(0.0103, 0.0164) for _ in range(n)]
            if sum(x > 0 for x in draws) >= threshold:
                wins += 1
        return wins / trials

    scaling = [pass_rate(n, threshold=sign_test_threshold(n)) for n in (12, 60)]
    fixed = [pass_rate(n, threshold=math.ceil(0.75 * n)) for n in (12, 60)]
    assert scaling[1] > scaling[0], "the scaling rule must get stronger with more seeds"
    assert fixed[1] < fixed[0], "a fixed proportion must get weaker with more seeds"


def test_invalid_arguments_are_rejected() -> None:
    """A zero sample, or an out-of-range alpha, is a caller error worth raising on."""
    with pytest.raises(ValueError, match="n must be positive"):
        sign_test_threshold(0)
    with pytest.raises(ValueError, match="alpha must lie strictly between 0 and 1"):
        sign_test_threshold(12, alpha=1.0)
