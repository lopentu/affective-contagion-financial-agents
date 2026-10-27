"""Tests for PR1's second-hop outcome and the checks that gate reading it.

These check that the analysis matches the frozen plan. A module that silently computed a different
round, or that reported an 8%-powered null as informative, would produce a wrong number that reading
the output could not reveal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from affective_contagion.second_hop import (
    INFORMATIVE_NULL_CEILING,
    second_hop_separation,
)

ROLES = ("sentiment_analyst", "fundamental_analyst", "bull_researcher", "trader")


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    treated_shock: float,
    peer_next: float,
    treated_next: float = 0.40,
    imbalance: float = 0.0,
    price: float = 100.0,
    action: str = "HOLD",
    size: float = 0.0,
    parse_ok: bool = True,
    coverage: float = 0.9,
) -> Path:
    """Write a minimal run with a shock at round 4 and one post-shock round."""
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "experiment": {"seed": seed, "treated_role": ROLES[0], "shock_round": 4},
                "parse_failure_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for round_index in (4, 5):
        for role in ROLES:
            treated = role == ROLES[0]
            if round_index == 4:
                arousal = treated_shock if treated else 0.30
            else:
                arousal = treated_next if treated else peer_next
            rows.append(
                {
                    "round_index": round_index,
                    "role": role,
                    "message_affect": {"arousal": arousal, "coverage": coverage},
                    "decision": {"action": action, "size": size},
                    # Only the treated shock-round cell and the peer next-round cells are
                    # measurement cells, so failures elsewhere must not exclude the seed.
                    "parse_succeeded": parse_ok if (treated and round_index == 4) else True,
                }
            )
    (run / "records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    market = [
        {"round_index": r, "order_imbalance": imbalance if r == 4 else 0.0, "price": price}
        for r in (4, 5)
    ]
    (run / "market.jsonl").write_text(
        "\n".join(json.dumps(row) for row in market) + "\n", encoding="utf-8"
    )
    return run


def _arms(
    tmp_path: Path,
    peers: dict[int, tuple[float, float]],
    *,
    coverage: float = 0.9,
) -> tuple[list[Path], list[Path]]:
    """Build matched arms from {seed: (neutral_peer, affective_peer)}."""
    neutral: list[Path] = []
    affective: list[Path] = []
    for seed, (low, high) in peers.items():
        neutral.append(
            _write_run(
                tmp_path,
                f"n{seed}",
                seed=seed,
                treated_shock=0.40,
                peer_next=low,
                coverage=coverage,
            )
        )
        affective.append(
            _write_run(
                tmp_path,
                f"a{seed}",
                seed=seed,
                treated_shock=0.50,
                peer_next=high,
                coverage=coverage,
            )
        )
    return neutral, affective


def test_the_primary_outcome_reads_the_single_hop_round_and_excludes_the_treated_agent(
    tmp_path: Path,
) -> None:
    """Shock+1, peers only. Reading the treated agent's own message would not be a second hop."""
    neutral, affective = _arms(tmp_path, {0: (0.40, 0.45), 1: (0.40, 0.45)})
    report = second_hop_separation(neutral, affective)
    assert report.shock_round == 4
    assert report.peer_round == 5
    assert report.peers.mean_difference == pytest.approx(0.05)
    # The treated agent's own shock-round message is the FIRST hop, reported separately.
    assert report.first_hop.mean_difference == pytest.approx(0.10)


def test_an_underpowered_null_is_labelled_uninformative(tmp_path: Path) -> None:
    """A wide null must not read as evidence of absence, as Amendment A1 requires."""
    # Differences scattered around zero with a spread far above the ceiling worth caring about.
    peers = {
        seed: (0.40, 0.40 + delta)
        for seed, delta in enumerate((0.05, -0.05, 0.04, -0.04, 0.03, -0.03))
    }
    neutral, affective = _arms(tmp_path, peers)
    report = second_hop_separation(neutral, affective)
    assert not report.supported
    assert not report.informative_if_null
    assert report.peers.minimum_detectable_difference(report.peers.n) > INFORMATIVE_NULL_CEILING
    assert "UNINFORMATIVE NULL" in report.verdict
    assert "cannot answer" in report.verdict


def test_a_reversal_is_not_reported_as_a_failure_to_detect(tmp_path: Path) -> None:
    """Peers damping under treatment is a directional result, not an absence of one."""
    peers = {seed: (0.50, 0.44 + seed * 0.001) for seed in range(6)}
    neutral, affective = _arms(tmp_path, peers)
    report = second_hop_separation(neutral, affective)
    assert report.reversed_direction
    assert "REVERSED" in report.verdict
    # It must say what happened, and say explicitly that this is not an absence of an effect.
    assert "not a failure to detect" in report.verdict
    assert "UNINFORMATIVE" not in report.verdict


def test_a_peer_result_without_a_first_hop_is_uninterpretable(tmp_path: Path) -> None:
    """Peers cannot receive a manipulation the sender never transmitted."""
    neutral, affective = [], []
    for seed in range(6):
        # Identical treated arousal in both arms: no first hop.
        neutral.append(
            _write_run(tmp_path, f"n{seed}", seed=seed, treated_shock=0.40, peer_next=0.40)
        )
        affective.append(
            _write_run(tmp_path, f"a{seed}", seed=seed, treated_shock=0.40, peer_next=0.48)
        )
    report = second_hop_separation(neutral, affective)
    assert not report.first_hop_replicated
    assert "UNINTERPRETABLE" in report.verdict


def test_the_clean_subset_holds_only_seeds_whose_trade_matched(tmp_path: Path) -> None:
    """Where the shock-round trade matches, price is identical, so the difference is textual."""
    neutral, affective = [], []
    for seed in range(4):
        # Seeds 0 and 1 trade identically; 2 and 3 diverge, so price could carry the effect.
        diverges = seed >= 2
        neutral.append(
            _write_run(tmp_path, f"n{seed}", seed=seed, treated_shock=0.40, peer_next=0.40)
        )
        affective.append(
            _write_run(
                tmp_path,
                f"a{seed}",
                seed=seed,
                treated_shock=0.50,
                peer_next=0.45,
                action="SELL" if diverges else "HOLD",
                size=0.5 if diverges else 0.0,
                price=99.0 if diverges else 100.0,
            )
        )
    report = second_hop_separation(neutral, affective)
    assert report.clean_seeds == (0, 1)
    assert report.clean_subset is not None
    assert report.seeds_with_price_difference == 2


def test_a_parse_failure_in_a_measurement_cell_excludes_the_seed_from_both_arms(
    tmp_path: Path,
) -> None:
    """A1.4: the run-level ceiling can pass while a failure lands on a measured message."""
    neutral, affective = [], []
    for seed in range(4):
        neutral.append(
            _write_run(tmp_path, f"n{seed}", seed=seed, treated_shock=0.40, peer_next=0.40)
        )
        affective.append(
            _write_run(
                tmp_path,
                f"a{seed}",
                seed=seed,
                treated_shock=0.50,
                peer_next=0.45,
                # Seed 2's treated shock-round message did not parse, so it was a fallback string.
                parse_ok=seed != 2,
            )
        )
    report = second_hop_separation(neutral, affective)
    assert report.excluded_seeds == (2,)
    assert report.peers.n == 3


def test_a_zero_coverage_message_is_refused_rather_than_scored_as_neutral(tmp_path: Path) -> None:
    """0.5 sits inside the range peer messages occupy, so it must not enter the outcome silently."""
    neutral, affective = _arms(tmp_path, {0: (0.40, 0.45)}, coverage=0.0)
    with pytest.raises(ValueError, match="no message token"):
        second_hop_separation(neutral, affective)


def test_the_placebo_reads_the_treated_agent_after_the_shock(tmp_path: Path) -> None:
    """Under full topology it cannot read its own message, so any difference is price alone."""
    neutral, affective = [], []
    for seed in range(4):
        neutral.append(
            _write_run(
                tmp_path,
                f"n{seed}",
                seed=seed,
                treated_shock=0.40,
                peer_next=0.40,
                treated_next=0.40,
            )
        )
        affective.append(
            _write_run(
                tmp_path,
                f"a{seed}",
                seed=seed,
                treated_shock=0.50,
                peer_next=0.45,
                treated_next=0.42,
            )
        )
    report = second_hop_separation(neutral, affective)
    assert report.treated_placebo.mean_difference == pytest.approx(0.02)
    assert not math.isnan(report.minimum_coverage)


def test_supported_applies_the_gates_not_only_the_peer_criteria(tmp_path: Path) -> None:
    """`.supported` must not return True on a result the verdict calls uninterpretable.

    A public attribute that answered only the two peer criteria would let a caller reaching for the
    obvious name publish a result whose first hop never replicated.
    """
    neutral, affective = [], []
    for seed in range(6):
        # Identical treated arousal: the peer criteria can pass while the first hop is absent.
        neutral.append(
            _write_run(tmp_path, f"n{seed}", seed=seed, treated_shock=0.40, peer_next=0.40)
        )
        affective.append(
            _write_run(
                tmp_path,
                f"a{seed}",
                seed=seed,
                treated_shock=0.40,
                peer_next=0.48 + seed * 0.001,
            )
        )
    report = second_hop_separation(neutral, affective)
    assert report.peer_criteria_met, "fixture should satisfy the peer criteria"
    assert not report.first_hop_replicated
    assert not report.supported
    assert "UNINTERPRETABLE" in report.verdict
