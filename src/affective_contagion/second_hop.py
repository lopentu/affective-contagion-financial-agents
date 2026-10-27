"""Test whether the treatment reached agents who never read it, and whether price could explain it.

This step follows `first_hop`. Peers cannot receive a manipulation the sender never sent, so
`SecondHopReport` carries both results and its verdict does not read the peer result without the
sender result.

Agents read only the previous round's messages and the treated agent speaks at the shock round, so
shock+1 is the only round where a peer message can reflect the treatment through exactly one hop.
Shock+2 mixes that with peer-to-peer re-transmission. Analysis plan PR1 fixes this window.

The treatment reaches a shock+1 peer through two paths. The treated agent's message is the path of
interest. Its trade also moves price, and every agent observes price. A price-mediated effect is
transmission but not affective transmission, so PR1 measures it.

Within round `r` every agent observes `market.price`, and the round ends with `market.clear`, so
`market.jsonl[r]` holds the price a round `r+1` agent sees. A round-5 agent's `context.price` equals
`market.jsonl[4].price`. Both checks therefore read the shock round: imbalance because the treated
agent's shock-round trade lands there, price because peers then observe that value. Peers observe
price directly, while imbalance enters only through price.

Matched arms are byte-identical before the shock, and their shock-round peer messages are
byte-identical too, because agents are stateless and generation seeds do not depend on content.
Any shock-round price difference therefore comes from the treated agent's own trade. That
difference is deterministic within a pair, so it needs a count of affected seeds as well as a mean
test.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from affective_contagion.paired import PairedComparison, pair_by_seed, sign_test_threshold

# A run whose parser failed this often cannot support a message-level outcome. A fallback keeps up
# to 400 characters of the unparsed output as the message and substitutes HOLD at size 0, so
# failures distort both the action and the message outcomes. Set by PR1.
MAX_PARSE_FAILURE_RATE = 0.10

# A first-hop separation assumes complete, lossless onward transmission and is too high to use as
# the minimum detectable difference for a second-hop null. This ceiling is 25% of the observed
# first hop.
INFORMATIVE_NULL_CEILING = 0.0103


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    """Everything one run contributes, read once so the arms cannot be read inconsistently."""

    seed: int
    treated_role: str
    shock_round: int
    parse_failure_rate: float
    peer_arousal: dict[int, float]
    treated_arousal: dict[int, float]
    order_imbalance: dict[int, float]
    price: dict[int, float]
    # The treated agent's shock-round trade, as (action, size). It finds the seeds where price is
    # bit-identical across arms, and checks whether what peers received differed in recommended
    # action as well as tone.
    treated_decision: tuple[str, float]
    # Parse outcomes at the two cells the primary outcome is computed from. A run-level failure rate
    # can sit under its ceiling while a failure lands squarely on a measured message.
    measurement_parse_ok: bool
    min_coverage: float


def _record_arousal(run_name: str, record: dict) -> float:
    """Return one record's arousal, refusing anything that is not a real measurement.

    A NaN would propagate through mean and SD as undefined spread, and a zero-coverage message would
    enter the outcome as the neutral 0.5 default, which sits inside the range peer messages occupy.
    """
    affect = record["message_affect"]
    if affect is None:
        msg = f"{run_name} recorded no message affect; rerun with --vad-lexicon"
        raise ValueError(msg)
    arousal = float(affect["arousal"])
    if math.isnan(arousal):
        msg = f"{run_name} recorded NaN arousal at round {record['round_index']}"
        raise ValueError(msg)
    if float(affect["coverage"]) == 0.0:
        msg = (
            f"{run_name} round {record['round_index']} role {record['role']}: no message token is "
            f"in the lexicon, so its arousal is the neutral 0.5 default rather than a measurement. "
            f"Refusing to average it into the outcome."
        )
        raise ValueError(msg)
    return arousal


def _read_run(run: Path) -> RunMeasurement:
    """Read one run into the measurements PR1 needs, failing loudly on anything missing."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    treated_role = experiment["treated_role"]
    if treated_role is None:
        msg = f"{run.name} broadcast the stimulus, so it has no untreated peers to measure"
        raise ValueError(msg)

    shock_round = experiment["shock_round"]
    peer_round = shock_round + 1
    peer: dict[int, list[float]] = {}
    treated: dict[int, float] = {}
    treated_decision: tuple[str, float] = ("MISSING", float("nan"))
    measurement_parse_ok = True
    coverages: list[float] = []
    for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        arousal = _record_arousal(run.name, record)
        coverages.append(float(record["message_affect"]["coverage"]))
        round_index = record["round_index"]
        is_treated = record["role"] == treated_role
        if is_treated:
            treated[round_index] = arousal
            if round_index == shock_round:
                treated_decision = (
                    str(record["decision"]["action"]),
                    float(record["decision"]["size"]),
                )
        else:
            peer.setdefault(round_index, []).append(arousal)
        # A run-level failure rate can sit under its ceiling while a failure lands on one of the
        # two cells the primary outcome is computed from. `fallback_hold` both scores raw text and
        # delivers it, so such a failure corrupts the measurement and the channel at once.
        in_measurement_cell = (is_treated and round_index == shock_round) or (
            not is_treated and round_index == peer_round
        )
        if in_measurement_cell and not record["parse_succeeded"]:
            measurement_parse_ok = False

    market_path = run / "market.jsonl"
    imbalance: dict[int, float] = {}
    price: dict[int, float] = {}
    for line in market_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step = json.loads(line)
        imbalance[step["round_index"]] = float(step["order_imbalance"])
        price[step["round_index"]] = float(step["price"])

    return RunMeasurement(
        seed=experiment["seed"],
        treated_role=treated_role,
        shock_round=experiment["shock_round"],
        parse_failure_rate=float(manifest["parse_failure_rate"]),
        # Mean across peers within a run, so each seed contributes one number and the pairing stays
        # at the seed level. Averaging first also means a talkative agent cannot dominate the arm.
        peer_arousal={index: statistics.mean(values) for index, values in peer.items()},
        treated_arousal=treated,
        order_imbalance=imbalance,
        price=price,
        treated_decision=treated_decision,
        measurement_parse_ok=measurement_parse_ok,
        min_coverage=min(coverages) if coverages else float("nan"),
    )


@dataclass(frozen=True, slots=True)
class SecondHopReport:
    """PR1's primary outcome with the two checks that decide whether it can be read."""

    peer_round: int
    shock_round: int
    treated_role: str
    peers: PairedComparison
    first_hop: PairedComparison
    order_imbalance: PairedComparison
    price: PairedComparison
    # Placebo. Under `full` topology the treated agent cannot read its own shock-round message, so
    # its shock+1 prompt differs across arms only through price. Any arousal difference here is a
    # price-channel effect with no linguistic component, a within-run negative control.
    treated_placebo: PairedComparison
    # Clean subset: seeds where the treated agent's shock-round action and size match across arms,
    # so price is bit-identical and the peer difference is purely linguistic.
    clean_subset: PairedComparison | None
    clean_seeds: tuple[int, ...]
    # A count as well as a mean test, because within a pair the price difference is deterministic
    # rather than noisy.
    seeds_with_price_difference: int
    excluded_seeds: tuple[int, ...]
    minimum_coverage: float

    @property
    def first_hop_replicated(self) -> bool:
        """Report whether the sender transmitted anything in these seeds.

        If this is false the peer result is *uninterpretable* rather than negative: there was no
        manipulation in the channel for peers to receive.
        """
        return self.first_hop.excludes_zero and self.first_hop.mean_difference > 0

    @property
    def informative_if_null(self) -> bool:
        """Report whether a null here would be informative.

        A null is informative only if the design could have detected an effect worth caring about.
        Measured against `INFORMATIVE_NULL_CEILING` rather than against complete transmission.
        """
        return self.peers.minimum_detectable_difference(self.peers.n) <= INFORMATIVE_NULL_CEILING

    @property
    def reversed_direction(self) -> bool:
        """Report peers damping under the affective arm, with a negative interval excluding zero.

        This is a directional result, not a failure to detect, so it has its own verdict.
        """
        return self.peers.excludes_zero and self.peers.mean_difference < 0

    @property
    def price_channel_active(self) -> bool:
        """Report whether the arms differ on the non-linguistic path.

        True means the peer difference cannot be attributed to language: the treated agent also
        traded differently and peers observe price. PR1 then requires reporting an unseparated
        language-or-price path rather than affective transmission.
        """
        return self.order_imbalance.excludes_zero or self.price.excludes_zero

    @property
    def sign_rule_met(self) -> bool:
        """Require enough seeds to share a sign for a one-sided binomial test at 0.05.

        A fixed proportion imposes a floor of 0.674 standard deviations that no sample size
        removes, and it becomes weaker as n grows when the positive rate converges below 0.75. The
        binomial threshold scales with the requested alpha instead. For n = 12 it requires 10
        positive seeds.
        """
        return self.peers.positive_seeds >= sign_test_threshold(self.peers.n)

    @property
    def peer_criteria_met(self) -> bool:
        """Report PR1's two peer criteria: a positive interval excluding zero, and the sign rule."""
        return self.peers.excludes_zero and self.peers.mean_difference > 0 and self.sign_rule_met

    @property
    def supported(self) -> bool:
        """Report PR1's full verdict: the peer criteria *and* both gates.

        This is stricter than the peer criteria alone. A finding also requires a replicated first
        hop and an inactive price channel. `verdict` says which condition failed;
        this property reports only whether all conditions passed.
        """
        return (
            self.first_hop_replicated and self.peer_criteria_met and not self.price_channel_active
        )

    @property
    def verdict(self) -> str:
        """Return the one sentence PR1 licenses, and no stronger one."""
        low, high = self.peers.mean_difference_ci
        summary = (
            f"peer difference {self.peers.mean_difference:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}], "
            f"signs {self.peers.positive_seeds}/{self.peers.n}"
        )
        if not self.first_hop_replicated:
            return (
                "UNINTERPRETABLE: the first hop did not replicate in these seeds, so peers never "
                "received a manipulation. The peer comparison says nothing about affect."
            )
        if self.reversed_direction:
            return (
                f"REVERSED: {summary}. Peers were *less* aroused under the affective arm, with an "
                f"interval excluding zero. This is a directional result, not a failure to detect, "
                f"and it was not the hypothesis - report it as unexpected."
            )
        if not self.peer_criteria_met:
            floor = self.peers.minimum_detectable_difference(self.peers.n)
            if not self.informative_if_null:
                return (
                    f"UNINFORMATIVE NULL: {summary}. The design could detect no less than "
                    f"{floor:.4f}, above the {INFORMATIVE_NULL_CEILING:.4f} worth caring about, so "
                    f"this says nothing about whether transmission occurred. Do not report it as "
                    f"evidence of absence; it is evidence that this design cannot answer."
                )
            return (
                f"NOT SUPPORTED: {summary}. The design could detect {floor:.4f}, below the "
                f"{INFORMATIVE_NULL_CEILING:.4f} threshold, so this null is informative about "
                f"effects that large - but it is still a failure to detect, not a demonstration of "
                f"absence."
            )
        if self.price_channel_active:
            clean = (
                "no seed has an identical shock-round trade, so no clean subset exists"
                if self.clean_subset is None
                else (
                    f"on the {self.clean_subset.n} seed(s) where the trade is identical the "
                    f"difference is {self.clean_subset.mean_difference:+.4f}"
                )
            )
            return (
                f"CONFOUNDED: {summary}, but the arms also differ in shock-round trading in "
                f"{self.seeds_with_price_difference} of {self.peers.n} seeds and peers observe "
                f"price. Report as an unseparated language-or-price path, not as affective "
                f"transmission. Sensitivity: {clean}."
            )
        return (
            f"SUPPORTED: {summary}, with no detectable price difference and a placebo of "
            f"{self.treated_placebo.mean_difference:+.4f}. The linguistic reading is permitted; "
            f"absence of a price effect at this n is not proof of it, and PR1 A1.3 requires "
            f"checking that what peers received matched on payload before calling this affective."
        )


def second_hop_separation(
    neutral_runs: list[Path],
    affective_runs: list[Path],
    *,
    peer_round_offset: int = 1,
) -> SecondHopReport:
    """Compute PR1's primary outcome and its two gating checks from matched run directories.

    ``peer_round_offset`` defaults to 1, the single-hop round. A larger offset computes the
    compounded round as an exploratory secondary outcome.
    """
    measurements: dict[str, dict[int, RunMeasurement]] = {"neutral": {}, "affective": {}}
    for runs, arm in ((neutral_runs, "neutral"), (affective_runs, "affective")):
        for run in runs:
            measurement = _read_run(run)
            if measurement.seed in measurements[arm]:
                msg = f"seed {measurement.seed} appears twice in the {arm} arm"
                raise ValueError(msg)
            measurements[arm][measurement.seed] = measurement

    every = [*measurements["neutral"].values(), *measurements["affective"].values()]
    roles = {item.treated_role for item in every}
    shock_rounds = {item.shock_round for item in every}
    if len(roles) != 1 or len(shock_rounds) != 1:
        msg = f"arms must share one treated role and shock round; saw {roles} and {shock_rounds}"
        raise ValueError(msg)
    shock_round = shock_rounds.pop()
    peer_round = shock_round + peer_round_offset

    # PR1's exclusion rule: a seed is dropped from both arms if either run is over the run-level
    # parse-failure ceiling, or if a parse failure landed on one of the two cells the primary
    # outcome is computed from. `fallback_hold` both scores and delivers raw text, so one failure
    # there corrupts the measurement and the channel while the run-level rate stays under its
    # ceiling. Applied before any outcome is computed, and reported either way.
    excluded = {
        item.seed
        for item in every
        if item.parse_failure_rate > MAX_PARSE_FAILURE_RATE or not item.measurement_parse_ok
    }
    keep = (set(measurements["neutral"]) & set(measurements["affective"])) - excluded
    if not keep:
        msg = "no seed survives pairing and the parse-failure exclusions"
        raise ValueError(msg)

    def compare(field: str, round_index: int) -> PairedComparison:
        arms: list[dict[int, float]] = []
        for arm in ("neutral", "affective"):
            values: dict[int, float] = {}
            for seed in sorted(keep):
                series: dict[int, float] = getattr(measurements[arm][seed], field)
                if round_index not in series:
                    msg = (
                        f"{arm} seed {seed} has no {field} at round {round_index}; the run is too "
                        f"short for this comparison"
                    )
                    raise ValueError(msg)
                values[seed] = series[round_index]
            arms.append(values)
        return PairedComparison(pair_by_seed(arms[0], arms[1]))

    # Clean subset. Where the treated agent's shock-round action and size are identical across
    # arms, the fill is identical, so price is bit-identical and peers differ only in the text they
    # read. That subset gives the only purely linguistic estimate this design admits.
    clean_seeds = tuple(
        seed
        for seed in sorted(keep)
        if measurements["neutral"][seed].treated_decision
        == measurements["affective"][seed].treated_decision
    )
    peer_values = {
        arm: {seed: measurements[arm][seed].peer_arousal[peer_round] for seed in clean_seeds}
        for arm in ("neutral", "affective")
    }
    clean_subset = (
        PairedComparison(pair_by_seed(peer_values["neutral"], peer_values["affective"]))
        if clean_seeds
        else None
    )
    # Within a pair the price difference is deterministic, so a mean test can pass while individual
    # pairs each carried an open channel. Count the affected seeds as well.
    price_differences = sum(
        measurements["neutral"][seed].price[shock_round]
        != measurements["affective"][seed].price[shock_round]
        for seed in sorted(keep)
    )
    return SecondHopReport(
        peer_round=peer_round,
        shock_round=shock_round,
        treated_role=roles.pop(),
        peers=compare("peer_arousal", peer_round),
        first_hop=compare("treated_arousal", shock_round),
        order_imbalance=compare("order_imbalance", shock_round),
        price=compare("price", shock_round),
        treated_placebo=compare("treated_arousal", peer_round),
        clean_subset=clean_subset,
        clean_seeds=clean_seeds,
        seeds_with_price_difference=price_differences,
        excluded_seeds=tuple(sorted(excluded)),
        minimum_coverage=min(item.min_coverage for item in every),
    )
