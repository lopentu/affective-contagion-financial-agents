"""Check that a matched pair stayed matched until the treatment landed.

Run this between generating two arms and calculating any paired statistic from them. It is a
run-health check, because unmatched generation can be silent.

Matched arms share every input and every random stream, and the treatment reaches one agent at one
round. Two properties therefore *must* hold, by construction rather than by luck:

1. every pre-treatment generation is byte-identical across arms; and
2. at the treatment round, only the treated agent's generation differs - its peers have not yet read
   anything and see an identical price.

Generation can be nondeterministic even when prompts and generation seeds are byte-identical, so
both properties must be checked from the recorded outputs.

A pre-treatment divergence propagates. One agent's message enters every peer's next inbox, so a
single flipped token desynchronises the rest of the run. The paired difference then carries that
nuisance variation on top of the treatment, and the SD of differences is used to size later
experiments.

A run pair that fails the check can still be described, but its paired intervals understate
precision and must say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PairingReport:
    """How much of one matched pair stayed byte-identical before the treatment acted."""

    seed: int
    treated_role: str
    shock_round: int
    pre_treatment_cells: int
    pre_treatment_identical: int
    treatment_round_peers: int
    treatment_round_peers_identical: int
    treated_cell_differs: bool

    @property
    def pre_treatment_intact(self) -> bool:
        """Report whether the arms were identical everywhere before the treatment."""
        return self.pre_treatment_identical == self.pre_treatment_cells

    @property
    def treatment_round_intact(self) -> bool:
        """Report whether only the treated agent differed at the treatment round.

        Its peers have not read anything yet and observe the same price, so a difference here is
        nondeterminism rather than transmission.
        """
        return self.treatment_round_peers_identical == self.treatment_round_peers

    @property
    def matched(self) -> bool:
        """Report whether the pair is a valid matched pair, treatment included."""
        return (
            self.pre_treatment_intact and self.treatment_round_intact and self.treated_cell_differs
        )

    @property
    def summary(self) -> str:
        """Describe the pair in one line, naming the consequence rather than only the counts."""
        if self.matched:
            return (
                f"seed {self.seed}: MATCHED - {self.pre_treatment_identical}"
                f"/{self.pre_treatment_cells} pre-treatment cells identical, only the treated "
                f"agent differs at round {self.shock_round}"
            )
        if not self.treated_cell_differs:
            return (
                f"seed {self.seed}: NO TREATMENT EFFECT IN THE CHANNEL - the treated agent's "
                f"round-{self.shock_round} generation is byte-identical across arms, so the "
                f"manipulation did not change what it wrote at all"
            )
        return (
            f"seed {self.seed}: NOT MATCHED - only {self.pre_treatment_identical}"
            f"/{self.pre_treatment_cells} pre-treatment cells identical and "
            f"{self.treatment_round_peers_identical}/{self.treatment_round_peers} untreated peers "
            f"identical at round {self.shock_round}. Paired differences carry nondeterminism as "
            f"well as treatment, so the SD is inflated and intervals understate precision."
        )


def _generations(run: Path) -> tuple[dict[tuple[int, str], str], str, int]:
    """Return raw generations keyed by (round, role), plus the treated role and shock round."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    treated_role = experiment["treated_role"]
    if treated_role is None:
        msg = f"{run.name} broadcast the stimulus, so it has no single treated cell to check"
        raise ValueError(msg)
    generations = {
        (record["round_index"], record["role"]): record["generation"]["raw_output"]
        for record in (
            json.loads(line)
            for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return generations, treated_role, experiment["shock_round"]


def check_pair(neutral: Path, affective: Path) -> PairingReport:
    """Compare one matched pair cell by cell, before and at the treatment round."""
    left, role, shock_round = _generations(neutral)
    right, other_role, other_round = _generations(affective)
    if role != other_role or shock_round != other_round:
        msg = (
            f"arms disagree on the treatment: {role!r} at {shock_round} versus "
            f"{other_role!r} at {other_round}"
        )
        raise ValueError(msg)
    if set(left) != set(right):
        msg = "arms do not cover the same (round, role) cells, so they cannot be compared"
        raise ValueError(msg)
    pre = [cell for cell in left if cell[0] < shock_round]
    peers = [cell for cell in left if cell[0] == shock_round and cell[1] != role]
    treated_cell = (shock_round, role)
    manifest = json.loads((neutral / "manifest.json").read_text(encoding="utf-8"))
    seed = manifest["experiment"]["seed"]
    return PairingReport(
        seed=seed,
        treated_role=role,
        shock_round=shock_round,
        pre_treatment_cells=len(pre),
        pre_treatment_identical=sum(left[cell] == right[cell] for cell in pre),
        treatment_round_peers=len(peers),
        treatment_round_peers_identical=sum(left[cell] == right[cell] for cell in peers),
        treated_cell_differs=left[treated_cell] != right[treated_cell],
    )
