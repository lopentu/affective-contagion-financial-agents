"""The copying control, built from what each reader actually saw.

PR16 defines the control as deleting from every reply every word type that appeared in the message
the reader saw, taking the union over both arms so the same words leave both replies of a pair, plus
a closed list of affect-naming terms. An E36 reader saw the two payload renderings as the treated
sender's line, plus, in the embedded context only, the four other peers' messages from the neutral
source run (its own role excluded). This module builds that set per (seed, context, reader role)
and recomputes PR16's outcomes on the released readings.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from affective_contagion.text_affect import VADLexicon
from affective_contagion.verification import (
    _WORD,
    ECHO_TERMS,
    _arousal,
    _bootstrap,
    _paired_by_role,
    _release_rows,
)


@dataclass(frozen=True, slots=True)
class RoleEstimate:
    """One role's paired effect under the control, with its bootstrap interval."""

    role: str
    published: float
    controlled: float
    interval: tuple[float, float]
    pairs: int


@dataclass(frozen=True, slots=True)
class CopyControlResult:
    """PR16's outcomes recomputed from what each reader saw."""

    roles: tuple[RoleEstimate, ...]
    spread_published: float
    spread_controlled: float
    permutation_p: float
    mean_published: float
    mean_controlled: float
    overlap_min: float
    overlap_max: float
    permutation_draws: int


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").casefold()))


def input_words(
    renderings: Mapping[str, str],
    peers: Mapping[str, str],
    *,
    condition: str,
    reader_role: str,
    arm: str | None = None,
) -> set[str]:
    """Return the word types in what one reader saw.

    `arm=None` gives the union over both arms, which the control deletes; a named arm gives that
    arm's own input, which the overlap descriptive uses. Peers enter only in the embedded context
    and never include the reader's own role.
    """
    arms = renderings.values() if arm is None else (renderings[arm],)
    out: set[str] = set()
    for text in arms:
        out |= _words(text)
    if condition == "embedded":
        for role, text in peers.items():
            if role != reader_role:
                out |= _words(text)
    return out


def _peers_by_seed(inboxes: Sequence[dict], *, treated_role: str) -> dict[int, dict[str, str]]:
    """Map seed to the neutral source run's untreated peers' decision messages by role."""
    out: dict[int, dict[str, str]] = {}
    for entry in inboxes:
        if entry["arm"] != "neutral":
            continue
        out[int(entry["seed"])] = {
            message["role"]: message["decision_message"]
            for message in entry["messages"]
            if message["role"] != treated_role
        }
    return out


def _permutation_p(
    by_role: Mapping[str, Mapping[int, float]], observed: float, *, draws: int, seed: int = 0
) -> float:
    """Shuffle role labels within each pair and count spreads at least as large as observed."""
    roles = sorted(by_role)
    seeds = sorted(set.intersection(*(set(by_role[r]) for r in roles)))
    table = [[by_role[r][s] for r in roles] for s in seeds]
    rng = random.Random(seed)  # noqa: S311
    hits = 0
    for _ in range(draws):
        sums = [0.0] * len(roles)
        for row in table:
            shuffled = row[:]
            rng.shuffle(shuffled)
            for index, value in enumerate(shuffled):
                sums[index] += value
        spread = statistics.stdev(total / len(seeds) for total in sums)
        hits += spread >= observed - 1e-12
    return (hits + 1) / (draws + 1)


def copy_control_by_inbox(
    manifest_path: Path,
    lexicon: VADLexicon,
    renderings: Mapping[str, str],
    *,
    treated_role: str = "sentiment_analyst",
    draws: int = 2000,
) -> CopyControlResult:
    """Recompute PR16's outcomes with the deleted set built from each reader's own inbox."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _release_rows(manifest_path.parent / manifest["readings"]["file"])
    inboxes = _release_rows(manifest_path.parent / manifest["inboxes"]["file"])
    peers = _peers_by_seed(inboxes, treated_role=treated_role)
    echo = {term.casefold() for term in ECHO_TERMS}
    arm_key = {"neutral": "neutral", "anxiety_like": "anxiety_like"}

    def controlled(row: dict) -> float:
        drop = (
            input_words(
                renderings,
                peers[int(row["seed"])],
                condition=row["condition"],
                reader_role=row["reader_role"],
            )
            | echo
        )
        kept = " ".join(
            w for w in _WORD.findall((row["message"] or "").casefold()) if w not in drop
        )
        return _arousal(lexicon, kept, scrub=True)

    published = _paired_by_role(rows, lexicon)
    under_control = _paired_by_role(rows, lexicon, controlled)
    roles = tuple(
        RoleEstimate(
            role=role,
            published=statistics.fmean(published[role].values()),
            controlled=statistics.fmean(under_control[role].values()),
            interval=_bootstrap(list(under_control[role].values())),
            pairs=len(under_control[role]),
        )
        for role in sorted(under_control)
    )
    spread_published = statistics.stdev(r.published for r in roles)
    spread_controlled = statistics.stdev(r.controlled for r in roles)
    overlaps: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        seen = input_words(
            renderings,
            peers[int(row["seed"])],
            condition=row["condition"],
            reader_role=row["reader_role"],
            arm=arm_key[row["arm"]],
        )
        types = _words(row["message"])
        if types:
            overlaps.setdefault((row["reader_role"], row["arm"]), []).append(
                len(types & seen) / len(types)
            )
    cell_means = [statistics.fmean(v) for v in overlaps.values()]
    return CopyControlResult(
        roles=roles,
        spread_published=spread_published,
        spread_controlled=spread_controlled,
        permutation_p=_permutation_p(under_control, spread_controlled, draws=draws),
        mean_published=statistics.fmean(r.published for r in roles),
        mean_controlled=statistics.fmean(r.controlled for r in roles),
        overlap_min=min(cell_means),
        overlap_max=max(cell_means),
        permutation_draws=draws,
    )
