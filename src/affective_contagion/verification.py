"""Independently recompute published quantities and verify released archive hashes.

Functions use retained generations or explicit numeric inputs. Data absent from the release
cannot be recovered by these estimators; see docs/reproducibility.md for the release's limits.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

from affective_contagion.scrub import scrub_decision_vocabulary
from affective_contagion.text_affect import VADLexicon

# Terms marking teammate messages that name the sender's state rather than exhibit it.
ECHO_TERMS = ("alarmed", "anxious", "frightened", "dread", "alarming", "anxiety")
# Surface word split, matching the lexicon's own tokenisation.
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True, slots=True)
class Recomputed:
    """One recomputed number, with enough context to explain a mismatch."""

    key: str
    value: float
    interval: tuple[float, float] | None
    units: int
    detail: str


def _arousal(lexicon: VADLexicon, text: str, *, scrub: bool) -> float:
    """Score one message. `scrub` removes decision vocabulary, as the published analyses do."""
    prepared = scrub_decision_vocabulary(text or "") if scrub else (text or "")
    return float(lexicon.score(prepared).arousal)


def _bootstrap(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval over the mean. Seeded, so a rerun gives the same interval."""
    rng = random.Random(seed)  # noqa: S311
    draws = sorted(
        statistics.mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(resamples)
    )
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def matched_pairs(root: Path, *, treated_role: str = "sentiment_analyst") -> Iterator[dict]:
    """Yield {arm: {role: message}} for every complete matched pair of runs under `root`.

    Pairs are found by stripping the arm token from the run id. A run missing its partner is
    skipped rather than half-counted.
    """
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob("*/records.jsonl")):
        name = path.parent.name
        arm = "affective" if "affective" in name else ("neutral" if "neutral" in name else None)
        if arm is None:
            continue
        grouped.setdefault(name.replace(f"-{arm}", ""), {})[arm] = path
    for arms in grouped.values():
        if len(arms) == 2:
            yield {"paths": arms, "treated_role": treated_role}


def _teammate_messages(path: Path, round_index: int, treated_role: str) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        row["role"]: row["decision"]["message"]
        for row in rows
        if row["round_index"] == round_index and row["role"] != treated_role
    }


def _manifest_treated_role(pair: dict) -> str:
    """Read the treated role from the neutral run's manifest, which is the only authority for it."""
    run = pair["paths"]["neutral"].parent
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["experiment"]["treated_role"])


def peer_response(
    root: Path,
    lexicon: VADLexicon,
    *,
    round_index: int,
    expected_teammates: int = 5,
    drop_echo: bool = False,
) -> Recomputed:
    """Sum over agents other than the sentiment analyst of (anxious minus calm) arousal.

    This route assumes the sentiment analyst was the sender in every pair. Where the treated role
    varies across pairs, as PR5 balances it, use `peer_response_by_manifest`, which reads the
    treated role per pair.
    """
    return _peer_response(
        root,
        lexicon,
        round_index=round_index,
        expected_teammates=expected_teammates,
        drop_echo=drop_echo,
        treated_for=lambda _pair: "sentiment_analyst",
        key="peer_response",
    )


def peer_response_by_manifest(
    root: Path,
    lexicon: VADLexicon,
    *,
    round_index: int,
    expected_teammates: int = 5,
    drop_echo: bool = False,
) -> Recomputed:
    """Sum over the untreated agents of (anxious minus calm) arousal, averaged over matched pairs.

    The treated sender is read from each pair's manifest and excluded; a pair whose teammate
    count is not `expected_teammates` is skipped rather than silently changing the divisor.
    """
    return _peer_response(
        root,
        lexicon,
        round_index=round_index,
        expected_teammates=expected_teammates,
        drop_echo=drop_echo,
        treated_for=_manifest_treated_role,
        key="peer_response_by_manifest",
    )


def treated_self_response(root: Path, lexicon: VADLexicon, *, round_index: int) -> Recomputed:
    """Average the treated agent's own (anxious minus calm) arousal at `round_index` over pairs.

    Under the injected design this agent has no news at round 9 and reads teammates whose round-8
    messages were identical across arms, so its own shift should be near zero unless the price it
    saw differed. It shows what `peer_response` adds when it counts the treated agent.
    """
    diffs: list[float] = []
    for pair in matched_pairs(root, treated_role=""):
        treated = _manifest_treated_role(pair)
        arms = {}
        for arm, path in pair["paths"].items():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            own = [r for r in rows if r["round_index"] == round_index and r["role"] == treated]
            if len(own) != 1:
                break
            arms[arm] = own[0]["decision"]["message"]
        if len(arms) != 2:
            continue
        diffs.append(
            _arousal(lexicon, arms["affective"], scrub=True)
            - _arousal(lexicon, arms["neutral"], scrub=True)
        )
    if not diffs:
        msg = f"no complete matched pairs under {root}"
        raise ValueError(msg)
    return Recomputed(
        key="treated_self_response",
        value=statistics.mean(diffs),
        interval=_bootstrap(diffs),
        units=len(diffs),
        detail=f"{len(diffs)} pairs, round {round_index}, the treated agent's own message",
    )


def _peer_response(
    root: Path,
    lexicon: VADLexicon,
    *,
    round_index: int,
    expected_teammates: int,
    drop_echo: bool,
    treated_for: Callable[[dict], str],
    key: str,
) -> Recomputed:
    sums: list[float] = []
    for pair in matched_pairs(root, treated_role=""):
        treated = treated_for(pair)
        arms = {
            arm: _teammate_messages(path, round_index, treated)
            for arm, path in pair["paths"].items()
        }
        if any(len(v) != expected_teammates for v in arms.values()):
            continue
        roles = set(arms["affective"]) & set(arms["neutral"])
        if drop_echo:
            roles = {
                r
                for r in roles
                if not any(w in (arms["affective"][r] or "").casefold() for w in ECHO_TERMS)
            }
            if not roles:
                continue
        total = sum(
            _arousal(lexicon, arms["affective"][r], scrub=True)
            - _arousal(lexicon, arms["neutral"][r], scrub=True)
            for r in roles
        )
        # Rescale when echo-dropping leaves fewer than the full set, so the figure stays comparable.
        sums.append(total * expected_teammates / len(roles))
    if not sums:
        msg = f"no complete matched pairs with {expected_teammates} teammates under {root}"
        raise ValueError(msg)
    mean = statistics.mean(sums)
    return Recomputed(
        key=key,
        value=mean,
        interval=_bootstrap(sums),
        units=len(sums),
        detail=f"{len(sums)} pairs, round {round_index}, {expected_teammates} teammates, "
        f"echo-dropped={drop_echo}",
    )


def exact_spearman_p(rho: float, n: int, *, one_sided: bool = False) -> Recomputed:
    """Exact permutation p for a Spearman correlation, by enumeration.

    The test is two-sided unless `one_sided` is set. It needs no data. At n = 7 the asymptotic
    formula gives 0.014 where the exact value is 0.024, so the value is enumerated.
    """
    if not 3 <= n <= 8:
        msg = f"enumeration is only sane for 3 <= n <= 8; got {n}"
        raise ValueError(msg)
    base = list(range(1, n + 1))
    mean_base = statistics.mean(base)
    denominator = sum((x - mean_base) ** 2 for x in base)
    hits = 0
    total = 0
    for candidate in permutations(base):
        total += 1
        numerator = sum(
            (x - mean_base) * (y - mean_base) for x, y in zip(base, candidate, strict=True)
        )
        value = numerator / denominator
        if one_sided:
            hit = value <= rho + 1e-9 if rho < 0 else value >= rho - 1e-9
        else:
            hit = abs(value) >= abs(rho) - 1e-9
        if hit:
            hits += 1
    return Recomputed(
        key="exact_spearman_p",
        value=hits / total,
        interval=None,
        units=total,
        detail=(
            f"{hits} of {total} orderings reach rho {'<=' if rho < 0 else '>='} {rho} (one-sided)"
            if one_sided
            else f"{hits} of {total} orderings reach |rho| >= {abs(rho)}"
        ),
    )


def _release_rows(path: Path) -> list[dict]:
    """Read one gzipped JSONL release file."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_release(manifest_path: Path) -> tuple[bool, str]:
    """Check a released bundle against the hashes its manifest pins.

    Without this check, a release edited after export would still produce numbers that agree
    with themselves.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for part in ("readings", "inboxes"):
        entry = manifest[part]
        path = manifest_path.parent / entry["file"]
        if not path.exists():
            return False, f"{entry['file']} is missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            return False, f"{entry['file']} does not match its manifest hash"
    return True, (
        f"{manifest['readings']['rows']} readings and {manifest['inboxes']['rows']} pair-arms, "
        f"hashes as recorded, {manifest['repository']} @ {manifest['revision'][:12]}"
    )


def _paired_by_role(
    rows: Sequence[dict],
    lexicon: VADLexicon,
    score: Callable[[dict], float] | None = None,
) -> dict[str, dict[int, float]]:
    """Per-role, per-pair anxious-minus-calm arousal, draws averaged then contexts pooled.

    Draws collapse to a (seed, role, context, arm) cell, arms difference within that cell, then
    contexts average within (seed, role). The route follows the design and does not reuse the
    original analysis code.
    """

    def default(row: dict) -> float:
        return _arousal(lexicon, row["message"], scrub=True)

    scorer = score or default
    cells: dict[tuple, list[float]] = {}
    for row in rows:
        key = (int(row["seed"]), row["reader_role"], row["condition"], row["arm"])
        cells.setdefault(key, []).append(float(scorer(row)))
    per_pair: dict[tuple[int, str], list[float]] = {}
    for (seed, role, condition, arm), values in cells.items():
        if arm != "anxiety_like":
            continue
        calm = cells.get((seed, role, condition, "neutral"))
        if calm:
            difference = statistics.fmean(values) - statistics.fmean(calm)
            per_pair.setdefault((seed, role), []).append(difference)
    out: dict[str, dict[int, float]] = {}
    for (seed, role), diffs in per_pair.items():
        out.setdefault(role, {})[seed] = statistics.fmean(diffs)
    return out


def role_effect(manifest_path: Path, lexicon: VADLexicon, *, role: str) -> Recomputed:
    """Recompute one reader role's arousal response from the released readings."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _release_rows(manifest_path.parent / manifest["readings"]["file"])
    by_role = _paired_by_role(rows, lexicon)
    if role not in by_role:
        msg = f"role {role!r} is not present in the release"
        raise KeyError(msg)
    values = list(by_role[role].values())
    return Recomputed(
        key=f"role_{role}",
        value=statistics.fmean(values),
        interval=None,
        units=len(values),
        detail=f"{len(values)} pairs, contexts pooled, decision vocabulary scrubbed",
    )


def _per_role_means(manifest_path: Path, lexicon: VADLexicon) -> list[float]:
    """Compute the five per-role paired means as published."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _release_rows(manifest_path.parent / manifest["readings"]["file"])
    by_role = _paired_by_role(rows, lexicon)
    return [statistics.fmean(list(pairs.values())) for pairs in by_role.values()]


def role_spread(manifest_path: Path, lexicon: VADLexicon) -> Recomputed:
    """Recompute the spread of the five role effects.

    The copy-controlled spread is computed by `copy_control.copy_control_by_inbox`.
    """
    means = _per_role_means(manifest_path, lexicon)
    return Recomputed(
        key="role_spread",
        value=statistics.stdev(means),
        interval=None,
        units=len(means),
        detail=f"{len(means)} roles, as published",
    )


def role_mean(manifest_path: Path, lexicon: VADLexicon) -> Recomputed:
    """Recompute the mean across the five role effects.

    The copy-controlled mean is computed by `copy_control.copy_control_by_inbox`.
    """
    means = _per_role_means(manifest_path, lexicon)
    return Recomputed(
        key="role_mean",
        value=statistics.fmean(means),
        interval=None,
        units=len(means),
        detail=f"{len(means)} roles, as published",
    )


# E19's eight rungs: parameter count in billions and the first-hop effect. Seven rungs come from
# E19 and the primary model's rung from E15 seeds 6-17. PR2 B3.3 fixes the trend test as a
# one-sided Spearman test across all eight.
LADDER_EFFECTS: tuple[tuple[float, float], ...] = (
    (2.3, 0.0178),
    (3.4, 0.0111),
    (4.0, 0.0173),
    (4.7, 0.0170),
    (5.1, 0.0169),
    (8.0, 0.0009),
    (8.8, 0.0075),
    (9.7, 0.0001),
)


def spearman_rho(pairs: Sequence[tuple[float, float]]) -> float:
    """Spearman's rho between the two columns of `pairs`, with no ties expected."""
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    rank_x = [sorted(xs).index(x) + 1 for x in xs]
    rank_y = [sorted(ys).index(y) + 1 for y in ys]
    n = len(pairs)
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rank_x, rank_y, strict=True)) / (
        n * (n * n - 1)
    )
