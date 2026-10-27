"""E51 outcomes 6 and 7: the per-role teammate table and the classifier discriminant, by manifest.

Both replace numbers that inherited E27's sender error. Outcome 6 re-reports E38's injected table
with each pair's treated role excluded, so every one of the six roles has 120 eligible pairs (the
144 minus the 24 in which it was the sender). Outcome 7 re-reports E43's addendum: the GoEmotions
discriminant, anxious minus calm, summed over the five untreated teammates and divided by five, with
and without dropping teammates whose anxious-arm message names the sender's state. The plan is PR26
Amendment C.

Usage:
    uv run python scripts/e51_teammates.py --runs outputs/runs/e51
        --vad-lexicon data/raw/NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt
        --output outputs/e51_teammates.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from affective_contagion.emotion_probe import TransformersEmotionScorer
from affective_contagion.text_affect import VADLexicon
from affective_contagion.verification import (
    ECHO_TERMS,
    _arousal,
    _bootstrap,
    _manifest_treated_role,
    matched_pairs,
)


def _round_messages(path: Path, round_index: int) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["role"]: r["decision"]["message"] for r in rows if r["round_index"] == round_index}


def per_role(root: Path, lexicon: VADLexicon, *, round_index: int) -> dict[str, dict]:
    """Outcome 6: each role's paired arousal shift over the pairs in which it was not the sender."""
    diffs: dict[str, list[float]] = {}
    for pair in matched_pairs(root, treated_role=""):
        treated = _manifest_treated_role(pair)
        arms = {arm: _round_messages(p, round_index) for arm, p in pair["paths"].items()}
        for role in set(arms["affective"]) & set(arms["neutral"]):
            if role == treated:
                continue
            diffs.setdefault(role, []).append(
                _arousal(lexicon, arms["affective"][role], scrub=True)
                - _arousal(lexicon, arms["neutral"][role], scrub=True)
            )
    return {
        role: {
            "value": statistics.fmean(values),
            "interval": _bootstrap(values),
            "pairs": len(values),
            "positive": sum(v > 0 for v in values),
        }
        for role, values in sorted(diffs.items())
    }


def discriminant(root: Path, *, round_index: int, drop_echo: bool) -> dict:
    """Outcome 7: the E43 discriminant per teammate; the treated role comes from the manifest."""
    scorer = TransformersEmotionScorer()
    texts: list[str] = []
    index: list[tuple[int, str, str]] = []
    pairs = list(matched_pairs(root, treated_role=""))
    for number, pair in enumerate(pairs):
        treated = _manifest_treated_role(pair)
        for arm, path in pair["paths"].items():
            for role, text in _round_messages(path, round_index).items():
                if role != treated:
                    texts.append(text or "")
                    index.append((number, arm, role))
    scores = dict(zip(index, scorer.score_all(texts), strict=True))
    sums: list[float] = []
    for number in range(len(pairs)):
        roles = {role for n, arm, role in index if n == number and arm == "affective"}
        if drop_echo:
            roles = {
                role
                for role in roles
                if not any(
                    w in (texts[index.index((number, "affective", role))] or "").casefold()
                    for w in ECHO_TERMS
                )
            }
        if not roles:
            continue
        total = sum(
            scores[(number, "affective", role)].discriminant
            - scores[(number, "neutral", role)].discriminant
            for role in roles
        )
        sums.append(total * 5 / len(roles))
    return {
        "per_teammate": statistics.fmean(sums) / 5,
        "interval_per_teammate": tuple(v / 5 for v in _bootstrap(sums)),
        "pairs": len(sums),
        "echo_dropped": drop_echo,
    }


def main() -> int:
    """Compute outcomes 6 and 7 at round 9 and write them as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--vad-lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=9)
    arguments = parser.parse_args()
    lexicon = VADLexicon.from_tsv(
        arguments.vad_lexicon, scale="signed", match="surface_then_suffix"
    )
    result = {
        "6_per_role_by_manifest": per_role(arguments.runs, lexicon, round_index=arguments.round),
        "7_discriminant": discriminant(
            arguments.runs, round_index=arguments.round, drop_echo=False
        ),
        "7_discriminant_echo_dropped": discriminant(
            arguments.runs, round_index=arguments.round, drop_echo=True
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for role, row in result["6_per_role_by_manifest"].items():
        lo, hi = row["interval"]
        positive = f"{row['positive']}/{row['pairs']}"
        print(f"{role:20s} {row['value']:+.4f} [{lo:+.4f}, {hi:+.4f}]  {positive}")
    for key in ("7_discriminant", "7_discriminant_echo_dropped"):
        row = result[key]
        lo, hi = row["interval_per_teammate"]
        print(f"{key:28s} {row['per_teammate']:+.4f} [{lo:+.4f}, {hi:+.4f}]  pairs {row['pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
