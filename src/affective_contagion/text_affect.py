"""Score how much affect is in a message with a lexical VAD instrument.

This module measures; it creates no stimulus, which `payload.py` does. Every message an agent
produces is scored here, and the resulting arousal series is the primary language-level outcome.

Coverage is reported per message because an out-of-vocabulary message scores the neutral default.
The instrument is a bag of words with no word order. Negation does not invert or modify a word's
rating: in "not panicking", "panicking" keeps its own score, and "not" counts only through its own
rating if the lexicon lists it.

`validate_affect_separation` stops a run whose manipulation this instrument cannot detect, so a
null result cannot come from stimulus wording the instrument does not register.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

TokenScore = tuple[float, float, float]
TOKEN_PATTERN = re.compile(r"[a-z']+")
VADScale = Literal["unit", "signed"]
MatchMode = Literal["surface", "surface_then_suffix"]

# Fallback suffix rules, tried in order and only when the surface form is absent. This is not a
# lemmatizer. NRC VAD carries inflections as separate entries with different scores, so
# normalising every form to one lemma would replace a listed value with a wrong one. The surface
# form therefore always wins.
_SUFFIX_RULES = (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ing", ""), ("ly", ""))
_MIN_STEM = 3


@dataclass(frozen=True, slots=True)
class VADScore:
    """Mean lexical VAD values and observed-token coverage."""

    valence: float
    arousal: float
    dominance: float
    coverage: float


class VADLexicon:
    """Immutable lexical scorer with explicit out-of-vocabulary behavior."""

    def __init__(self, entries: dict[str, TokenScore], *, match: MatchMode = "surface") -> None:
        """Validate and copy lexical entries."""
        self._match = match
        if not entries:
            msg = "VAD lexicon must not be empty"
            raise ValueError(msg)
        self._entries = {key.lower(): value for key, value in entries.items()}
        if any(
            len(value) != 3 or any(not 0 <= item <= 1 for item in value)
            for value in entries.values()
        ):
            msg = (
                "VAD entries must contain three values in [0, 1]; "
                'NRC VAD v2.x ships on [-1, 1], so load it with scale="signed"'
            )
            raise ValueError(msg)

    @classmethod
    def from_tsv(
        cls, path: Path, *, scale: VADScale = "unit", match: MatchMode = "surface"
    ) -> VADLexicon:
        """Load NRC-style term, valence, arousal, dominance tab-separated rows.

        ``scale`` declares the source range: ``unit`` for [0, 1] as in NRC VAD v1.0, or ``signed``
        for [-1, 1] as in NRC VAD v2.x, which is rescaled to [0, 1]. The range is declared rather
        than inferred, because a lexicon whose entries happen to be non-negative would be silently
        misread and shift every downstream arousal value by half the range.

        Rows without exactly four tab-separated fields, or with unparseable numbers, are skipped.
        That tolerates the header row the NRC distributions ship.
        """
        if scale not in {"unit", "signed"}:
            msg = f"unknown VAD scale: {scale!r}"
            raise ValueError(msg)
        entries: dict[str, TokenScore] = {}
        saw_negative = False
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            try:
                values = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            if scale == "signed":
                saw_negative = saw_negative or any(value < 0 for value in values)
                values = ((values[0] + 1) / 2, (values[1] + 1) / 2, (values[2] + 1) / 2)
            entries[parts[0]] = values
        if scale == "signed" and not saw_negative:
            # A [0, 1] file read as signed maps silently into [0.5, 1], halving every contrast and
            # leaving numbers that still look plausible. Fail instead.
            msg = (
                "no negative value found, so this lexicon is not on [-1, 1]; "
                'load it with scale="unit"'
            )
            raise ValueError(msg)
        return cls(entries, match=match)

    def _lookup(self, word: str) -> TokenScore | None:
        """Resolve one token, preferring its exact surface form."""
        entry = self._entries.get(word)
        if entry is not None:
            return entry
        if self._match == "surface":
            return None
        # Possessives survive tokenisation as "bank's" and never match.
        candidates = [word[:-2]] if word.endswith("'s") else []
        candidates += [
            word[: len(word) - len(suffix)] + replacement
            for suffix, replacement in _SUFFIX_RULES
            if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM
        ]
        for candidate in candidates:
            entry = self._entries.get(candidate)
            if entry is not None:
                return entry
        return None

    def score(self, text: str) -> VADScore:
        """Score known words; use neutral values and zero coverage if none are known."""
        words = TOKEN_PATTERN.findall(text.lower())
        resolved = [self._lookup(word) for word in words]
        hits = [entry for entry in resolved if entry is not None]
        if not hits:
            return VADScore(0.5, 0.5, 0.5, 0.0)
        values = np.asarray(hits, dtype=np.float64)
        return VADScore(
            valence=float(values[:, 0].mean()),
            arousal=float(values[:, 1].mean()),
            dominance=float(values[:, 2].mean()),
            coverage=len(hits) / max(len(words), 1),
        )


DEFAULT_MIN_AROUSAL_DELTA = 0.02


def affect_separation(lexicon: VADLexicon, neutral: str, anxiety_like: str) -> float:
    """Return how much more aroused the anxiety-like arm scores than the neutral arm."""
    return lexicon.score(anxiety_like).arousal - lexicon.score(neutral).arousal


def validate_affect_separation(
    lexicon: VADLexicon,
    arms: Mapping[str, str],
    *,
    min_arousal_delta: float = DEFAULT_MIN_AROUSAL_DELTA,
) -> float:
    """Require the anxiety-like arm to score measurably more aroused, or fail the run.

    Payload matching guarantees the arms carry the same financial content. It says nothing about
    whether the affective manipulation changes the measured affect, so both checks are needed.
    Inflected words absent from the lexicon and low-arousal words can make the anxiety-like arm
    score below neutral, producing a null result caused by wording rather than agent behaviour.

    The instrument is a bag of words with no lemmatisation, negation handling, or word order, so
    passing this check confirms the intended direction is measurable. It does not show that the
    manipulation is valid, or that it is distinct from confounds such as negative sentiment or
    uncertainty.
    """
    required = ("neutral", "anxiety_like")
    missing = [name for name in required if name not in arms]
    if missing:
        msg = f"affect separation needs both arms; missing {missing}"
        raise ValueError(msg)
    delta = affect_separation(lexicon, arms["neutral"], arms["anxiety_like"])
    if delta < min_arousal_delta:
        msg = (
            f"anxiety-like arm scores {delta:+.4f} arousal against neutral, below the required "
            f"{min_arousal_delta:+.4f}: the manipulation is not measurable by this instrument"
        )
        raise ValueError(msg)
    return delta
