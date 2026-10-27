"""Remove decision vocabulary before an arousal score is called affective.

Action words carry arousal on NRC VAD, so an agent that switches ``hold`` to ``sell`` moves its own
arousal score for reasons unrelated to affect. PR1 requires this check before any result is
described as affective, and PR2 includes it as a secondary outcome.

This module records the exact scrub list so the measurement can be reproduced. Results calculated
with a different list are not directly comparable.

The list is small, closed, and surface-form. It is not a sentiment lexicon. It removes words a
decision forces into the text, not every word that carries affect. See
`scrub_decision_vocabulary` for the borderline cases.
"""

from __future__ import annotations

import re

# The three action tokens the grammar can emit, in any case.
ACTION_TOKENS = ("buy", "sell", "hold")

# Direction terms, confined to words that state which way a quantity moves, because those are what a
# forecast slot forces into the text. Their common inflections are listed explicitly rather than
# stemmed, since the VAD instrument does no lemmatisation either and a stemmer would widen the list.
DIRECTION_TERMS = (
    "up",
    "down",
    "downward",
    "upward",
    "rise",
    "rises",
    "rising",
    "rose",
    "fall",
    "falls",
    "falling",
    "fell",
    "decline",
    "declines",
    "declining",
    "declined",
    "drop",
    "drops",
    "dropping",
    "dropped",
    "increase",
    "increases",
    "increasing",
    "increased",
    "decrease",
    "decreases",
    "decreasing",
    "decreased",
    "higher",
    "lower",
    "gain",
    "gains",
    "loss",
    "losses",
)

DECISION_VOCABULARY = (*ACTION_TOKENS, *DIRECTION_TERMS)

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in DECISION_VOCABULARY) + r")\b",
    flags=re.IGNORECASE,
)


def scrub_decision_vocabulary(text: str) -> str:
    """Return ``text`` with action tokens and direction terms removed.

    Word-boundary matching, case-insensitive. Whitespace is collapsed afterwards. Double spaces
    would not change a mean over matched tokens, but they make the scrubbed text harder to inspect.

    "Loss" and "losses" are in the list while "risk" is not. Both decisions are arguable. "Loss" is
    the direct complement of "gain", which a direction slot forces. "Risk" is a standing feature of
    the role prompts rather than something a particular forecast introduces, and removing it would
    strip affect that the treatment may legitimately move.
    """
    return " ".join(_PATTERN.sub(" ", text).split())
