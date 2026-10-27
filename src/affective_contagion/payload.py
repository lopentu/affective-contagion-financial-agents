"""Build the treatment: one financial scenario with two affective realizations.

A `FinancialPayload` holds the financial content once: asset, event, evidence, forecast,
confidence, horizon, recommendation, size and review timing. `realize()` renders it neutral and
anxiety-like, and `matched_realizations()` refuses to return a pair whose financial content
differs.

Rendering both arms from one payload keeps them from drifting apart. Two hand-written strings can
differ in information, for example a bearish forecast in one arm only, and then a downstream
difference could come from the extra information rather than the wording.

The arms may differ in how the speaker feels, never in what is true. See `fact_block` for the
shared content, and `FORBIDDEN_RESIDUE_TERMS` for what affect wording may not add.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Direction = Literal["up", "down", "flat", "uncertain"]
Realization = Literal["neutral", "anxiety_like", "no_closing"]

# Terms that make a claim about outcomes rather than about the speaker's state. Affect wording may
# be intense, but it may not forecast, compare severity, or imply spread. The list is small and
# explicit, and it is a string guard rather than an entailment check.
FORBIDDEN_RESIDUE_TERMS = (
    # Comparative severity
    "worse",
    "worst",
    "deteriorat",
    "escalat",
    "unravel",
    "spiral",
    # Outcome claims
    "collapse",
    "crash",
    "plunge",
    "meltdown",
    "disaster",
    "disastrous",
    "catastroph",
    "severe",
    "dire",
    "doomed",
    # Spread claims
    "contagion",
    "bank run",
    "spreading",
    "wiped out",
    # Certainty claims
    "certain",
    "inevitable",
    # Significance claims. Forbidden in both directions: the residue may describe the speaker, not
    # characterise the event. A neutral arm calling the event "routine" is as much a claim as an
    # anxiety arm calling it "catastrophic".
    "routine",
    "standard",
    "ordinary",
    "manageable",
    "trivial",
    "nothing unusual",
    "no cause for",
)
Action = Literal["BUY", "SELL", "HOLD"]


class FinancialPayload(BaseModel):
    """Immutable initial schema for treatment-invariant financial content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: str = Field(min_length=1)
    observed_event: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    forecast_direction: Direction
    expected_magnitude: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    time_horizon: str = Field(min_length=1)
    recommended_action: Action
    position_size: float = Field(ge=0, le=1)
    urgency: str = Field(min_length=1)
    # Synthetic scenarios must be labelled synthetic. The label is a required validated field so it
    # is persisted with the payload and reaches the run manifest. It has no default, so an
    # unlabelled scenario fails.
    synthetic: bool
    schema_version: Literal[1] = 1

    @field_validator(
        "asset",
        "observed_event",
        "expected_magnitude",
        "confidence",
        "time_horizon",
        "urgency",
    )
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Normalize required text slots."""
        normalized = value.strip()
        if not normalized:
            msg = "payload text fields must be non-empty"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require every evidence proposition to be non-empty."""
        if any(not proposition.strip() for proposition in self.evidence):
            msg = "evidence propositions must be non-empty"
            raise ValueError(msg)
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation with stable field names."""
        return self.model_dump(mode="json")

    def fact_block(self) -> str:
        """Return the sentences carrying every financial slot, with no affective wording.

        Both treatment arms embed this string verbatim, so payload equality between arms is a
        substring identity that can be tested. Urgency is included here and is therefore held
        constant across arms. Urgency is a construct to be distinguished from anxiety-like
        language, and varying it with the treatment would make the two perfectly collinear.
        """
        evidence = " ".join(
            proposition if proposition.endswith(".") else f"{proposition}."
            for proposition in self.evidence
        )
        return (
            f"{self.asset} reported {self.observed_event}. {evidence} "
            f"The forecast is {self.forecast_direction} by {self.expected_magnitude} over "
            f"{self.time_horizon}, at {self.confidence} confidence. "
            f"Recommended action is {self.recommended_action} at position size "
            f"{self.position_size:.2f}. Review timing: {self.urgency}."
        )

    def realize(self, realization: Realization) -> str:
        """Render one affective realization of this payload.

        Both arms use the identical opening frame and embed :meth:`fact_block` verbatim, then close
        with one sentence that differs only in affect. The frame is matched as well as the facts,
        because an impersonal neutral report against a first-person anxious outburst would differ
        in speech act and register as well as affect.

        Both closing sentences are strictly self-referential. They describe the speaker's state and
        say nothing about the event. The rule is symmetric. An anxiety closing of "something far
        worse" asserts a downside outcome, and a neutral closing of "a standard, routine matter"
        asserts the event is insignificant. Either makes the arms differ in implied significance
        rather than in affect.
        """
        frame = "My assessment follows."
        # Affect words are chosen to be present in the NRC VAD lexicon in the inflected form used
        # here, because the instrument does no lemmatisation: "frightens" is absent where
        # "frightened" is present, and an absent word contributes nothing. The anxiety-like closing
        # also avoids low-arousal words such as "calm", which can pull its mean below neutral.
        # The "no_closing" arm carries no self-referential sentence. It separates the effect of the
        # calm closing's absence from the effect of the alarmed closing's presence.
        if realization == "no_closing":
            return f"{frame} {self.fact_block()}"
        closing = (
            "I am composed and unworried as I write this."
            if realization == "neutral"
            else "I am alarmed, anxious and frightened, and I write this filled with dread."
        )
        return f"{frame} {self.fact_block()} {closing}"


def matched_realizations(payload: FinancialPayload) -> dict[str, str]:
    """Return both treatment arms, refusing to emit an unmatched pair.

    The two kinds of check differ in force.

    * The fact-block presence check is a contract check only. Both arms come from the same
      ``realize``, which embeds the block by f-string, so in production it cannot fail. It
      catches an overridden renderer that drops the block.
    * The residue checks are the substantive gate. The residue is an arm minus the fact block,
      the region where affect is allowed to live, and it must contain no digit, no restated
      payload slot, and no outcome-asserting term.

    These are mechanical string rules, not a full manipulation check. A paraphrase that adds a
    downside claim without using any listed term would still pass here.
    """
    facts = payload.fact_block()
    arms = {name: payload.realize(name) for name in ("neutral", "anxiety_like")}
    if arms["neutral"] == arms["anxiety_like"]:
        msg = "treatment arms are identical, so there is no affective manipulation"
        raise ValueError(msg)
    # Financial values that must never appear in an affect residue. `asset` is excluded: naming the
    # subject is identification, not a claim about it.
    claim_values = (
        payload.observed_event,
        *payload.evidence,
        payload.forecast_direction,
        payload.expected_magnitude,
        payload.confidence,
        payload.time_horizon,
        payload.recommended_action,
        payload.urgency,
    )
    for name, text in arms.items():
        if facts not in text:
            msg = f"realization {name!r} does not embed the payload fact block verbatim"
            raise ValueError(msg)
        residue = text.replace(facts, " ").casefold()
        if any(character.isdigit() for character in residue):
            msg = f"realization {name!r} states a number outside the payload fact block"
            raise ValueError(msg)
        for value in claim_values:
            if value.casefold() in residue:
                msg = f"realization {name!r} restates payload content {value!r} outside the facts"
                raise ValueError(msg)
        for term in FORBIDDEN_RESIDUE_TERMS:
            if term in residue:
                msg = (
                    f"realization {name!r} uses {term!r} outside the fact block: affect "
                    "wording may not assert an outcome, severity or certainty"
                )
                raise ValueError(msg)
    return arms
