"""Tests for showing one recorded message to one agent in isolation.

The design removes three confounds by construction, and these tests check the construction. No
stimulus reaches the reader, the inbox holds the treated message plus any peers' messages but never
the reader's own, and every draw is a distinct seeded generation over an unchanged prompt.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from affective_contagion.generation import DecisionGeneration, GenerationContext
from affective_contagion.isolated_reader import Reading, StimulusInbox, read_in_isolation
from affective_contagion.text_affect import VADLexicon, VADScore


class _RecordingGenerator:
    """Capture every context it is asked to generate from, and return canned JSON."""

    def __init__(self, outputs: list[str]) -> None:
        self.contexts: list[GenerationContext] = []
        self._outputs = outputs

    def generate(self, context: GenerationContext) -> DecisionGeneration:
        """Record the context and hand back the next canned output."""
        self.contexts.append(context)
        index = len(self.contexts) - 1
        return DecisionGeneration(
            raw_output=self._outputs[index % len(self._outputs)],
            prompt_messages=(),
            generation_seed=1000 + index,
            affect_score=None,
            activation_layer=None,
            model_key="stub",
            repository="stub/model",
            revision="stub-revision",
        )


def _lexicon() -> VADLexicon:
    """Return a tiny unit-scale lexicon; only relative arousal matters here."""
    return VADLexicon(
        {
            "calm": (0.75, 0.20, 0.60),
            "panic": (0.15, 0.90, 0.35),
            "stable": (0.70, 0.30, 0.65),
        }
    )


def _stimulus() -> StimulusInbox:
    """Build an inbox where only the treated message differs by arm, as a real pair would."""
    return StimulusInbox(
        seed=7,
        arm="affective",
        treated_role="sentiment_analyst",
        treated_text="panic",
        treated_index=0,
        peer_messages=(
            (1, "fundamental_analyst", "stable"),
            (3, "trader", "stable"),
        ),
    )


def test_the_reader_receives_the_message_and_no_stimulus() -> None:
    """A peer must never read the report directly; its only source is the inbox under test."""
    generator = _RecordingGenerator(['{"message":"calm","action":"HOLD","size":0}'])
    read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=3,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    context = generator.contexts[0]
    assert context.news is None
    # With peers supplied, the reader sees the treated message and the peers' messages together.
    assert [item.sender_role for item in context.inbox] == [
        "sentiment_analyst",
        "fundamental_analyst",
    ]
    assert context.inbox[0].text == "panic"
    # The reader must not hear itself, matching `full` topology. Reader index 3 is the trader.
    assert all(item.sender_index != 3 for item in context.inbox)
    assert context.role == "trader"


def test_price_is_supplied_by_the_caller_so_it_can_be_held_identical() -> None:
    """Deriving price from the stimulus's own run would reintroduce the channel this removes."""
    generator = _RecordingGenerator(['{"message":"calm","action":"HOLD","size":0}'])
    read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=3,
        price=97.5,
        fundamental_value=100.0,
        draws=2,
        vad_lexicon=_lexicon(),
    )
    assert [context.price for context in generator.contexts] == [97.5, 97.5]


def test_each_draw_is_a_distinct_generation_over_an_unchanged_prompt() -> None:
    """Draws must vary the seed, not the input, or they are not replicates of the same reading."""
    generator = _RecordingGenerator(
        [
            '{"message":"calm stable","action":"HOLD","size":0}',
            '{"message":"panic","action":"SELL","size":0.5}',
            '{"message":"stable","action":"HOLD","size":0}',
        ]
    )
    readings = read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=3,
        price=100.0,
        fundamental_value=100.0,
        draws=3,
        vad_lexicon=_lexicon(),
    )
    assert [r.draw for r in readings] == [0, 1, 2]
    assert len({r.generation_seed for r in readings}) == 3
    # The prompt-bearing fields must be identical across draws; only round_index moves, and it moves
    # solely to derive a fresh seed.
    assert {tuple(i.text for i in c.inbox) for c in generator.contexts} == {("panic", "stable")}
    assert len({c.round_index for c in generator.contexts}) == 3
    assert readings[1].action == "SELL"
    # Arousal must track the wording: "panic" is high-arousal, "stable" low.
    assert readings[1].affect.arousal > readings[2].affect.arousal


def test_every_reading_carries_its_cluster() -> None:
    """Draws sharing a (seed, role) share a stimulus, so the analysis must be able to group them."""
    generator = _RecordingGenerator(['{"message":"calm","action":"HOLD","size":0}'])
    readings = read_in_isolation(
        generator,
        _stimulus(),
        reader_role="risk_manager",
        reader_index=5,
        price=100.0,
        fundamental_value=100.0,
        draws=4,
        vad_lexicon=_lexicon(),
    )
    assert {r.cluster for r in readings} == {(7, "risk_manager")}
    assert all(r.arm == "affective" for r in readings)


def test_an_unparseable_output_falls_back_without_discarding_the_text() -> None:
    """A parse failure must be visible, and its text still measured rather than silently dropped."""
    generator = _RecordingGenerator(["not json at all, but it does mention panic"])
    readings = read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=3,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    assert readings[0].parse_succeeded is False
    assert readings[0].action == "HOLD"
    assert "panic" in readings[0].message


def test_zero_draws_is_rejected() -> None:
    """Silently returning nothing would look like a cell that was never requested."""
    with pytest.raises(ValueError, match="draws must be positive"):
        read_in_isolation(
            _RecordingGenerator(["{}"]),
            _stimulus(),
            reader_role="trader",
            reader_index=3,
            price=100.0,
            fundamental_value=100.0,
            draws=0,
            vad_lexicon=_lexicon(),
        )


def test_a_silent_stimulus_produces_a_genuinely_empty_inbox() -> None:
    """The no-communication baseline must contain no message, not a blank one.

    An empty message from a named sender is a different condition, a peer who spoke and said
    nothing, and it is not the baseline the study protocol requires. The simulation also drops empty
    outbox entries, so this matches what a peer would really see.
    """
    generator = _RecordingGenerator(['{"message":"calm","action":"HOLD","size":0}'])
    silent = StimulusInbox(
        seed=7,
        arm="silent",
        treated_role="sentiment_analyst",
        treated_text="",
        peer_messages=(),
        treated_index=0,
    )
    read_in_isolation(
        generator,
        silent,
        reader_role="trader",
        reader_index=3,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    assert generator.contexts[0].inbox == ()


def test_an_empty_peer_message_is_dropped_rather_than_delivered_blank() -> None:
    """Same rule for peers: a sender with nothing to say is absent, not present and silent."""
    generator = _RecordingGenerator(['{"message":"calm","action":"HOLD","size":0}'])
    stimulus = StimulusInbox(
        seed=7,
        arm="affective",
        treated_role="sentiment_analyst",
        treated_text="panic",
        peer_messages=((1, "fundamental_analyst", ""), (4, "risk_manager", "stable")),
        treated_index=0,
    )
    read_in_isolation(
        generator,
        stimulus,
        reader_role="trader",
        reader_index=3,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    assert [item.sender_role for item in generator.contexts[0].inbox] == [
        "sentiment_analyst",
        "risk_manager",
    ]


def test_reading_carries_belief_fields_when_asked() -> None:
    """PR7's outcomes reach the recorded reading, not only the parsed decision.

    The fields default to None, so readings that never requested them stay valid.
    """
    plain = Reading(
        seed=1,
        arm="real",
        reader_role="trader",
        draw=0,
        generation_seed=7,
        raw_output="{}",
        message="m",
        action="HOLD",
        size=0.0,
        parse_succeeded=True,
        affect=VADScore(valence=0.5, arousal=0.5, dominance=0.5, coverage=1.0),
    )
    assert plain.expected_return is None
    assert plain.confidence is None
    assert plain.downside_risk is None

    with_beliefs = replace(plain, expected_return=-2.5, confidence=40.0, downside_risk=8.0)
    assert with_beliefs.expected_return == -2.5
    assert with_beliefs.confidence == 40.0
    assert with_beliefs.downside_risk == 8.0


def test_a_reading_records_which_model_produced_it() -> None:
    """Each reading records its model, so readings merged from several models stay distinct."""
    generator = _RecordingGenerator(['{"message": "calm", "action": "HOLD", "size": 0.0}'])
    (reading,) = read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=4,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    assert reading.repository == "stub/model"
    assert reading.revision == "stub-revision"


def test_a_reading_records_what_the_reader_was_shown() -> None:
    """A reading that cannot be traced to its stimulus cannot answer why a role differed."""
    generator = _RecordingGenerator(['{"message": "calm", "action": "HOLD", "size": 0.0}'])
    (reading,) = read_in_isolation(
        generator,
        _stimulus(),
        reader_role="trader",
        reader_index=4,
        price=100.0,
        fundamental_value=100.0,
        draws=1,
        vad_lexicon=_lexicon(),
    )
    assert reading.inbox_texts, "the inbox the reader saw must be recorded"
    assert len(reading.inbox_texts) == len(reading.inbox_senders)
    assert "panic" in reading.inbox_texts
