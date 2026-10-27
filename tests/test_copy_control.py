"""Known-answer tests for the copying control built from each reader's own inbox."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from affective_contagion.copy_control import copy_control_by_inbox, input_words
from affective_contagion.text_affect import VADLexicon

RENDERINGS = {"neutral": "meridian steady", "anxiety_like": "meridian panic"}
PEERS = {"trader": "hedge", "risk_manager": "steady"}


def _lexicon() -> VADLexicon:
    return VADLexicon(
        {
            "panic": (0.1, 0.9, 0.2),
            "steady": (0.6, 0.2, 0.7),
            "tense": (0.2, 0.8, 0.3),
            "hedge": (0.5, 0.6, 0.5),
            "meridian": (0.5, 0.5, 0.5),
        }
    )


def test_bare_readers_see_only_the_renderings() -> None:
    """In the bare context the deleted set is the two renderings; peers never enter."""
    assert input_words(RENDERINGS, PEERS, condition="bare", reader_role="trader") == {
        "meridian",
        "steady",
        "panic",
    }


def test_embedded_readers_see_peers_but_not_themselves() -> None:
    """In the embedded context the other peers' words join the set; the reader's own do not."""
    got = input_words(RENDERINGS, PEERS, condition="embedded", reader_role="trader")
    assert "steady" in got
    assert "hedge" not in got, "the trader did not read its own message"
    assert "hedge" in input_words(
        RENDERINGS, PEERS, condition="embedded", reader_role="risk_manager"
    )


def _write_release(root: Path) -> Path:
    """Two roles, two seeds, bare context: one role copies, the other writes its own word."""
    readings = []
    for seed in (1, 2):
        for arm, copier, writer in (
            ("anxiety_like", "panic panic", "tense"),
            ("neutral", "steady steady", "steady"),
        ):
            for role, text in (("bull_researcher", copier), ("bear_researcher", writer)):
                readings.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "condition": "bare",
                        "reader_role": role,
                        "replicate": 1,
                        "draw": 0,
                        "message": text,
                    }
                )
    inboxes = [
        {
            "seed": seed,
            "arm": arm,
            "messages": [
                {"role": "sentiment_analyst", "decision_message": "x", "delivered_text": "x"},
                {"role": "trader", "decision_message": "hedge", "delivered_text": "hedge"},
            ],
        }
        for seed in (1, 2)
        for arm in ("neutral", "affective")
    ]
    with gzip.open(root / "r.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in readings)
    with gzip.open(root / "i.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in inboxes)
    manifest = root / "m.json"
    manifest.write_text(
        json.dumps({"readings": {"file": "r.jsonl.gz"}, "inboxes": {"file": "i.jsonl.gz"}}),
        encoding="utf-8",
    )
    return manifest


def test_copying_is_removed_and_own_words_are_kept(tmp_path: Path) -> None:
    """A reply made of inbox words goes to zero; a reply with its own word keeps its shift."""
    result = copy_control_by_inbox(_write_release(tmp_path), _lexicon(), RENDERINGS, draws=50)
    by_role = {r.role: r for r in result.roles}
    copier, writer = by_role["bull_researcher"], by_role["bear_researcher"]
    assert copier.published == pytest.approx(0.7, abs=1e-9)
    assert copier.controlled == pytest.approx(0.0, abs=1e-9), "both replies empty after deletion"
    assert writer.published == pytest.approx(0.6, abs=1e-9)
    # "tense" was not in the inbox and stays: 0.8 against an emptied calm reply scored 0.5.
    assert writer.controlled == pytest.approx(0.3, abs=1e-9)
    assert result.overlap_max == pytest.approx(1.0)
    assert 0.0 < result.permutation_p <= 1.0
