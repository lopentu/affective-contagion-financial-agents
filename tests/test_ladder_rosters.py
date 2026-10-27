"""The comparison ladder must differ only in the model.

If two rungs also differed in temperature, role prompt, or decoding, a difference in outcome could
come from that instead of the model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from affective_contagion.configuration import load_yaml_model
from affective_contagion.roster import ModelRoster

LADDER = Path("configs/models/ladder")
EXPECTED = {
    "qwen3_4b",
    "qwen35_2b",
    "qwen35_4b",
    "qwen35_9b",
    "gemma4_e2b",
    "gemma4_e4b",
    "granite41_3b",
    "granite41_8b",
}


def _rosters() -> dict[str, ModelRoster]:
    return {path.stem: load_yaml_model(path, ModelRoster) for path in sorted(LADDER.glob("*.yaml"))}


def test_every_rung_is_present() -> None:
    """Three families at two sizes each, plus the primary and its same-size newer sibling."""
    assert {path.stem for path in LADDER.glob("*.yaml")} == EXPECTED


def test_rungs_differ_only_in_the_model_block() -> None:
    """Strip comments and the model lines; exactly one distinct body must remain."""
    bodies = set()
    for path in sorted(LADDER.glob("*.yaml")):
        kept = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
            and not line.strip().startswith(("key:", "repository:", "revision:", "- key:"))
            and "model_key:" not in line
        ]
        bodies.add(hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest())
    assert len(bodies) == 1


def test_decoding_and_prompts_are_identical_across_rungs() -> None:
    """The settings that could otherwise explain a difference in outcome."""
    rosters = _rosters()
    specs = [next(iter(r.models_by_key.values())) for r in rosters.values()]
    assert {s.temperature for s in specs} == {0.7}
    assert {s.max_new_tokens for s in specs} == {256}
    assert {s.dtype for s in specs} == {"bfloat16"}
    assert {s.quantization for s in specs} == {"none"}
    assert {s.constrained_decision_format for s in specs} == {True}
    # Determinism controls must be enabled because broken pairing inflates the spread.
    assert {s.deterministic for s in specs} == {True}
    prompts = {tuple((a.role, a.system_prompt) for a in r.agents) for r in rosters.values()}
    assert len(prompts) == 1


def test_every_revision_is_pinned_to_a_commit() -> None:
    """Require exact revisions because a branch name can resolve to changing model contents."""
    for name, roster in _rosters().items():
        for spec in roster.models_by_key.values():
            # `revision` is optional in the schema, so an unpinned roster would load happily and
            # silently track a moving branch.
            revision = spec.revision
            assert revision is not None, f"{name} has no pinned revision"
            assert len(revision) == 40, f"{name} is not pinned to a full commit sha"
            assert set(revision) <= set("0123456789abcdef"), name


def test_no_rung_enables_remote_code() -> None:
    """Remote code would need auditing per checkpoint, and none of these architectures needs it."""
    for roster in _rosters().values():
        for spec in roster.models_by_key.values():
            assert spec.trust_remote_code is False


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_each_rung_loads(stem: str) -> None:
    """A roster that does not parse would fail hours into a batch instead of now."""
    roster = load_yaml_model(LADDER / f"{stem}.yaml", ModelRoster)
    assert len(roster.agents) == 6
    assert len(roster.models_by_key) == 1
