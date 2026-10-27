"""Tests for model-specific residual-stream affect measurement."""

import json
from pathlib import Path

import numpy as np
import pytest

from affective_contagion.activation import AffectDirection, load_affect_direction


def test_direction_validates_projection_shape_and_freezes_vector() -> None:
    """Representation spaces must not silently mix dimensions."""
    direction = AffectDirection("model", "org/repo", "abc", 3, np.array([3.0, 4.0]))
    assert direction.project(np.array([3.0, 4.0])) == pytest.approx(5.0)
    assert not direction.vector.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        direction.project(np.ones(3))


def test_direction_round_trip_preserves_representation_metadata(tmp_path: Path) -> None:
    """Stored directions must remain inseparable from checkpoint and layer identity."""
    path = tmp_path / "direction.npz"
    expected = AffectDirection("model", "org/repo", "abc", 3, np.array([1.0, 2.0]))
    metadata = {"schema_version": 1, "model_key": "model", "repository": "org/repo"}
    metadata |= {"revision": "abc", "layer": 3}
    np.savez_compressed(path, vector=expected.vector, metadata=np.asarray(json.dumps(metadata)))
    actual = load_affect_direction(path)
    assert (actual.model_key, actual.repository, actual.revision, actual.layer) == (
        "model",
        "org/repo",
        "abc",
        3,
    )
    assert np.allclose(actual.vector, expected.vector)
