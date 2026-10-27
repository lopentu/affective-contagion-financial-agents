"""Projection of a captured residual-stream activation onto a model-specific affect direction.

The simulation can record this projection when a roster sets `activation_layer` and a direction
file is supplied. The paper's runs did neither, so their activation series are all NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AffectDirection:
    """Unit affect direction tied to one exact model representation space."""

    model_key: str
    repository: str
    revision: str | None
    layer: int
    vector: FloatVector

    def __post_init__(self) -> None:
        """Copy, validate, normalize, and freeze the numerical vector."""
        if not self.model_key or not self.repository or self.layer < 0:
            msg = "direction metadata is invalid"
            raise ValueError(msg)
        values = np.asarray(self.vector, dtype=np.float64).copy()
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            msg = "direction vector must be one-dimensional, non-empty, and finite"
            raise ValueError(msg)
        magnitude = float(np.linalg.norm(values, ord=2))
        if magnitude == 0:
            msg = "direction vector must be non-zero"
            raise ValueError(msg)
        values /= magnitude
        values.setflags(write=False)
        object.__setattr__(self, "vector", values)

    def project(self, residual_stream: FloatVector) -> float:
        """Project one residual-stream activation onto this unit direction."""
        activation = np.asarray(residual_stream, dtype=np.float64)
        if activation.shape != self.vector.shape:
            msg = (
                f"activation shape {activation.shape} does not match direction "
                f"shape {self.vector.shape}"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(activation)):
            msg = "activation must contain only finite values"
            raise ValueError(msg)
        return float(activation @ self.vector)


def load_affect_direction(path: Path) -> AffectDirection:
    """Load a direction without allowing pickled object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        vector = np.asarray(archive["vector"], dtype=np.float64)
    if metadata.pop("schema_version", None) != 1:
        msg = "unsupported affect-direction schema"
        raise ValueError(msg)
    return AffectDirection(vector=vector, **metadata)
