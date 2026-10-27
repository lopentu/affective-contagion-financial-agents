"""External YAML configuration loading at validated Pydantic boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

Schema = TypeVar("Schema", bound=BaseModel)


def load_yaml_model(path: Path, schema: type[Schema]) -> Schema:
    """Load one YAML mapping and reject malformed or extra fields through its schema."""
    content: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        msg = f"{path} must contain a YAML mapping"
        raise TypeError(msg)
    return schema.model_validate(content)
