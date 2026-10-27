"""Cross-platform runtime and source provenance collection."""

from __future__ import annotations

import hashlib
import importlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Environment details needed to interpret or reproduce a run."""

    git_commit: str | None
    git_dirty: bool | None
    uv_lock_sha256: str
    python_version: str
    operating_system: str
    accelerator: str | None
    torch_version: str | None
    transformers_version: str | None


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_runtime_provenance(repository_root: Path, lock_file: Path) -> RuntimeProvenance:
    """Collect Git, lock, Python, OS, and available accelerator identifiers."""
    lock_hash = hashlib.sha256(lock_file.read_bytes()).hexdigest()
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    status = _git_output(repository_root, "status", "--porcelain")
    accelerator: str | None = None
    torch_version: str | None = None
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        pass
    else:
        torch_version = str(torch.__version__)
        if torch.cuda.is_available():
            accelerator = str(torch.cuda.get_device_name(0))
    # The tokenizer and generation call conventions moved between Transformers major versions, so
    # the library version is needed to interpret a run, not only the torch version.
    transformers_version: str | None = None
    try:
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError:
        pass
    else:
        transformers_version = str(transformers.__version__)
    return RuntimeProvenance(
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
        uv_lock_sha256=lock_hash,
        python_version=sys.version,
        operating_system=platform.platform(),
        accelerator=accelerator,
        torch_version=torch_version,
        transformers_version=transformers_version,
    )
