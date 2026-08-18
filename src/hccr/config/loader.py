"""Load YAML configuration files without coupling configuration to the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """Return a mapping from a YAML document, rejecting non-mapping roots."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML must be installed to load configuration files"
        ) from error

    with path.open("r", encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value
