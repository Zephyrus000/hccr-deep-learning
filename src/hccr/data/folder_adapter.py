"""Adapter for character-folder image exports such as the current CASIA data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FolderSample:
    path: Path
    unicode_label: str
    sample_id: str


def iter_folder_samples(root: Path) -> Iterator[FolderSample]:
    """Yield deterministic samples from ``<label>/<image>`` directories."""
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if len(label_dir.name) != 1:
            raise ValueError(
                f"label directory is not one Unicode character: {label_dir}"
            )
        for path in sorted(child for child in label_dir.iterdir() if child.is_file()):
            relative = path.relative_to(root).as_posix()
            yield FolderSample(
                path=path,
                unicode_label=label_dir.name,
                sample_id=hashlib.sha256(relative.encode()).hexdigest(),
            )
