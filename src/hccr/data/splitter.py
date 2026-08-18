"""Deterministic writer-disjoint splitting for datasets that expose writers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable


class WriterDisjointSplitter:
    """Assign complete writer groups to validation without image-level leakage."""

    def __init__(self, validation_fraction: float = 0.1, seed: int = 20260817) -> None:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.validation_fraction = validation_fraction
        self.seed = seed

    def validation_writers(self, writer_ids: Iterable[str]) -> set[str]:
        writers = sorted({writer_id for writer_id in writer_ids if writer_id})
        if not writers:
            raise ValueError("writer-disjoint splitting requires non-empty writer IDs")
        count = max(1, math.floor(len(writers) * self.validation_fraction))
        ranked = sorted(
            writers,
            key=lambda writer: hashlib.sha256(
                f"{self.seed}\0{writer}".encode()
            ).digest(),
        )
        return set(ranked[:count])
