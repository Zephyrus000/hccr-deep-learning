"""Disk-backed exact-image leakage checks for large dataset builds."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class ExactImageAudit:
    """Reject the same bitmap across splits, even under renamed writer files."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE images (hash TEXT PRIMARY KEY, split TEXT, location TEXT)"
        )
        self.checked_samples = 0
        self.same_split_duplicates = 0

    def add(self, pixel_hash: str, split: str, location: str) -> None:
        inserted = self.connection.execute(
            "INSERT OR IGNORE INTO images VALUES (?, ?, ?)",
            (pixel_hash, split, location),
        )
        self.checked_samples += 1
        if inserted.rowcount == 0:
            previous_split, previous_location = self.connection.execute(
                "SELECT split, location FROM images WHERE hash = ?", (pixel_hash,)
            ).fetchone()
            if previous_split != split:
                raise ValueError(
                    f"exact-image leakage between {previous_split} and {split}: "
                    f"{previous_location} and {location}; no samples were removed"
                )
            self.same_split_duplicates += 1

    def close(self) -> None:
        self.connection.close()
