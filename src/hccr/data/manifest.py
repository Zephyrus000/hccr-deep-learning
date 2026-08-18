"""CSV manifest I/O and invariant checks independent of model code."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {
    "sample_id",
    "source_file",
    "writer_id",
    "unicode_label",
    "class_id",
    "split",
}
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestAudit:
    sample_count: int
    class_count: int
    split_counts: dict[str, int]
    writer_overlap_count: int


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"manifest is missing required columns: {path}")
        return list(reader)


def audit_manifest(rows: list[dict[str, str]]) -> ManifestAudit:
    sample_ids: set[str] = set()
    source_files: set[str] = set()
    class_labels: dict[str, str] = {}
    writer_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        sample_id = row["sample_id"]
        source_file = row["source_file"]
        split = row["split"]
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id}")
        if not source_file or source_file in source_files:
            raise ValueError(f"duplicate or empty source_file: {source_file}")
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {split}")
        label = row["unicode_label"]
        class_id = row["class_id"]
        if class_labels.setdefault(class_id, label) != label:
            raise ValueError(f"class_id maps to multiple labels: {class_id}")
        writer = row["writer_id"].strip()
        if writer:
            writer_splits[writer].add(split)
        sample_ids.add(sample_id)
        source_files.add(source_file)
        split_counts[split] += 1

    overlaps = sum(len(splits) > 1 for splits in writer_splits.values())
    return ManifestAudit(len(rows), len(class_labels), dict(split_counts), overlaps)
