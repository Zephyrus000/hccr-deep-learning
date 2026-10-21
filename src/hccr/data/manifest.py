"""CSV manifest I/O and invariant checks independent of model code."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

REQUIRED_COLUMNS = {
    "sample_id",
    "source_file",
    "writer_id",
    "unicode_label",
    "class_id",
    "split",
}
VALID_SPLITS = {"train", "validation", "test", "icdar"}


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
    source_files: set[tuple[str, int | None]] = set()
    class_labels: dict[str, str] = {}
    writer_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        sample_id = row["sample_id"]
        source_file = row["source_file"]
        split = row["split"]
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id}")
        offset = row.get("record_offset", "")
        record_offset = int(offset) if offset else None
        if record_offset is not None and record_offset < 0:
            raise ValueError("record_offset must be non-negative")
        if Path(source_file).suffix.lower() == ".gnt" and record_offset is None:
            raise ValueError(f"GNT source requires record_offset: {source_file}")
        source_key = (
            source_file,
            record_offset if Path(source_file).suffix.lower() == ".gnt" else None,
        )
        if not source_file or source_key in source_files:
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
        source_files.add(source_key)
        split_counts[split] += 1

    overlaps = sum(len(splits) > 1 for splits in writer_splits.values())
    return ManifestAudit(len(rows), len(class_labels), dict(split_counts), overlaps)


def writer_provenance(rows: list[dict[str, str]]) -> dict[str, object]:
    """Audit writer IDs for all split pairs without inferring human identity."""
    known = [row for row in rows if row["writer_id"].strip()]
    if not known:
        availability = "unavailable"
    elif len(known) == len(rows):
        availability = "complete"
    else:
        availability = "partial"
    development_writers = {
        row["writer_id"].strip()
        for row in known
        if row["split"] in {"train", "validation"}
    }
    test_writers = {row["writer_id"].strip() for row in known if row["split"] == "test"}
    overlap_count = len(development_writers.intersection(test_writers))
    writers_by_split = {
        split: {row["writer_id"].strip() for row in known if row["split"] == split}
        for split in ("train", "validation", "test", "icdar")
    }
    pairwise = {
        f"{left}/{right}": len(writers_by_split[left] & writers_by_split[right])
        for left, right in combinations(writers_by_split, 2)
    }
    any_overlap = any(pairwise.values())
    has_both_partitions = bool(development_writers and test_writers)
    verified = (
        False
        if any_overlap
        else True if availability == "complete" and has_both_partitions else None
    )
    return {
        "availability": availability,
        "available_splits": sorted({row["split"] for row in rows}),
        "overlap_check": (
            "failed"
            if any_overlap
            else "passed" if verified is True else "not_verifiable"
        ),
        "writer_disjoint_verified": verified,
        "known_writer_samples": len(known),
        "missing_writer_samples": len(rows) - len(known),
        "train_validation_test_overlap_count": overlap_count,
        "pairwise_writer_overlap_counts": pairwise,
        "verification_basis": "manifest_writer_identifiers_not_biometric_identity",
    }


def validate_training_manifest(
    path: Path, rows: list[dict[str, str]]
) -> dict[str, object]:
    """Fail before training on known writer leakage or a modified audited split."""
    provenance = writer_provenance(rows)
    if any(provenance["pairwise_writer_overlap_counts"].values()):
        raise ValueError(
            "writer leakage between manifest splits: "
            f"{provenance['pairwise_writer_overlap_counts']}"
        )
    from hccr.data.casia_manifest import gnt_writer_id

    gnt_sources = {}
    for row in rows:
        source_file = row["source_file"]
        if source_file.lower().endswith(".gnt"):
            if source_file not in gnt_sources:
                gnt_sources[source_file] = gnt_writer_id(Path(source_file))
            if row["writer_id"] != gnt_sources[source_file]:
                raise ValueError(
                    f"GNT writer_id disagrees with source filename: {source_file}"
                )
    report_path = path.with_name("audit_report.json")
    has_competition = any(
        writer.startswith("icdar2013:") for writer in gnt_sources.values()
    )
    if has_competition and not report_path.is_file():
        raise ValueError(
            "ICDAR training requires an audited manifest with audit_report.json"
        )
    if report_path.is_file():
        from hccr.data.lmdb_store import manifest_digest

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("source_kind") != "gnt":
            if has_competition:
                raise ValueError("ICDAR data requires a GNT protocol audit report")
            return provenance
        if report.get("manifest_digest") != manifest_digest(path):
            raise ValueError(
                "manifest changed after dataset audit; rebuild the dataset"
            )
        mapping_digest = report.get("label_mapping_digest")
        if (
            mapping_digest
            and manifest_digest(path.with_name("class_mapping_full.json"))
            != mapping_digest
        ):
            raise ValueError("class mapping changed after dataset audit")
        if has_competition:
            if (
                report.get("holdout_policy")
                != "casia_test_and_icdar_separate_never_used_for_selection"
            ):
                raise ValueError(
                    "rebuild ICDAR manifest with separate CASIA test and icdar splits"
                )
            if report.get("image_overlap_check", {}).get("status") != "passed":
                raise ValueError("ICDAR exact-image leakage audit has not passed")
            if report["image_overlap_check"].get("checked_samples") != len(rows):
                raise ValueError(
                    "ICDAR image audit does not cover every manifest sample"
                )
            if any(len(row.get("pixel_sha256", "")) != 64 for row in rows):
                raise ValueError("ICDAR manifest is missing audited pixel fingerprints")
            for row in rows:
                competition = row["writer_id"].startswith("icdar2013:")
                if competition != (row["split"] == "icdar"):
                    raise ValueError("ICDAR competition records must be icdar-only")
        provenance["dataset_protocol"] = report.get("protocol")
        provenance["image_overlap_check"] = report.get(
            "image_overlap_check", {"status": "not_run"}
        )
        provenance["manifest_digest"] = report["manifest_digest"]
        provenance["label_mapping_digest"] = mapping_digest
    return provenance
