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
VALID_SPLITS = {"train", "calibration", "validation", "test", "icdar"}


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
        if row["split"] in {"train", "calibration", "validation"}
    }
    test_writers = {row["writer_id"].strip() for row in known if row["split"] == "test"}
    overlap_count = len(development_writers.intersection(test_writers))
    writers_by_split = {
        split: {row["writer_id"].strip() for row in known if row["split"] == split}
        for split in ("train", "calibration", "validation", "test", "icdar")
        if any(row["split"] == split for row in rows) or split != "calibration"
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
        else True
        if availability == "complete" and has_both_partitions
        else None
    )
    return {
        "availability": availability,
        "available_splits": sorted({row["split"] for row in rows}),
        "overlap_check": (
            "failed"
            if any_overlap
            else "passed"
            if verified is True
            else "not_verifiable"
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
    from hccr.data.casia_manifest import OFFICIAL_PROTOCOLS, gnt_writer_id

    report_path = path.with_name("audit_report.json")
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    claimed = {
        row.get("dataset_protocol", "") for row in rows
    } & OFFICIAL_PROTOCOLS.keys()
    if claimed and report.get("protocol") not in claimed:
        raise ValueError(
            "official proposal manifest requires its matching audit_report.json"
        )
    identity_mapping = report.get("writer_identity_mapping", {})

    gnt_sources = {}
    for row in rows:
        source_file = row["source_file"]
        if source_file.lower().endswith(".gnt"):
            if source_file not in gnt_sources:
                gnt_sources[source_file] = gnt_writer_id(Path(source_file))
            expected_writer = identity_mapping.get(
                gnt_sources[source_file], gnt_sources[source_file]
            )
            if row["writer_id"] != expected_writer:
                raise ValueError(
                    f"GNT writer_id disagrees with source filename: {source_file}"
                )
    has_competition = any(
        writer.startswith("icdar2013:") for writer in gnt_sources.values()
    )
    if has_competition and not report_path.is_file():
        raise ValueError(
            "ICDAR training requires an audited manifest with audit_report.json"
        )
    if report_path.is_file():
        from hccr.data.lmdb_store import manifest_digest

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
        if report.get("protocol") in OFFICIAL_PROTOCOLS:
            _validate_official_manifest(path, rows, report)
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
                competition = gnt_sources[row["source_file"]].startswith("icdar2013:")
                if competition != (row["split"] == "icdar"):
                    raise ValueError("ICDAR competition records must be icdar-only")
        provenance["dataset_protocol"] = report.get("protocol")
        provenance["image_overlap_check"] = report.get(
            "image_overlap_check", {"status": "not_run"}
        )
        provenance["manifest_digest"] = report["manifest_digest"]
        provenance["label_mapping_digest"] = mapping_digest
        provenance["official_source_provenance_required"] = report.get(
            "official_source_provenance_required", False
        )
        if report.get("protocol") in OFFICIAL_PROTOCOLS:
            provenance["hwdb_writer_disjoint_verified"] = True
            provenance["icdar_identity_reconciliation"] = report.get(
                "icdar_identity_reconciliation"
            )
            if (
                has_competition
                and report.get("icdar_identity_reconciliation")
                != "source_metadata_crosswalk"
            ):
                provenance["writer_disjoint_verified"] = None
                provenance["overlap_check"] = (
                    "source_ids_disjoint_cross_namespace_identity_unverified"
                )
    return provenance


def _validate_official_manifest(
    path: Path, rows: list[dict[str, str]], report: dict
) -> None:
    """Validate source provenance and fit/calibration support before model fitting."""
    from hccr.data.casia_manifest import OFFICIAL_PROTOCOLS, gnt_writer_id

    audit = audit_manifest(rows)
    versions, label_set, classes = OFFICIAL_PROTOCOLS[report["protocol"]]
    if (
        report.get("schema_version", 0) < 4
        or report.get("official_source_provenance_required") is not True
    ):
        raise ValueError("official protocol requires schema 4 source provenance")
    if (
        report.get("label_set") != label_set
        or audit.class_count != classes
        or report.get("class_count") != classes
    ):
        raise ValueError("official protocol label set or class count disagrees")
    if any(
        len(report.get(field, "")) != 64
        for field in ("manifest_digest", "label_mapping_digest")
    ):
        raise ValueError(
            "official protocol requires manifest and class-map SHA-256 hashes"
        )
    if report.get("split_counts") != audit.split_counts:
        raise ValueError("official audit split counts disagree with manifest")
    if not {"train", "calibration", "validation", "test"}.issubset(audit.split_counts):
        raise ValueError(
            "official proposal requires fit, calibration, validation, and test"
        )
    labels = json.loads(
        path.with_name("class_mapping_full.json").read_text(encoding="utf-8")
    )["labels"]
    expected_mapping = {str(item["class_id"]): item["unicode_label"] for item in labels}
    if (
        set(expected_mapping) != {str(index) for index in range(classes)}
        or len(labels) != classes
    ):
        raise ValueError("official class mapping is not contiguous or has wrong length")
    support: dict[str, set[str]] = defaultdict(set)
    observed_versions: dict[str, set[str]] = defaultdict(set)
    sources = {source["source_file"]: source for source in report.get("sources", [])}
    if not sources or any(
        len(source.get("sha256", "")) != 64 for source in sources.values()
    ):
        raise ValueError("official protocol requires source SHA-256 inventory")
    observed_writers: dict[str, set[str]] = defaultdict(set)
    observed_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observed_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in rows:
        source = sources.get(row["source_file"])
        if not source or not row["source_file"].lower().endswith(".gnt"):
            raise ValueError("official manifest requires audited GNT sources")
        partition = (
            "train"
            if row["split"] in {"train", "calibration", "validation"}
            else row["split"]
        )
        version = row.get("source_version")
        if version not in (("ICDAR2013",) if partition == "icdar" else versions):
            raise ValueError("official manifest has unknown source version")
        if (
            row.get("official_partition") != partition
            or source.get("assigned_split") != row["split"]
        ):
            raise ValueError("official source partition disagrees with assigned split")
        alias = gnt_writer_id(Path(row["source_file"]))
        if (
            row.get("source_writer_id") != alias
            or row.get("dataset_protocol") != report["protocol"]
        ):
            raise ValueError("official source identity or dataset protocol disagrees")
        for field in (
            "source_version",
            "official_partition",
            "source_writer_id",
            "writer_id",
        ):
            if source.get(field) != row.get(field):
                raise ValueError(f"official source provenance disagrees for {field}")
        if expected_mapping.get(row["class_id"]) != row["unicode_label"]:
            raise ValueError("official class mapping disagrees with manifest labels")
        support[row["split"]].add(row["class_id"])
        observed_writers[row["split"]].add(row["writer_id"])
        observed_versions[partition].add(version)
        observed_counts[version][row["split"]] += 1
        observed_class_counts[row["split"]][row["class_id"]] += 1
    for partition in ("train", "test"):
        if observed_versions[partition] != set(versions):
            raise ValueError("official manifest does not cover every source version")
    for split in ("train", "calibration"):
        if support[split] != set(expected_mapping):
            raise ValueError(f"classes missing from {split}")
    if report.get("writer_counts") != {
        split: len(writers) for split, writers in observed_writers.items()
    } or report.get("writer_lists") != {
        split: sorted(writers) for split, writers in observed_writers.items()
    }:
        raise ValueError("official audit writer lists disagree with manifest")
    if report.get("class_sample_counts_by_split") != {
        split: {class_id: counts.get(class_id, 0) for class_id in expected_mapping}
        for split, counts in observed_class_counts.items()
    } or report.get("class_counts_by_split") != {
        split: len(counts) for split, counts in observed_class_counts.items()
    }:
        raise ValueError("official class support counts disagree with manifest")
    if (
        report.get("image_overlap_check", {}).get("status") != "passed"
        or report["image_overlap_check"].get("checked_samples") != len(rows)
        or report["image_overlap_check"].get("cross_split_duplicates") != 0
    ):
        raise ValueError("official exact-image leakage audit must cover every sample")
    if any(len(row.get("pixel_sha256", "")) != 64 for row in rows):
        raise ValueError("official manifest requires native pixel fingerprints")
    version_counts = report.get("sample_counts_by_version_and_split", {})
    if set(version_counts) != set(observed_counts):
        raise ValueError("official per-version sample inventory is missing")
    if version_counts != {
        version: {split: counts.get(split, 0) for split in audit.split_counts}
        for version, counts in observed_counts.items()
    }:
        raise ValueError("official per-version sample counts disagree")
    aliases = {row["source_writer_id"] for row in rows if row["split"] == "icdar"}
    reconciled = aliases and aliases.issubset(report.get("writer_identity_mapping", {}))
    if aliases:
        expected = (
            "source_metadata_crosswalk"
            if reconciled
            else "unverified_source_namespace_ids"
        )
        if report.get("icdar_identity_reconciliation") != expected:
            raise ValueError(
                "ICDAR identity reconciliation claim disagrees with crosswalk"
            )
