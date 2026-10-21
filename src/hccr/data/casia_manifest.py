"""Build frozen manifests and shared class mappings from official CASIA GNTs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm.auto import tqdm

from hccr.data.gnt import iter_gnt_records
from hccr.data.leakage import ExactImageAudit
from hccr.data.lmdb_store import manifest_digest
from hccr.data.splitter import WriterDisjointSplitter

MANIFEST_COLUMNS = (
    "sample_id",
    "source_file",
    "record_offset",
    "writer_id",
    "raw_label",
    "unicode_label",
    "class_id",
    "width",
    "height",
    "split",
    "pixel_sha256",
)


def gnt_writer_id(path: Path) -> str:
    """Preserve CASIA numeric IDs and ICDAR's literal C-prefixed writer IDs."""
    match = re.fullmatch(r"(C?)(\d+)(?:-[A-Za-z]+)*", path.stem)
    if match is None:
        raise ValueError(f"cannot derive CASIA writer ID from filename: {path}")
    if match[1]:
        return f"icdar2013:C{int(match[2]):03d}"
    return f"casia:{int(match[2])}"


def _keep_label(label: str, label_set: str) -> bool:
    if len(label) != 1:
        return False
    if label_set == "gb2312-level1":
        try:
            code = label.encode("gb2312")
        except UnicodeEncodeError:
            return False
        return len(code) == 2 and b"\xb0\xa1" <= code <= b"\xd7\xf9"
    value = ord(label)
    return (
        0x3400 <= value <= 0x4DBF
        or 0x4E00 <= value <= 0x9FFF
        or 0xF900 <= value <= 0xFAFF
        or 0x20000 <= value <= 0x323AF
        or value == 0x3007
    )


def build_casia_manifest(
    train_dir: Path | Sequence[Path],
    test_dir: Path | Sequence[Path],
    source_root: Path,
    output_dir: Path,
    *,
    protocol: str = "custom",
    label_set: str = "all-hanzi",
    encoding: str = "gb18030",
    seed: int = 20260817,
    validation_fraction: float = 0.1,
    expected_class_count: int | None = None,
    show_progress: bool = False,
    development_writer_ids: set[str] | None = None,
    expected_development_writers: int | None = None,
    expected_test_writers: int | None = None,
    expected_test_samples: int | None = None,
    audit_pixel_duplicates: bool = False,
    icdar_dir: Path | None = None,
    expected_icdar_writers: int | None = None,
    expected_icdar_samples: int | None = None,
) -> dict:
    """Preserve official train/test partitions and hold out training writers.

    Class IDs follow Unicode-sorted training labels, never test-set ordering.
    A separate output directory is published only after both passes succeed.
    """
    if label_set not in {"gb2312-level1", "all-hanzi"}:
        raise ValueError("label_set must be gb2312-level1 or all-hanzi")
    splitter = WriterDisjointSplitter(validation_fraction, seed)
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output_dir}")
    train_directories = [train_dir] if isinstance(train_dir, Path) else list(train_dir)
    partitions = {}
    excluded_sources = []
    test_directories = [test_dir] if isinstance(test_dir, Path) else list(test_dir)
    source_partitions = {"train": train_directories, "test": test_directories}
    if icdar_dir is not None:
        source_partitions["icdar"] = [icdar_dir]
    for split, directories in source_partitions.items():
        files = sorted(
            {
                path
                for directory in directories
                for path in directory.resolve().rglob("*.gnt")
            }
        )
        if split == "train" and development_writer_ids is not None:
            excluded_sources = [
                path.relative_to(source_root).as_posix()
                for path in files
                if gnt_writer_id(path) not in development_writer_ids
            ]
            files = [
                path for path in files if gnt_writer_id(path) in development_writer_ids
            ]
        if not files:
            raise FileNotFoundError(f"no selected .gnt files in {directories}")
        for path in files:
            path.relative_to(source_root)
        partitions[split] = files
    writers = {
        split: {gnt_writer_id(path) for path in files}
        for split, files in partitions.items()
    }
    for left, right in combinations(writers, 2):
        overlap = writers[left] & writers[right]
        if overlap:
            raise ValueError(f"{left}/{right} writer overlap: {sorted(overlap)}")
    for partition, expected in (
        ("train", expected_development_writers),
        ("test", expected_test_writers),
        ("icdar", expected_icdar_writers),
    ):
        if expected is not None and len(writers.get(partition, set())) != expected:
            raise ValueError(
                f"expected {expected} {partition} writers, "
                f"found {len(writers.get(partition, set()))}"
            )
    if protocol.startswith("icdar2013") and icdar_dir is None:
        raise ValueError("ICDAR protocols require a separate icdar_dir and CASIA test")
    if any(
        writer.startswith("icdar2013:")
        for partition in ("train", "test")
        for writer in writers[partition]
    ):
        raise ValueError("competition records belong only in the separate icdar split")
    if icdar_dir is not None:
        if any(not writer.startswith("icdar2013:") for writer in writers["icdar"]):
            raise ValueError("ICDAR test requires C-prefixed competition writer files")
        if any(
            not writer.startswith("casia:")
            for partition in ("train", "test")
            for writer in writers[partition]
        ):
            raise ValueError("competition writers cannot enter CASIA train/test")
        if not audit_pixel_duplicates:
            raise ValueError("ICDAR protocols require the exact-image leakage audit")
        if expected_icdar_writers == 60 and writers["icdar"] != {
            f"icdar2013:C{i:03d}" for i in range(1, 61)
        }:
            raise ValueError("ICDAR benchmark requires writers C001 through C060")
    if len(writers["train"]) < 2:
        raise ValueError("writer-disjoint validation requires at least two writers")
    validation_writers = splitter.validation_writers(writers["train"])
    split_labels: dict[str, set[str]] = defaultdict(set)
    counts: Counter = Counter()
    sources = []
    assignments = []
    for partition, files in partitions.items():
        for path in tqdm(
            files, desc=f"audit {partition} GNT", disable=not show_progress
        ):
            writer_id = gnt_writer_id(path)
            split = (
                "validation"
                if partition == "train" and writer_id in validation_writers
                else partition
            )
            assignments.append((path, writer_id, split))
            stat = path.stat()
            sources.append(
                {
                    "source_file": path.relative_to(source_root).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "official_partition": partition,
                    "assigned_split": split,
                }
            )
            for record in iter_gnt_records(path, allow_empty=True):
                counts["records_seen"] += 1
                if not record.width:
                    counts["empty_records"] += 1
                    continue
                label = record.tag.rstrip(b"\x00").decode(encoding)
                if _keep_label(label, label_set):
                    split_labels[split].add(label)
                    counts[split] += 1
                else:
                    counts["excluded_by_label_set"] += 1
    training_labels = split_labels["train"] | split_labels["validation"]
    if not training_labels or not split_labels["test"]:
        raise ValueError("label filter produced an empty training or test partition")
    for partition in partitions:
        if not split_labels[partition]:
            raise ValueError(f"label filter produced an empty {partition} partition")
        unseen = split_labels[partition] - training_labels
        if unseen:
            raise ValueError(
                f"{partition} labels absent from training: {sorted(unseen)}"
            )
    missing_train = training_labels - split_labels["train"]
    if missing_train:
        raise ValueError(f"classes missing from train: {sorted(missing_train)}")
    labels = sorted(training_labels)
    if expected_class_count is not None and len(labels) != expected_class_count:
        raise ValueError(
            f"expected {expected_class_count} classes, found {len(labels)}"
        )
    class_ids = {label: index for index, label in enumerate(labels)}
    if expected_test_samples is not None and counts["test"] != expected_test_samples:
        raise ValueError(
            f"expected {expected_test_samples} test samples, found {counts['test']}"
        )
    if expected_icdar_samples is not None and counts["icdar"] != expected_icdar_samples:
        raise ValueError(
            f"expected {expected_icdar_samples} icdar samples, found {counts['icdar']}"
        )
    split_writers = {
        **writers,
        "train": writers["train"] - validation_writers,
        "validation": validation_writers,
    }
    split_names = ("train", "validation", "test") + (("icdar",) if icdar_dir else ())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temp:
        staging = Path(temp)
        manifest_path = staging / "manifest.csv"
        image_audit = (
            ExactImageAudit(staging / "image_audit.sqlite")
            if audit_pixel_duplicates
            else None
        )
        try:
            _write_manifest(
                manifest_path,
                assignments,
                sources,
                encoding,
                label_set,
                class_ids,
                image_audit,
                show_progress,
            )
        finally:
            if image_audit is not None:
                image_audit.close()
                (staging / "image_audit.sqlite").unlink(missing_ok=True)
        mapping = {
            "schema_version": 1,
            "labels": [
                {"class_id": index, "unicode_label": label}
                for index, label in enumerate(labels)
            ],
        }
        mapping_path = staging / "class_mapping_full.json"
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "schema_version": 3,
            "protocol": protocol,
            "source_kind": "gnt",
            "label_set": label_set,
            "encoding": encoding,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "validation_split_policy": "writer_disjoint_from_official_train_only",
            "holdout_policy": "casia_test_and_icdar_separate_never_used_for_selection",
            "writer_id_policy": "source_filename_ids_preserving_competition_C_prefix",
            "writer_disjoint_verified": True,
            "writer_verification_basis": (
                "declared_source_protocol_and_filename_ids; "
                "not_biometric_verification"
            ),
            "protocol_references": [
                "https://nlpr.ia.ac.cn/databases/handwriting/Download.html",
                "https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html",
            ],
            "writer_counts": {
                split: len(split_writers[split]) for split in split_names
            },
            "pairwise_writer_overlap_counts": {
                f"{left}/{right}": len(split_writers[left] & split_writers[right])
                for left, right in combinations(split_names, 2)
            },
            "validation_writers": sorted(validation_writers),
            "class_count": len(labels),
            "class_counts_by_split": {
                split: len(split_labels[split]) for split in split_names
            },
            "missing_labels_by_split": {
                split: sorted(training_labels - split_labels[split])
                for split in split_names
                if split != "train"
            },
            "split_counts": {split: counts[split] for split in split_names},
            "records_seen": counts["records_seen"],
            "excluded_by_label_set": counts["excluded_by_label_set"],
            "empty_records": counts["empty_records"],
            "excluded_development_sources": excluded_sources,
            "image_overlap_check": {
                "status": "passed" if image_audit is not None else "not_run",
                "method": "sha256_native_dimensions_and_grayscale_pixels_without_label",
                "checked_samples": image_audit.checked_samples if image_audit else 0,
                "same_split_duplicates": (
                    image_audit.same_split_duplicates if image_audit else None
                ),
                "cross_split_duplicates": 0 if image_audit else None,
                "policy": "fail_on_cross_split_duplicates_without_removing_samples",
            },
            "manifest_digest": manifest_digest(manifest_path),
            "label_mapping_digest": manifest_digest(mapping_path),
            "source_root": str(source_root),
            "sources": sources,
        }
        for name, payload in (
            (
                "labels.json",
                {"labels": {str(i): label for i, label in enumerate(labels)}},
            ),
            ("audit_report.json", report),
        ):
            (staging / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        staging.rename(output_dir)
    return report


def _write_manifest(
    manifest_path: Path,
    assignments: list[tuple[Path, str, str]],
    sources: list[dict],
    encoding: str,
    label_set: str,
    class_ids: dict[str, int],
    image_audit: ExactImageAudit | None,
    show_progress: bool,
) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        csv_writer.writeheader()
        for (path, writer_id, split), source in tqdm(
            zip(assignments, sources, strict=True),
            total=len(assignments),
            desc="write GNT manifest",
            disable=not show_progress,
        ):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                source["size_bytes"],
                source["mtime_ns"],
            ):
                raise ValueError(f"GNT changed during manifest build: {path}")
            source["sha256"] = manifest_digest(path)
            for record in iter_gnt_records(
                path, pixel_hashes=image_audit is not None, allow_empty=True
            ):
                if not record.width:
                    continue
                label = record.tag.rstrip(b"\x00").decode(encoding)
                if not _keep_label(label, label_set):
                    continue
                relative = source["source_file"]
                if image_audit is not None:
                    image_audit.add(
                        record.pixel_sha256, split, f"{relative}:{record.offset}"
                    )
                identity = f"{relative}\0{record.offset}".encode()
                csv_writer.writerow(
                    {
                        "sample_id": hashlib.sha256(identity).hexdigest(),
                        "source_file": relative,
                        "record_offset": record.offset,
                        "writer_id": writer_id,
                        "raw_label": record.tag.hex(),
                        "unicode_label": label,
                        "class_id": class_ids[label],
                        "width": record.width,
                        "height": record.height,
                        "split": split,
                        "pixel_sha256": record.pixel_sha256,
                    }
                )
