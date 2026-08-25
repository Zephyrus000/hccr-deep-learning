"""Audit a folder or ZIP-based CASIA export and create frozen CSV manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
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
)


class PngValidationError(ValueError):
    """Raised when a file is not a structurally valid PNG."""


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int


@dataclass
class AuditSummary:
    split: str
    files_seen: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    train_images: int = 0
    validation_images: int = 0
    test_images: int = 0


@dataclass(frozen=True)
class ZipImageMember:
    archive_name: str
    source_file: str

    @property
    def name(self) -> str:
        return PurePosixPath(self.source_file).name


def inspect_png(path: Path) -> ImageInfo:
    """Validate PNG signature, chunk CRCs, IHDR, IDAT, and IEND."""
    with path.open("rb") as image:
        return inspect_png_stream(image)


def inspect_png_stream(image: BinaryIO) -> ImageInfo:
    """Validate a PNG from a file or archive member stream."""
    try:
        if image.read(8) != PNG_SIGNATURE:
            raise PngValidationError("invalid PNG signature")

        found_ihdr = False
        found_idat = False
        decompressor: zlib.Decompress | None = None
        width = height = 0
        while True:
            length_bytes = image.read(4)
            if not length_bytes:
                raise PngValidationError("missing IEND chunk")
            if len(length_bytes) != 4:
                raise PngValidationError("truncated chunk length")
            length = int.from_bytes(length_bytes, "big")
            chunk_type = image.read(4)
            payload = image.read(length)
            crc_bytes = image.read(4)
            if len(chunk_type) != 4 or len(payload) != length or len(crc_bytes) != 4:
                raise PngValidationError("truncated PNG chunk")
            expected_crc = int.from_bytes(crc_bytes, "big")
            actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise PngValidationError("PNG chunk CRC mismatch")

            if chunk_type == b"IHDR":
                if found_ihdr or length != 13:
                    raise PngValidationError("invalid IHDR chunk")
                width = int.from_bytes(payload[:4], "big")
                height = int.from_bytes(payload[4:8], "big")
                if width == 0 or height == 0:
                    raise PngValidationError("zero-sized image")
                found_ihdr = True
            elif chunk_type == b"IDAT":
                if not found_ihdr:
                    raise PngValidationError("IDAT precedes IHDR")
                found_idat = True
                if decompressor is None:
                    decompressor = zlib.decompressobj()
                try:
                    decompressor.decompress(payload)
                except zlib.error as error:
                    raise PngValidationError("invalid PNG image data") from error
            elif chunk_type == b"IEND":
                if length != 0 or not found_ihdr or not found_idat:
                    raise PngValidationError("invalid IEND chunk")
                if decompressor is None:
                    raise PngValidationError("missing PNG image data")
                try:
                    decompressor.flush()
                except zlib.error as error:
                    raise PngValidationError("invalid PNG image data") from error
                if not decompressor.eof:
                    raise PngValidationError("truncated PNG image data")
                if image.read(1):
                    raise PngValidationError("trailing bytes after IEND")
                return ImageInfo(width=width, height=height)
    except zlib.error as error:
        raise PngValidationError("invalid PNG image data") from error


def inspect_path(path: Path) -> tuple[Path, ImageInfo | None, str | None]:
    """Return inspection failures as data so a worker cannot abort the audit."""
    try:
        return path, inspect_png(path), None
    except (OSError, PngValidationError) as error:
        return path, None, str(error)


def inspect_zip_member(
    archive: ZipFile, member: ZipImageMember
) -> tuple[ZipImageMember, ImageInfo | None, str | None]:
    """Inspect one ZIP member and return failures as audit data."""
    try:
        with archive.open(member.archive_name) as image:
            return member, inspect_png_stream(image), None
    except (BadZipFile, KeyError, OSError, PngValidationError) as error:
        return member, None, str(error)


def zip_partition_members(
    archive: ZipFile, zip_prefix: str, partition_directory: str
) -> dict[str, list[ZipImageMember]]:
    """Group direct `<label>/<image>` members under an archive partition."""
    normalized_prefix = _normalize_archive_path(zip_prefix, allow_empty=True)
    normalized_partition = _normalize_archive_path(
        partition_directory, allow_empty=False
    )
    partition_root = (
        f"{normalized_prefix}/{normalized_partition}"
        if normalized_prefix
        else normalized_partition
    )
    member_prefix = f"{partition_root}/"
    source_prefix = f"{normalized_prefix}/" if normalized_prefix else ""
    members: dict[str, list[ZipImageMember]] = {}
    for info in archive.infolist():
        archive_name = info.filename
        normalized_name = archive_name.replace("\\", "/").lstrip("/")
        if info.is_dir() or not normalized_name.startswith(member_prefix):
            continue
        relative_partition_path = normalized_name[len(member_prefix) :]
        parts = relative_partition_path.split("/")
        if len(parts) != 2 or not parts[1]:
            continue
        label = parts[0]
        if len(label) != 1:
            raise ValueError(
                "label directory is not one Unicode character: " f"{normalized_name}"
            )
        source_file = normalized_name.removeprefix(source_prefix)
        members.setdefault(label, []).append(
            ZipImageMember(archive_name=archive_name, source_file=source_file)
        )
    if not members:
        raise FileNotFoundError(f"no images found under ZIP path: {partition_root}")
    for files in members.values():
        files.sort(key=lambda member: member.source_file)
    return members


def _normalize_archive_path(value: str, *, allow_empty: bool) -> str:
    normalized = value.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"ZIP path must not contain '..': {value}")
    result = "/".join(parts)
    if not result and not allow_empty:
        raise ValueError("ZIP partition directory must not be empty")
    return result


def labels_in(directory: Path) -> list[str]:
    labels = []
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if len(child.name) != 1:
            raise ValueError(f"label directory is not one Unicode character: {child}")
        labels.append(child.name)
    return sorted(labels)


def validation_files(
    files: Iterable[Path], label: str, seed: int, fraction: float
) -> set[Path]:
    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{seed}\0{label}\0{path.name}".encode()
        ).digest(),
    )
    count = len(ranked)
    validation_count = 0 if count < 2 else max(1, math.floor(count * fraction))
    return set(ranked[:validation_count])


def sample_id(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()


def write_mapping(path: Path, labels: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "labels": [
            {"class_id": class_id, "unicode_label": label}
            for class_id, label in enumerate(labels)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


def audit_partition(
    directory: Path,
    source_root: Path,
    split: str,
    class_ids: dict[str, int],
    writer: csv.DictWriter,
    invalid_rows: list[dict[str, str]],
    digest: object,
    seed: int,
    validation_fraction: float,
    executor: Executor,
) -> AuditSummary:
    summary = AuditSummary(split=split)
    for label in sorted(class_ids):
        label_directory = directory / label
        files = sorted(path for path in label_directory.iterdir() if path.is_file())
        if not files:
            invalid_rows.append({"path": str(label_directory), "reason": "empty class"})
            continue
        validation = (
            validation_files(files, label, seed, validation_fraction)
            if split == "train"
            else set()
        )
        for path, image, error in executor.map(inspect_path, files):
            summary.files_seen += 1
            if error is not None:
                summary.invalid_images += 1
                invalid_rows.append({"path": str(path), "reason": error})
                continue

            if image is None:
                raise RuntimeError("inspection result did not contain an image")

            final_split = "validation" if path in validation else split
            source_file = path.relative_to(source_root).as_posix()
            row = {
                "sample_id": sample_id(source_file),
                "source_file": source_file,
                "record_offset": "",
                "writer_id": "",
                "raw_label": label,
                "unicode_label": label,
                "class_id": class_ids[label],
                "width": image.width,
                "height": image.height,
                "split": final_split,
            }
            writer.writerow(row)
            # hashlib's concrete hash type is intentionally not public.
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
            digest.update(payload)  # type: ignore[attr-defined]
            digest.update(b"\n")  # type: ignore[attr-defined]
            summary.valid_images += 1
            if final_split == "train":
                summary.train_images += 1
            elif final_split == "validation":
                summary.validation_images += 1
            else:
                summary.test_images += 1
    return summary


def audit_zip_partition(
    archive: ZipFile,
    members_by_label: dict[str, list[ZipImageMember]],
    split: str,
    class_ids: dict[str, int],
    writer: csv.DictWriter,
    invalid_rows: list[dict[str, str]],
    digest: object,
    seed: int,
    validation_fraction: float,
    executor: Executor,
) -> AuditSummary:
    """Audit a train/test partition directly inside a ZIP archive."""
    summary = AuditSummary(split=split)
    for label in sorted(class_ids):
        members = members_by_label.get(label, [])
        if not members:
            invalid_rows.append({"path": f"ZIP label {label}", "reason": "empty class"})
            continue
        validation = (
            {
                path.as_posix()
                for path in validation_files(
                    [PurePosixPath(member.source_file) for member in members],
                    label,
                    seed,
                    validation_fraction,
                )
            }
            if split == "train"
            else set()
        )
        inspected = executor.map(
            lambda member: inspect_zip_member(archive, member), members
        )
        for member, image, error in inspected:
            summary.files_seen += 1
            if error is not None:
                summary.invalid_images += 1
                invalid_rows.append({"path": member.archive_name, "reason": error})
                continue
            if image is None:
                raise RuntimeError("inspection result did not contain an image")

            final_split = "validation" if member.source_file in validation else split
            row = {
                "sample_id": sample_id(member.source_file),
                "source_file": member.source_file,
                "record_offset": "",
                "writer_id": "",
                "raw_label": label,
                "unicode_label": label,
                "class_id": class_ids[label],
                "width": image.width,
                "height": image.height,
                "split": final_split,
            }
            writer.writerow(row)
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
            digest.update(payload)  # type: ignore[attr-defined]
            digest.update(b"\n")  # type: ignore[attr-defined]
            summary.valid_images += 1
            if final_split == "train":
                summary.train_images += 1
            elif final_split == "validation":
                summary.validation_images += 1
            else:
                summary.test_images += 1
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-zip",
        type=Path,
        help="Audit images directly from a ZIP instead of extracted folders.",
    )
    parser.add_argument(
        "--zip-prefix",
        default="",
        help="Optional archive prefix before ZIP train/test directories.",
    )
    parser.add_argument(
        "--zip-train-dir",
        default="CASIA-HWDB_Train/Train",
        help="Train directory relative to --zip-prefix.",
    )
    parser.add_argument(
        "--zip-test-dir",
        default="CASIA-HWDB_Test/Test",
        help="Test directory relative to --zip-prefix.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--train-dir", type=Path, default=Path("data/raw/CASIA-HWDB_Train/Train")
    )
    parser.add_argument(
        "--test-dir", type=Path, default=Path("data/raw/CASIA-HWDB_Test/Test")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/casia_hwdb")
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent PNG inspections (default: 16).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.zip_prefix and args.source_zip is None:
        raise ValueError("--zip-prefix requires --source-zip")

    archive = ZipFile(args.source_zip) if args.source_zip is not None else None
    try:
        if archive is not None:
            train_members = zip_partition_members(
                archive, args.zip_prefix, args.zip_train_dir
            )
            test_members = zip_partition_members(
                archive, args.zip_prefix, args.zip_test_dir
            )
            train_labels = sorted(train_members)
            test_labels = sorted(test_members)
        else:
            if not args.train_dir.is_dir() or not args.test_dir.is_dir():
                raise FileNotFoundError("train-dir and test-dir must both exist")
            train_members = test_members = None
            train_labels = labels_in(args.train_dir)
            test_labels = labels_in(args.test_dir)
        if train_labels != test_labels:
            raise ValueError(
                "Train and Test label sets differ; refusing to create a manifest"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_temp = args.output_dir / "manifest.csv.tmp"
        manifest_path = args.output_dir / "manifest.csv"
        invalid_path = args.output_dir / "invalid_images.json"
        report_path = args.output_dir / "audit_report.json"
        class_ids = {label: index for index, label in enumerate(train_labels)}
        invalid_rows: list[dict[str, str]] = []
        digest = hashlib.sha256()

        with (
            manifest_temp.open("w", newline="", encoding="utf-8") as handle,
            ThreadPoolExecutor(max_workers=args.workers) as executor,
        ):
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            if archive is not None:
                if train_members is None or test_members is None:
                    raise RuntimeError("ZIP partitions were not initialized")
                train_summary = audit_zip_partition(
                    archive,
                    train_members,
                    "train",
                    class_ids,
                    writer,
                    invalid_rows,
                    digest,
                    args.seed,
                    args.validation_fraction,
                    executor,
                )
                test_summary = audit_zip_partition(
                    archive,
                    test_members,
                    "test",
                    class_ids,
                    writer,
                    invalid_rows,
                    digest,
                    args.seed,
                    args.validation_fraction,
                    executor,
                )
            else:
                train_summary = audit_partition(
                    args.train_dir,
                    args.data_root,
                    "train",
                    class_ids,
                    writer,
                    invalid_rows,
                    digest,
                    args.seed,
                    args.validation_fraction,
                    executor,
                )
                test_summary = audit_partition(
                    args.test_dir,
                    args.data_root,
                    "test",
                    class_ids,
                    writer,
                    invalid_rows,
                    digest,
                    args.seed,
                    args.validation_fraction,
                    executor,
                )

        report = {
            "schema_version": 1,
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "writer_id_policy": "unavailable_in_source_export",
            "source_kind": "zip" if archive is not None else "filesystem",
            "source_prefix": (
                _normalize_archive_path(args.zip_prefix, allow_empty=True)
                if archive is not None
                else ""
            ),
            "class_count": len(train_labels),
            "manifest_digest": digest.hexdigest(),
            "partitions": [asdict(train_summary), asdict(test_summary)],
            "invalid_entry_count": len(invalid_rows),
        }
        invalid_path.write_text(
            json.dumps(invalid_rows, ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
        if invalid_rows:
            manifest_temp.unlink(missing_ok=True)
            print(
                f"Audit failed: {len(invalid_rows)} invalid entries. "
                f"See {invalid_path}."
            )
            return 1

        manifest_temp.replace(manifest_path)
        write_mapping(args.output_dir / "class_mapping_full.json", train_labels)
        write_mapping(
            args.output_dir / "class_mapping_hccr1_100.json", train_labels[:100]
        )
        print(f"Audit passed. Manifest: {manifest_path}")
        return 0
    finally:
        if archive is not None:
            archive.close()


if __name__ == "__main__":
    sys.exit(main())
