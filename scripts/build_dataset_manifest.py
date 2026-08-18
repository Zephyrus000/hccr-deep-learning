"""Audit the folder-based CASIA export and create its frozen CSV manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Iterable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

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


def inspect_png(path: Path) -> ImageInfo:
    """Validate PNG signature, chunk CRCs, IHDR, IDAT, and IEND."""
    with path.open("rb") as image:
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


def inspect_path(path: Path) -> tuple[Path, ImageInfo | None, str | None]:
    """Return inspection failures as data so a worker cannot abort the audit."""
    try:
        return path, inspect_png(path), None
    except (OSError, PngValidationError) as error:
        return path, None, str(error)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.train_dir.is_dir() or not args.test_dir.is_dir():
        raise FileNotFoundError("train-dir and test-dir must both exist")

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
        "class_count": len(train_labels),
        "manifest_digest": digest.hexdigest(),
        "partitions": [asdict(train_summary), asdict(test_summary)],
        "invalid_entry_count": len(invalid_rows),
    }
    invalid_path.write_text(
        json.dumps(invalid_rows, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    if invalid_rows:
        manifest_temp.unlink(missing_ok=True)
        print(f"Audit failed: {len(invalid_rows)} invalid entries. See {invalid_path}.")
        return 1

    manifest_temp.replace(manifest_path)
    write_mapping(args.output_dir / "class_mapping_full.json", train_labels)
    write_mapping(args.output_dir / "class_mapping_hccr1_100.json", train_labels[:100])
    print(f"Audit passed. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
