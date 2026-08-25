from __future__ import annotations

import binascii
import csv
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.build_dataset_manifest import (
    PngValidationError,
    inspect_png,
    validation_files,
)
from scripts.build_dataset_manifest import main as build_manifest

from hccr.data.dataset import select_class_subset


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")
    )


def minimal_png() -> bytes:
    header = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\x08\x00\x00\x00\x00"
    image_data = zlib.compress(b"\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", image_data)
        + png_chunk(b"IEND", b"")
    )


class DatasetManifestTests(unittest.TestCase):
    def test_builds_manifest_directly_from_zip_with_custom_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "raw.zip"
            with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                archive.writestr("custom/payload/training/一/1.png", minimal_png())
                archive.writestr("custom/payload/training/一/2.png", minimal_png())
                archive.writestr("custom/payload/testing/一/3.png", minimal_png())
            output = root / "processed"

            exit_code = build_manifest(
                [
                    "--source-zip",
                    str(archive_path),
                    "--zip-prefix",
                    "/custom\\payload//",
                    "--zip-train-dir",
                    "training",
                    "--zip-test-dir",
                    "testing",
                    "--output-dir",
                    str(output),
                    "--validation-fraction",
                    "0.5",
                    "--workers",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                {row["split"] for row in rows}, {"train", "validation", "test"}
            )
            self.assertEqual(
                {row["source_file"] for row in rows},
                {
                    "training/一/1.png",
                    "training/一/2.png",
                    "testing/一/3.png",
                },
            )
            report = json.loads((output / "audit_report.json").read_text())
            self.assertEqual(report["source_kind"], "zip")
            self.assertEqual(report["source_prefix"], "custom/payload")

    def test_class_subset_is_deterministic_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                (
                    "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
                    "a,file-a,,A,2,train\n"
                    "b,file-b,,B,7,validation\n"
                    "c,file-c,,C,9,test\n"
                ),
                encoding="utf-8",
            )
            subset = select_class_subset(manifest, 2, seed=7)
            self.assertEqual(subset, select_class_subset(manifest, 2, seed=7))
            self.assertEqual(set(subset.values()), {0, 1})

    def test_inspect_png_reads_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.png"
            path.write_bytes(minimal_png())
            image = inspect_png(path)
        self.assertEqual((image.width, image.height), (1, 1))

    def test_inspect_png_rejects_bad_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.png"
            path.write_bytes(minimal_png()[:-1] + b"x")
            with self.assertRaises(PngValidationError):
                inspect_png(path)

    def test_validation_selection_is_reproducible_and_non_overlapping(self) -> None:
        files = [Path(f"{index}.png") for index in range(10)]
        first = validation_files(files, "一", 20260817, 0.1)
        second = validation_files(files, "一", 20260817, 0.1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertTrue(first.issubset(set(files)))
