from __future__ import annotations

import binascii
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.build_dataset_manifest import (
    PngValidationError,
    inspect_png,
    validation_files,
)

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
