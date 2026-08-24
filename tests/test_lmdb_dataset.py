from __future__ import annotations

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from hccr.data import HCCRDataset, build_lmdb_image_store, read_manifest


class LMDBDatasetTests(unittest.TestCase):
    def test_lmdb_matches_filesystem_and_survives_spawn_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            rows = read_manifest(manifest)
            lmdb_path = manifest.with_name("images.lmdb")
            report = build_lmdb_image_store(manifest, rows, root, lmdb_path)
            filesystem = HCCRDataset(manifest, "train", storage_backend="filesystem")
            stored = HCCRDataset(manifest, "train", storage_backend="lmdb")

            filesystem_sample = filesystem[0]
            stored_sample = stored[0]
            self.assertTrue(torch.equal(filesystem_sample[0], stored_sample[0]))
            self.assertEqual(filesystem_sample[1:], stored_sample[1:])
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(stored.storage_backend, "lmdb")

            restored = pickle.loads(pickle.dumps(stored))
            self.assertIsNone(restored.image_store._environment)
            self.assertTrue(torch.equal(restored[1][0], stored[1][0]))
            worker_dataset = HCCRDataset(
                manifest,
                "train",
                storage_backend="lmdb",
                metadata_mode="index",
            )
            worker_batches = list(
                DataLoader(
                    worker_dataset,
                    batch_size=1,
                    num_workers=2,
                    multiprocessing_context="spawn",
                )
            )
            self.assertEqual([batch[2].item() for batch in worker_batches], [0, 1])
            worker_dataset.close()
            restored.close()
            stored.close()

    def test_compact_metadata_and_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_fixture(root)
            rows = read_manifest(manifest)
            lmdb_path = manifest.with_name("images.lmdb")
            build_lmdb_image_store(manifest, rows, root, lmdb_path)

            augmentations = HCCRDataset(
                manifest,
                "train",
                storage_backend="lmdb",
                metadata_mode="augmentations",
            )
            indices = HCCRDataset(
                manifest,
                "train",
                storage_backend="lmdb",
                metadata_mode="index",
            )
            self.assertEqual(augmentations[0][2], "")
            self.assertEqual(indices[1][2], 1)
            self.assertEqual(
                indices.metadata_for_index(1),
                {"sample_id": "b", "source_file": "images/b.png"},
            )

            manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different manifest"):
                HCCRDataset(manifest, "train", storage_backend="lmdb")
            augmentations.close()
            indices.close()

    @staticmethod
    def _write_fixture(root: Path) -> Path:
        image_directory = root / "images"
        image_directory.mkdir()
        Image.new("L", (8, 8), 32).save(image_directory / "a.png")
        Image.new("L", (8, 8), 224).save(image_directory / "b.png")
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "sample_id",
                    "source_file",
                    "writer_id",
                    "unicode_label",
                    "class_id",
                    "split",
                )
            )
            writer.writerow(("a", "images/a.png", "", "A", 0, "train"))
            writer.writerow(("b", "images/b.png", "", "B", 1, "train"))
        return manifest
