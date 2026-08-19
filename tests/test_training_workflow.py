from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hccr.data.dataset import HCCRDataset
from hccr.training.workflow import TrainingConfig, run_training


class TrainingWorkflowTests(unittest.TestCase):
    def test_small_run_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (
                ("train-a", 0),
                ("train-b", 64),
                ("valid-a", 128),
                ("valid-b", 192),
            ):
                image_path = root / "raw" / f"{name}.png"
                image_path.parent.mkdir(exist_ok=True)
                Image.new("L", (12, 12), color=value).save(image_path)
            manifest_path = root / "data" / "processed" / "manifest.csv"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                (
                    "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
                    "train-a,raw/train-a.png,,A,0,train\n"
                    "train-b,raw/train-b.png,,B,1,train\n"
                    "valid-a,raw/valid-a.png,,A,0,validation\n"
                    "valid-b,raw/valid-b.png,,B,1,validation\n"
                ),
                encoding="utf-8",
            )
            run_training(
                TrainingConfig(
                    manifest_path=manifest_path,
                    output_dir=root / "experiments",
                    num_classes=2,
                    epochs=1,
                    batch_size=2,
                    image_size=16,
                    width=4,
                    scheduler="none",
                    early_stopping_patience=None,
                )
            )
            run_directory = next((root / "experiments").glob("20*"))
            metadata = json.loads((run_directory / "metadata.json").read_text())
            metrics = json.loads((run_directory / "metrics.json").read_text())
            checkpoint_metadata = json.loads(
                (run_directory / "checkpoint_metadata.json").read_text()
            )
            self.assertIn("run_id", metadata)
            self.assertEqual(metrics["best"], checkpoint_metadata["metrics"])
            self.assertIn("expected_calibration_error", checkpoint_metadata["metrics"])
            self.assertIn("mean_confidence", checkpoint_metadata["metrics"])
            self.assertEqual(
                checkpoint_metadata["preprocess"],
                {
                    "image_size": 16,
                    "margin": 4,
                    "invert": False,
                    "center_by_centroid": False,
                    "otsu_binarize": False,
                    "median_filter_size": None,
                },
            )
            self.assertIn("labels_digest", checkpoint_metadata)
            self.assertIsNone(checkpoint_metadata["class_subset_digest"])
            self.assertTrue((run_directory / "checkpoint.pt").is_file())
            self.assertTrue((run_directory / "checkpoint_metadata.json").is_file())
            self.assertTrue((run_directory / "labels.json").is_file())
            self.assertTrue((run_directory / "preprocessing_gallery.png").is_file())

    def test_manifest_paths_relative_to_data_raw_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "data" / "raw" / "character.png"
            image_path.parent.mkdir(parents=True)
            Image.new("L", (8, 8), color=0).save(image_path)
            manifest_path = root / "data" / "processed" / "manifest.csv"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                (
                    "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
                    "sample,character.png,,A,0,train\n"
                ),
                encoding="utf-8",
            )
            dataset = HCCRDataset(manifest_path, "train")
            self.assertEqual(dataset.root, root / "data" / "raw")
