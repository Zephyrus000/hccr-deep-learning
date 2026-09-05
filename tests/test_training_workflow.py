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
            experiments_dir = root / "experiments"
            experiments_dir.mkdir()
            (experiments_dir / "experiment_summary.csv").write_text(
                "run_id,margin_warmup_epochs\nlegacy-run,2\n",
                encoding="utf-8",
            )
            run_training(
                TrainingConfig(
                    manifest_path=manifest_path,
                    output_dir=experiments_dir,
                    num_classes=2,
                    epochs=1,
                    batch_size=2,
                    image_size=16,
                    width=4,
                    stage_depths=(1, 1, 1),
                    label_smoothing=0.05,
                    scheduler="none",
                    reference_batch_size=2,
                    learning_rate_scaling="none",
                    lr_warmup_ratio=0.0,
                    early_stopping_patience=None,
                    benchmark_warmup_iterations=1,
                    benchmark_iterations=2,
                    benchmark_repetitions=2,
                    bn_recalibration_batches=1,
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
            self.assertIn("macro_recall", checkpoint_metadata["metrics"])
            self.assertIn("tail_recall", checkpoint_metadata["metrics"])
            self.assertIn("validation_stability", metrics)
            self.assertEqual(checkpoint_metadata["model"]["stage_depths"], [1, 1, 1])
            self.assertEqual(checkpoint_metadata["model"]["dropout"], 0.1)
            self.assertEqual(checkpoint_metadata["model"]["stem_stride"], 2)
            self.assertTrue(checkpoint_metadata["model"]["reparameterize_depthwise"])
            self.assertEqual(checkpoint_metadata["schema_version"], 4)
            self.assertEqual(
                checkpoint_metadata["model"]["classification_head"], "cosface"
            )
            self.assertEqual(checkpoint_metadata["model"]["logit_scale"], 32.0)
            self.assertEqual(checkpoint_metadata["model"]["angular_margin"], 0.1)
            self.assertEqual(
                checkpoint_metadata["model"]["effective_input_channels"], 1
            )
            self.assertEqual(
                checkpoint_metadata["training"],
                {
                    "loss": "cross_entropy",
                    "label_smoothing": 0.05,
                    "margin_schedule": "linear_warmup",
                    "margin_warmup_ratio": 0.2,
                    "resolved_margin_warmup_epochs": 1,
                    "optimizer_step_policy": "reference_batch",
                    "reference_batch_size": 2,
                    "batch_size": 2,
                    "configured_epochs": 1,
                    "resolved_epochs": 1,
                    "steps_per_epoch": 1,
                    "reference_steps_per_epoch": 1,
                    "total_optimizer_steps": 1,
                    "learning_rate_scaling": "none",
                    "base_learning_rate": 0.001,
                    "resolved_learning_rate": 0.001,
                    "lr_warmup_ratio": 0.0,
                    "warmup_steps": 0,
                    "augmentation": "random_affine_and_blur",
                },
            )
            self.assertEqual(
                checkpoint_metadata["preprocess"],
                {"image_size": 16, "margin": 4},
            )
            self.assertIn("labels_digest", checkpoint_metadata)
            self.assertIsNone(checkpoint_metadata["class_subset_digest"])
            self.assertTrue((run_directory / "checkpoint.pt").is_file())
            self.assertTrue((run_directory / "checkpoint_metadata.json").is_file())
            self.assertTrue((run_directory / "checkpoint_recalibrated.pt").is_file())
            self.assertTrue(
                (run_directory / "checkpoint_recalibrated_metadata.json").is_file()
            )
            self.assertTrue((run_directory / "bn_recalibration.json").is_file())
            self.assertTrue(
                (run_directory / "bn_recalibrated" / "per_class_metrics.csv").is_file()
            )
            self.assertTrue(
                (run_directory / "bn_recalibrated" / "calibration_bins.json").is_file()
            )
            self.assertTrue(
                (run_directory / "bn_recalibrated" / "validation_errors.csv").is_file()
            )
            self.assertTrue((run_directory / "labels.json").is_file())
            self.assertTrue((run_directory / "batch_training_plan.json").is_file())
            self.assertTrue((run_directory / "preprocessing_gallery.png").is_file())
            self.assertTrue((run_directory / "augmentation_gallery.png").is_file())
            self.assertTrue((run_directory / "validation_stability.json").is_file())
            self.assertTrue((run_directory / "class_tiers.json").is_file())
            self.assertTrue((run_directory / "confusion_pairs.csv").is_file())
            curves = json.loads((run_directory / "curves.json").read_text())["epochs"]
            self.assertIn("batch_norm", curves[0])
            self.assertNotIn("bn_recalibrated_top1", curves[0])
            self.assertIn("bn_recalibrated", metrics)
            self.assertIn("expected_calibration_error", metrics["bn_recalibrated"])
            recalibrated_metadata = json.loads(
                (run_directory / "checkpoint_recalibrated_metadata.json").read_text()
            )
            self.assertEqual(
                recalibrated_metadata["metrics"], metrics["bn_recalibrated"]
            )
            recalibration = json.loads(
                (run_directory / "bn_recalibration.json").read_text()
            )
            self.assertEqual(recalibration["source_checkpoint"], "checkpoint.pt")
            self.assertEqual(recalibration["checkpoint"], "checkpoint_recalibrated.pt")
            resource_profile = json.loads(
                (run_directory / "resource_profile.json").read_text()
            )
            self.assertEqual(
                resource_profile["parameter_count"],
                resource_profile["backbone_parameter_count"]
                + resource_profile["head_parameter_count"],
            )
            self.assertEqual(resource_profile["effective_input_channels"], 1)
            self.assertEqual(resource_profile["estimated_input_adapter_macs"], 0)
            self.assertIn("mac_coverage", resource_profile)
            self.assertEqual(
                resource_profile["full_class_projection"]["num_classes"], 7186
            )
            self.assertIsNotNone(resource_profile["end_to_end_batch1_benchmark"])
            summary_header = (
                (experiments_dir / "experiment_summary.csv")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("stage_depths", summary_header)
            self.assertIn("backbone_parameter_count", summary_header)
            self.assertIn("classification_head", summary_header)
            self.assertIn("label_smoothing", summary_header)
            self.assertIn("full_class_parameter_count", summary_header)
            self.assertIn("end_to_end_latency_p95_ms", summary_header)
            self.assertIn("peak_training_cuda_memory_mib", summary_header)
            self.assertIn("margin_warmup_ratio", summary_header)
            self.assertIn("optimizer_step_policy", summary_header)
            self.assertIn("batch_size", summary_header)
            self.assertIn("total_optimizer_steps", summary_header)
            self.assertNotIn("margin_warmup_epochs", summary_header)
            summary_rows = (
                (experiments_dir / "experiment_summary.csv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(summary_rows), 3)

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
