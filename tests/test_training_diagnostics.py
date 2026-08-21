from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hccr.models import EfficientHCCRNet
from hccr.preprocessing import EvalPreprocessor
from hccr.training.diagnostics import (
    estimate_macs_by_component,
    profile_model,
    recalibrate_batch_norm,
    summarize_batch_norm_state,
    summarize_validation_stability,
    write_training_diagnostics,
)
from hccr.training.losses import build_classification_loss
from hccr.training.trainer import train_epoch


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_profile_and_epoch_diagnostics_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 2))
            profile = profile_model(
                model,
                image_size=4,
                device="cpu",
                output_dir=output,
                warmup_iterations=1,
                benchmark_iterations=3,
                benchmark_repetitions=2,
            )
            loader = DataLoader(
                TensorDataset(torch.rand(4, 1, 4, 4), torch.tensor([0, 1, 0, 1])),
                batch_size=2,
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            epoch = train_epoch(
                model,
                (
                    (
                        images,
                        targets,
                        {"applied_augmentations": ["elastic"] * len(targets)},
                    )
                    for images, targets in loader
                ),
                optimizer,
                "cpu",
                build_classification_loss(),
            )
            write_training_diagnostics(output, [{"epoch": 1.0, **epoch}])
            self.assertGreater(profile["parameter_count"], 0)
            self.assertEqual(len(profile["inference_benchmarks"]), 3)
            self.assertEqual(profile["benchmark_protocol"]["repetitions"], 2)
            self.assertEqual(
                len(profile["inference_benchmarks"][0]["repeat_summaries"]), 2
            )
            self.assertIn("intra_op_threads", profile["device_metadata"])
            self.assertGreater(profile["estimated_macs"], 0)
            self.assertGreater(epoch["gradient_norm_max"], 0)
            self.assertIn("stages", epoch)
            self.assertEqual(epoch["augmentation_counts"], {"elastic": 4})
            self.assertEqual(epoch["augmentation_rates"], {"elastic": 1.0})
            self.assertTrue((output / "resource_profile.json").exists())
            self.assertIn(
                "epochs", json.loads((output / "training_diagnostics.json").read_text())
            )

    def test_batch_norm_recalibration_updates_only_running_statistics(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, padding=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.Dropout(0.5),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(2, 2),
        )
        model.eval()
        loader = [
            (torch.full((2, 1, 4, 4), value), torch.zeros(2), {})
            for value in (0.25, 0.75, 1.0)
        ]
        result = recalibrate_batch_norm(model, loader, "cpu", max_batches=2)
        state = summarize_batch_norm_state(model)
        self.assertFalse(model.training)
        self.assertFalse(model[2].training)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(state["layer_count"], 1)
        self.assertEqual(state["layers"]["1"]["num_batches_tracked"], 2)
        self.assertEqual(state["non_finite_layer_count"], 0)

    def test_validation_stability_detects_collapsed_epoch(self) -> None:
        stability = summarize_validation_stability(
            [
                {"epoch": 1.0, "top1": 0.65},
                {"epoch": 2.0, "top1": 0.85},
                {"epoch": 3.0, "top1": 0.61},
                {"epoch": 4.0, "top1": 0.90},
            ],
            drop_threshold=0.05,
        )
        self.assertAlmostEqual(stability["largest_single_epoch_drop"], 0.24)
        self.assertAlmostEqual(stability["max_drawdown_from_prior_best"], 0.24)
        self.assertEqual(stability["unstable_epoch_count"], 1)
        self.assertEqual(stability["best_epoch"], 4)


class ModelCostBreakdownTests(unittest.TestCase):
    def test_backbone_and_head_costs_sum_to_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = EfficientHCCRNet(num_classes=1000, width=64)
            profile = profile_model(
                model,
                image_size=16,
                device="cpu",
                output_dir=Path(directory),
                warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
            )
        self.assertEqual(profile["backbone_parameter_count"], 177_344)
        self.assertEqual(profile["head_parameter_count"], 257_000)
        self.assertEqual(
            profile["parameter_count"],
            profile["backbone_parameter_count"] + profile["head_parameter_count"],
        )
        self.assertEqual(
            profile["estimated_macs"],
            profile["estimated_backbone_macs"] + profile["estimated_head_macs"],
        )

    def test_full_class_growth_is_isolated_to_classifier(self) -> None:
        subset = EfficientHCCRNet(num_classes=1000, width=64)
        full = EfficientHCCRNet(num_classes=7186, width=64)
        subset_backbone = sum(
            parameter.numel()
            for name, parameter in subset.named_parameters()
            if not name.startswith("classifier.")
        )
        full_backbone = sum(
            parameter.numel()
            for name, parameter in full.named_parameters()
            if not name.startswith("classifier.")
        )
        self.assertEqual(subset_backbone, full_backbone)
        self.assertEqual(
            sum(parameter.numel() for parameter in full.parameters()), 2_024_146
        )

    def test_profile_declares_coverage_end_to_end_and_full_head_projection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = profile_model(
                EfficientHCCRNet(num_classes=10, width=8),
                image_size=16,
                device="cpu",
                output_dir=Path(directory),
                warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
                preprocessing_transform=EvalPreprocessor(image_size=16),
                full_class_num_classes=100,
            )
        self.assertFalse(profile["mac_coverage"]["complete"])
        self.assertIn(
            "BatchNorm2d", profile["mac_coverage"]["unsupported_operator_types"]
        )
        self.assertEqual(profile["full_class_projection"]["status"], "available")
        self.assertEqual(profile["full_class_projection"]["num_classes"], 100)
        self.assertEqual(
            profile["full_class_projection"]["head_parameter_count"], 3_300
        )
        self.assertEqual(
            profile["end_to_end_batch1_benchmark"]["scope"],
            "synthetic_raw_pil_to_logits",
        )


class AttentionCostTests(unittest.TestCase):
    def test_eca_conv1d_is_included_in_backbone_macs(self) -> None:
        with tempfile.TemporaryDirectory() as baseline_directory:
            baseline = profile_model(
                EfficientHCCRNet(num_classes=1000, width=64),
                16,
                "cpu",
                Path(baseline_directory),
                1,
                1,
                1,
            )
        with tempfile.TemporaryDirectory() as eca_directory:
            eca = profile_model(
                EfficientHCCRNet(num_classes=1000, width=64, attention="eca"),
                16,
                "cpu",
                Path(eca_directory),
                1,
                1,
                1,
            )
        self.assertEqual(eca["backbone_parameter_count"], 177_347)
        self.assertEqual(
            eca["estimated_backbone_macs"] - baseline["estimated_backbone_macs"],
            768,
        )


class DirectionalInputCostTests(unittest.TestCase):
    def test_fixed_filter_macs_are_included_in_backbone_total(self) -> None:
        image_size = 32
        sobel = EfficientHCCRNet(num_classes=11, width=8, input_mode="grayscale_sobel")
        gabor = EfficientHCCRNet(num_classes=11, width=8, input_mode="grayscale_gabor")
        sobel_macs = estimate_macs_by_component(sobel, image_size, "cpu")
        gabor_macs = estimate_macs_by_component(gabor, image_size, "cpu")
        self.assertEqual(
            sobel_macs["input_adapter"], image_size * image_size * 2 * 3 * 3
        )
        self.assertEqual(
            gabor_macs["input_adapter"], image_size * image_size * 4 * 7 * 7
        )
        self.assertEqual(
            sobel_macs["total"], sobel_macs["backbone"] + sobel_macs["head"]
        )
        self.assertGreater(gabor_macs["total"], sobel_macs["total"])
