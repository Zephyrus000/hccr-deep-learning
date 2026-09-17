from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hccr.models import EfficientHCCRNet, build_model
from hccr.preprocessing import EvalPreprocessor
from hccr.training.complexity import estimate_flops_by_component
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
            self.assertGreater(profile["estimated_flops"], 0)
            self.assertTrue(profile["flop_coverage"]["complete"])
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
        self.assertEqual(profile["head_parameter_count"], 256_000)
        self.assertEqual(
            profile["parameter_count"],
            profile["backbone_parameter_count"] + profile["head_parameter_count"],
        )
        self.assertEqual(
            profile["estimated_macs"],
            profile["estimated_backbone_macs"] + profile["estimated_head_macs"],
        )
        self.assertEqual(
            profile["estimated_flops"],
            profile["estimated_backbone_flops"] + profile["estimated_head_flops"],
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
            sum(parameter.numel() for parameter in full.parameters()), 2_016_960
        )

    def test_profile_separates_embedding_projection_from_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = profile_model(
                EfficientHCCRNet(
                    num_classes=1000,
                    width=64,
                    backbone_output_channels=320,
                    embedding_dim=160,
                ),
                image_size=16,
                device="cpu",
                output_dir=Path(directory),
                warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
                full_class_num_classes=7186,
            )
        self.assertEqual(profile["backbone_output_channels"], 320)
        self.assertEqual(profile["embedding_dim"], 160)
        self.assertEqual(profile["embedding_projection_parameter_count"], 51_200)
        self.assertEqual(profile["classifier_parameter_count"], 160_000)
        self.assertEqual(profile["head_parameter_count"], 211_200)
        self.assertEqual(
            profile["head_parameter_count"],
            profile["embedding_projection_parameter_count"]
            + profile["classifier_parameter_count"],
        )
        full_class = profile["full_class_projection"]
        self.assertEqual(full_class["embedding_projection_parameter_count"], 51_200)
        self.assertEqual(full_class["classifier_parameter_count"], 1_149_760)
        self.assertEqual(full_class["head_parameter_count"], 1_200_960)
        direct_full_flops, _coverage = estimate_flops_by_component(
            EfficientHCCRNet(
                num_classes=7186,
                width=64,
                backbone_output_channels=320,
                embedding_dim=160,
            ),
            16,
            "cpu",
        )
        self.assertEqual(full_class["total_flops"], direct_full_flops["total"])

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
        self.assertTrue(profile["flop_coverage"]["complete"])
        self.assertIn(
            "aten._native_batch_norm_legit_no_training",
            profile["flop_coverage"]["counted_operator_types"],
        )
        self.assertEqual(profile["full_class_projection"]["status"], "available")
        self.assertEqual(profile["full_class_projection"]["num_classes"], 100)
        self.assertEqual(
            profile["full_class_projection"]["head_parameter_count"], 3_200
        )
        self.assertEqual(
            profile["end_to_end_batch1_benchmark"]["scope"],
            "synthetic_raw_pil_to_logits",
        )


class RetainedInputCostTests(unittest.TestCase):
    def test_article_models_have_complete_operator_flop_coverage(self) -> None:
        specifications = (
            ("resnet18", {}),
            ("mobilenet_v3_small", {}),
            ("shufflenet_v2_x1_0", {}),
            ("efficientnet_b0", {}),
            (
                "efficient_hccr",
                {
                    "width": 8,
                    "stage_depths": (2, 3, 3),
                    "reparameterize_depthwise": False,
                },
            ),
            (
                "efficient_hccr",
                {
                    "width": 8,
                    "stage_depths": (2, 3, 3),
                    "reparameterize_depthwise": True,
                },
            ),
        )
        for model_name, options in specifications:
            with self.subTest(model=model_name, options=options):
                model = build_model(
                    model_name,
                    num_classes=11,
                    classification_head="cosface",
                    **options,
                )
                _flops, coverage = estimate_flops_by_component(model, 32, "cpu")
                self.assertTrue(coverage["complete"])
                self.assertEqual(coverage["unsupported_operator_types"], [])

    def test_operator_flops_include_non_mac_inference_work(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(2, 3),
        )
        flops, coverage = estimate_flops_by_component(model, 4, "cpu")
        self.assertEqual(flops["total"], 787)
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["unsupported_operator_types"], [])
        self.assertEqual(
            coverage["flops_by_operator"]["aten._native_batch_norm_legit_no_training"],
            132,
        )

    def test_grayscale_model_has_no_input_adapter_macs(self) -> None:
        image_size = 32
        model = EfficientHCCRNet(num_classes=11, width=8)
        macs = estimate_macs_by_component(model, image_size, "cpu")
        self.assertEqual(macs["input_adapter"], 0)
        self.assertEqual(macs["total"], macs["backbone"] + macs["head"])
