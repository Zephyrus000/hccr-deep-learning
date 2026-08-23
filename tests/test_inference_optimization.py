from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from hccr.models import EfficientHCCRNet, optimize_model_for_inference
from hccr.training.diagnostics import _compare_inference_outputs, profile_model


class FixedScaledLogits(nn.Module):
    def __init__(self, perturbation: float = 0.0) -> None:
        super().__init__()
        self.classifier = nn.Identity()
        self.classifier.scale = 32.0
        self.perturbation = perturbation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.arange(8, device=inputs.device, dtype=inputs.dtype) * 3.2
        return logits.expand(len(inputs), -1) + self.perturbation


class InferenceOptimizationTests(unittest.TestCase):
    def test_equivalence_uses_scale_aware_tolerance_and_ranking(self) -> None:
        result = _compare_inference_outputs(
            FixedScaledLogits(),
            FixedScaledLogits(perturbation=0.008),
            image_size=16,
            device="cpu",
        )

        self.assertFalse(result["strict_logit_allclose_passed"])
        self.assertTrue(result["normalized_logit_allclose_passed"])
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertEqual(result["top5_agreement"], 1.0)
        self.assertTrue(result["passed"])

    def test_equivalence_rejects_excessive_normalized_logit_drift(self) -> None:
        result = _compare_inference_outputs(
            FixedScaledLogits(),
            FixedScaledLogits(perturbation=0.05),
            image_size=16,
            device="cpu",
        )

        self.assertFalse(result["normalized_logit_allclose_passed"])
        self.assertFalse(result["passed"])

    def test_eval_caches_normalized_classifier_weight(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8).eval()
        cached_pointer = model.classifier._normalized_weight.data_ptr()
        inputs = torch.rand(2, 1, 32, 32)

        first = model(inputs)
        second = model(inputs)

        self.assertEqual(model.classifier._normalized_weight.data_ptr(), cached_pointer)
        torch.testing.assert_close(first, second)

    def test_optimized_copy_preserves_logits_and_folds_batch_norm(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8).eval()
        optimized = optimize_model_for_inference(model)
        inputs = torch.rand(2, 1, 32, 32)

        with torch.inference_mode():
            expected = model(inputs)
            actual = optimized(inputs)

        torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)
        self.assertFalse(
            any(isinstance(module, nn.BatchNorm2d) for module in optimized.modules())
        )
        self.assertIsInstance(optimized.embedding_dropout, nn.Identity)

    def test_multibranch_depthwise_fuses_to_one_spatial_kernel(self) -> None:
        model = EfficientHCCRNet(
            num_classes=11,
            width=8,
            dropout=0.0,
            reparameterize_depthwise=True,
        ).eval()
        optimized = optimize_model_for_inference(model)
        baseline_optimized = optimize_model_for_inference(
            EfficientHCCRNet(num_classes=11, width=8, dropout=0.0).eval()
        )
        inputs = torch.rand(2, 1, 32, 32)

        with torch.inference_mode():
            expected = model(inputs)
            actual = optimized(inputs)

        torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)
        self.assertGreater(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in optimized.parameters()),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in optimized.parameters()),
            sum(parameter.numel() for parameter in baseline_optimized.parameters()),
        )
        for block in optimized.features:
            self.assertEqual(len(block.depthwise_branches), 0)
            self.assertIsInstance(block.block[0], nn.Conv2d)
            self.assertEqual(block.block[0].kernel_size, (3, 3))
            self.assertIsInstance(block.block[3], nn.Conv2d)
            self.assertEqual(block.block[3].kernel_size, (1, 1))

    def test_profile_keeps_eager_and_adds_optimized_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = profile_model(
                EfficientHCCRNet(num_classes=11, width=4),
                image_size=16,
                device="cpu",
                output_dir=Path(directory),
                warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
            )

        self.assertIn("inference_benchmarks", profile)
        self.assertTrue(profile["optimized_inference"]["equivalence"]["passed"])
        self.assertEqual(len(profile["optimized_inference"]["benchmarks"]), 3)
        self.assertIn("parameter_count", profile["optimized_inference"])
        self.assertIn("estimated_macs", profile["optimized_inference"])
        self.assertIn("full_class_projection", profile["optimized_inference"])
