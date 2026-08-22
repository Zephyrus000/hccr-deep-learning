from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from hccr.models import EfficientHCCRNet, optimize_model_for_inference
from hccr.training.diagnostics import profile_model


class InferenceOptimizationTests(unittest.TestCase):
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
