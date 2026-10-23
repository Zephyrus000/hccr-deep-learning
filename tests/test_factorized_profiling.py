from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from hccr.benchmarking.complexity import estimate_flops_by_component
from hccr.benchmarking.profiling import (
    estimate_macs_by_component,
    profile_model,
)
from hccr.models import EfficientHCCRNet, optimize_model_for_inference


class FactorizedProfilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self, structure="dynamic_residual", gate_kind="hadamard"):
        return EfficientHCCRNet(
            num_classes=7,
            width=4,
            embedding_dim=8,
            classifier_structure=structure,
            classifier_rank=3,
            residual_class_indices=() if structure == "low_rank" else (1, 4),
            gate_kind=gate_kind,
        )

    def test_operator_coverage_and_classifier_macs_for_all_variants(self):
        for structure, gate, expected in (
            ("low_rank", "hadamard", 45),
            ("static_residual", "hadamard", 61),
            ("dynamic_residual", "hadamard", 91),
            ("dynamic_residual", "cosine", 61),
            ("dynamic_residual", "sample", 69),
            ("dynamic_residual", "global", 61),
            ("dynamic_residual", "class", 61),
        ):
            with self.subTest(structure=structure, gate=gate):
                model = self.make_model(structure, gate)
                macs = estimate_macs_by_component(model, 32, "cpu")
                flops, coverage = estimate_flops_by_component(model, 32, "cpu")
                self.assertEqual(macs["classifier"], expected)
                self.assertTrue(coverage["complete"], coverage)
                self.assertGreater(flops["classifier"], 2 * expected)

    def test_profile_reports_cache_storage_and_restores_training_mode(self):
        model = self.make_model()
        model.train()
        with tempfile.TemporaryDirectory() as folder:
            profile = profile_model(
                model, 32, "cpu", Path(folder), 1, 1, 1, full_class_num_classes=100
            )
        self.assertTrue(model.training)
        self.assertTrue(model.classifier.training)
        self.assertIsNone(model.classifier._deployment_cache)
        self.assertEqual(profile["estimated_classifier_macs"], 91)
        self.assertEqual(
            profile["classifier_deployment_cache_bytes"], 7 * 4 + 3 * 2 * 8
        )
        self.assertEqual(profile["full_class_projection"]["status"], "unavailable")
        self.assertTrue(profile["flop_coverage"]["complete"])
        self.assertNotIn("optimized_inference", profile)
        optimized = optimize_model_for_inference(model)
        model.eval()
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), optimized(inputs))

    def test_degenerate_fallback_work_is_counted(self):
        model = self.make_model("static_residual")
        head = model.classifier
        with torch.no_grad():
            q = torch.nn.functional.normalize(head.coefficients @ head.basis, dim=1)
            head.residuals.copy_(-2 * q[head.residual_indices])
        macs = estimate_macs_by_component(model, 32, "cpu")
        self.assertEqual(macs["classifier"], 61 + 7 * 3 * 8 + 2 * 8)
        flops, coverage = estimate_flops_by_component(model, 32, "cpu")
        self.assertTrue(coverage["complete"], coverage)
        self.assertGreater(flops["classifier"], 2 * 61)

    def test_dense_profile_includes_normalized_weight_cache_storage(self):
        model = EfficientHCCRNet(num_classes=7, width=4, embedding_dim=8)
        with tempfile.TemporaryDirectory() as folder:
            profile = profile_model(model, 32, "cpu", Path(folder), 1, 1, 1)
        self.assertEqual(profile["classifier_deployment_cache_bytes"], 7 * 8 * 4)
        self.assertIsNone(model.classifier._normalized_weight)

    def test_cache_construction_is_not_traced_as_per_image_work(self):
        model = self.make_model()
        first, _ = estimate_flops_by_component(model, 32, "cpu")
        second, _ = estimate_flops_by_component(model, 32, "cpu")
        self.assertEqual(first, second)
        with torch.no_grad():
            model.classifier.basis.add_(0.01)
        third, _ = estimate_flops_by_component(model, 32, "cpu")
        self.assertEqual(second, third)


if __name__ == "__main__":
    unittest.main()
