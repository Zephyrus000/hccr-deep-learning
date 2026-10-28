"""Behavior checks for the direct counterfactual reference used by the runner."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from hccr.models.heads import FactorizedClassifier

_PATH = Path(__file__).resolve().parents[1] / "scripts/run_local_post_training.py"
_SPEC = importlib.util.spec_from_file_location("hccr_local_post_training_tests", _PATH)
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


class LocalPostTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_head(self, structure="dynamic_residual"):
        torch.manual_seed(7)
        head = FactorizedClassifier(
            8, 7, "cosface", 32, 0.1, classifier_structure=structure,
            classifier_rank=3, residual_class_indices=[1, 3, 5],
        )
        with torch.no_grad():
            head.residuals.normal_(std=0.1)
            if structure == "dynamic_residual":
                head.gate_weight.normal_(std=0.2)
        return head.eval()

    def test_direct_dynamic_prototypes_match_factorized_forward(self):
        head = self.make_head()
        with torch.no_grad():
            x = torch.randn(6, 8)
            torch.testing.assert_close(_RUNNER.explicit_head(head, x), head(x), rtol=1e-5, atol=1e-5)

    def test_static_reference_uses_the_implemented_half_gate(self):
        head = self.make_head("static_residual")
        with torch.no_grad():
            x = torch.randn(6, 8)
            torch.testing.assert_close(_RUNNER.explicit_head(head, x), head(x), rtol=1e-5, atol=1e-5)

    def test_zero_gate_matches_normalized_low_rank_prototypes(self):
        head = self.make_head()
        x = torch.randn(6, 8)
        with torch.no_grad():
            z = torch.nn.functional.normalize(x, dim=1)
            q = torch.nn.functional.normalize(head.coefficients @ head.basis, dim=1)
            expected = (z @ q.T).clamp(-1 + 1e-7, 1 - 1e-7) * 32
            actual = _RUNNER.explicit_head(head, x, torch.zeros(6, 3))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_classification_gradient_observation_does_not_update_weights(self):
        head = self.make_head()
        before = {name: value.clone() for name, value in head.state_dict().items()}
        logits = head(torch.randn(6, 8), targets=torch.tensor([0, 1, 2, 3, 4, 5]))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 4, 5]))
        gradients = torch.autograd.grad(loss, [p for n, p in head.named_parameters() if n.startswith("gate_")])
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        for name, value in head.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_windows_entrypoint_records_environment_before_inventory_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            with (
                patch("sys.argv", [str(_PATH), "--output", str(output), "--statistics-only"]),
                patch.object(_RUNNER, "inventory", side_effect=FileNotFoundError("controlled missing inventory")),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = _RUNNER.main()
            self.assertEqual(result, 2)
            environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
            self.assertGreater(environment["disk"]["total"], 0)
            state = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["blocked_stages"], ["input_inventory"])

    def test_onnx_equivalence_rejects_bad_scores_and_nonfinite_outputs(self):
        reference = np.array([[1.0, 0.0]], dtype=np.float32)
        observation = _RUNNER.onnx_score_equivalence(reference, np.array([[1.01, 0.0]], dtype=np.float32))
        self.assertFalse(observation["allclose"])
        self.assertEqual(observation["violating_scores"], 1)
        self.assertEqual(observation["top1_disagreements"], 0)
        observation = _RUNNER.onnx_score_equivalence(reference, np.array([[np.inf, 0.0]], dtype=np.float32))
        self.assertFalse(observation["finite"])
        self.assertFalse(observation["allclose"])
        with self.assertRaisesRegex(ValueError, "shape"):
            _RUNNER.onnx_score_equivalence(reference, np.zeros((2, 2), dtype=np.float32))

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                         "optional ONNX deployment dependencies are unavailable")
    def test_onnx_export_preserves_batch_norm_and_dynamic_batch_scores(self):
        import onnx
        import onnxruntime as ort
        torch.manual_seed(29)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(3), torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4),
        ).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        images = torch.randn(40, 1, 8, 8)
        with torch.no_grad():
            targets = model(images).argmax(1)
        panels = {"validation": {"images": images, "targets": targets},
                  "fit": {"images": images[:8]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _RUNNER.onnx_dense_control(SimpleNamespace(output=output, threads=1, batch=32),
                                       {"seed": "29", "variant": "dense_reference"}, model, panels)
            folder = output / "onnx"
            graph = onnx.load(folder / "seed-29-dense_reference-fp32.onnx")
            self.assertEqual(sum(n.op_type == "BatchNormalization" for n in graph.graph.node), 1)
            self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
            receipt = json.loads((folder / "seed-29-dense_reference-equivalence.json").read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual([p["samples"] for p in receipt["probes"]], [1, 32])
            self.assertEqual(sum(p["samples"] for p in receipt["panel_batches"]), 40)
            self.assertEqual(receipt["panel_top1_disagreements"], 0)
            report = json.loads((folder / "seed-29-dense_reference-report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["models"][0]["validation_panel_top1"], 1.0)
            self.assertEqual(_RUNNER.onnx_session_options(1).graph_optimization_level,
                             ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
            self.assertEqual(_RUNNER.onnx_session_options(1, quantized=True).graph_optimization_level,
                             ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
