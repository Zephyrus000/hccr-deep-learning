from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hccr.evaluation.evaluator import evaluate
from hccr.models.heads import FactorizedClassifier
from hccr.training.gate_diagnostics import (
    GateTrainingDiagnostics,
    GateValidationDiagnostics,
    summarize_gate_values,
)
from hccr.training.trainer import train_epoch


class TinyGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.classifier = FactorizedClassifier(
            3,
            3,
            "cosface",
            8.0,
            0.1,
            classifier_structure="dynamic_residual",
            classifier_rank=2,
            residual_class_indices=[0, 1],
        )

    def forward(self, images, targets=None, margin_multiplier=1.0):
        return self.classifier(self.bn(images), targets, margin_multiplier)

    training_logits = forward


class GateDiagnosticsTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_validation_is_detached_and_preserves_rng(self):
        model = TinyGateModel().cuda()
        loader = [(torch.ones(4, 3), torch.tensor([0, 1, 2, 0]), torch.arange(4))]
        rng = torch.cuda.get_rng_state().clone()
        metrics = evaluate(model, loader, "cuda")
        self.assertEqual(metrics["gate_validation"]["samples"], 4)
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        self.assertFalse(model.classifier._forward_hooks)
        json.dumps(metrics, allow_nan=False)

    def test_rank_gather_retains_gate_observations_per_rank(self):
        model = TinyGateModel()
        context = SimpleNamespace(
            is_main_process=False,
            all_gather_object=lambda metrics: [metrics, copy.deepcopy(metrics)],
        )
        metrics = train_epoch(
            model,
            [(torch.ones(4, 3), torch.tensor([0, 1, 2, 0]), [""] * 4)],
            torch.optim.SGD(model.parameters(), lr=1e-3),
            "cpu",
            nn.CrossEntropyLoss(),
            distributed=context,
        )
        self.assertEqual(len(metrics["gate_training"]["ranks"]), 2)
        self.assertEqual(
            metrics["gate_training"]["ranks"][0], metrics["gate_training"]["ranks"][1]
        )
        self.assertEqual(metrics["optimizer_steps"], 1)

    def test_half_gate_is_unsaturated_and_has_no_input_variation(self):
        values = torch.full((4, 2), 0.5)
        result = summarize_gate_values(values, torch.ones_like(values).bool(), [0, 1])
        self.assertEqual(result["fraction_below_005"], 0)
        self.assertEqual(result["fraction_above_095"], 0)
        self.assertEqual(result["normalized_sigmoid_slope"], 1)
        self.assertEqual(result["within_class_variance"], 0)
        self.assertEqual(result["quantiles"]["0.5"], 0.5)

    def test_class_constants_are_balanced_and_not_input_variation(self):
        values = torch.tensor([[0.01, 0.99]]).repeat(4, 1)
        mask = torch.tensor([[True, True], [True, False], [True, False], [True, False]])
        result = summarize_gate_values(values, mask, [0, 1])
        self.assertAlmostEqual(result["mean"], 0.5, places=6)
        self.assertEqual(result["fraction_below_005"], 0.5)
        self.assertEqual(result["fraction_above_095"], 0.5)
        self.assertAlmostEqual(result["within_class_variance"], 0)
        self.assertGreater(result["pooled_variance"], 0.24)
        self.assertEqual([r["count"] for r in result["per_class"]], [4, 1])
        self.assertAlmostEqual(sum(result["histogram_class_balanced_fractions"]), 1)

    def test_switching_gates_have_saturation_and_input_variation(self):
        values = torch.tensor([[0.01, 0.99], [0.99, 0.01]])
        result = summarize_gate_values(values, torch.ones_like(values).bool(), [0, 1])
        self.assertEqual(result["fraction_below_005"], 0.5)
        self.assertEqual(result["fraction_above_095"], 0.5)
        self.assertGreater(result["within_class_variance"], 0.24)

    def test_missing_target_support_is_explicit(self):
        result = summarize_gate_values(
            torch.ones(2, 2), torch.zeros(2, 2).bool(), [0, 1]
        )
        self.assertEqual(result["supported_classes"], 0)
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["per_class"][0]["variance"])
        json.dumps(result, allow_nan=False)

    def test_scoped_observation_is_bounded_and_does_not_change_state(self):
        model = TinyGateModel().eval()
        images = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        expected = model(images).detach()
        state = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        modes = [m.training for m in model.modules()]
        hooks = len(model.classifier._forward_hooks)
        with GateValidationDiagnostics(model, max_samples=3) as diagnostics:
            for _ in range(3):
                diagnostics.set_targets(torch.tensor([0, 2]))
                torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
        summary = diagnostics.summary()
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["target"]["count"], 2)
        self.assertEqual(summary["impostor"]["count"], 4)
        self.assertEqual(summary["target"]["effective_residual_magnitude_mean"], 0)
        self.assertAlmostEqual(
            summary["target"]["absolute_score_change_mean"], 0, places=6
        )
        self.assertEqual(len(model.classifier._forward_hooks), hooks)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for batch in diagnostics.batches:
            self.assertTrue(all(not tensor.requires_grad for tensor in batch))

    def test_exception_removes_hook(self):
        model = TinyGateModel().eval()
        with (
            self.assertRaisesRegex(RuntimeError, "injected"),
            GateValidationDiagnostics(model),
        ):
            raise RuntimeError("injected")
        self.assertFalse(model.classifier._forward_hooks)

    def test_evaluator_integration_is_validation_only_and_writes_json(self):
        model = TinyGateModel()
        images, targets = torch.ones(4, 3), torch.tensor([0, 1, 2, 0])
        rows = {
            "sample_id": ["a", "b", "c", "d"],
            "source_file": ["a.png", "b.png", "c.png", "d.png"],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            validation = evaluate(model, [(images, targets, rows)], "cpu", output)
            report = json.loads((output / "gate_diagnostics.json").read_text())
            self.assertEqual(report, validation["gate_validation"])
            self.assertEqual(report["samples"], 4)
            self.assertIsNotNone(report["panel_identity_sha256"])
        test = evaluate(model, [(images, targets, rows)], "cpu", evaluation_name="test")
        self.assertNotIn("gate_validation", test)
        self.assertEqual(test["top1"], validation["top1"])
        dense = evaluate(nn.Linear(3, 3), [(images, targets, rows)], "cpu")
        self.assertNotIn("gate_validation", dense)

    def test_validation_prefix_is_bounded_and_replays_the_same_panel(self):
        model = TinyGateModel()
        loader = [
            (
                torch.ones(4200, 3),
                torch.zeros(4200, dtype=torch.long),
                torch.arange(4200),
            )
        ]
        first = evaluate(model, loader, "cpu")["gate_validation"]
        second = evaluate(model, loader, "cpu")["gate_validation"]
        self.assertEqual(first["samples"], 4096)
        self.assertEqual(first["target"]["count"], 4096)
        self.assertEqual(
            first["panel_identity_sha256"], second["panel_identity_sha256"]
        )
        self.assertEqual(first, second)

    def test_effective_magnitude_and_score_change_measure_actual_residual(self):
        head = FactorizedClassifier(
            2,
            3,
            "normalized_softmax",
            8.0,
            0.0,
            classifier_structure="static_residual",
            classifier_rank=2,
            residual_class_indices=[0],
        )
        with torch.no_grad():
            head.basis.copy_(torch.eye(2))
            head.coefficients.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            head.residuals.copy_(torch.tensor([[0.0, 1.0]]))
        head.eval()
        with GateValidationDiagnostics(head) as diagnostics:
            diagnostics.set_targets(torch.tensor([0]))
            head(torch.tensor([[0.0, 1.0]]))
        target = diagnostics.summary()["target"]
        self.assertEqual(target["effective_residual_magnitude_mean"], 0.5)
        self.assertAlmostEqual(target["absolute_score_change_mean"], 1 / 5**0.5)

    def test_training_observations_are_total_gradients_and_do_not_change_updates(self):
        torch.manual_seed(7)
        model = TinyGateModel()
        baseline = copy.deepcopy(model)
        loader = [(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]), [""] * 4)] * 4
        initial_rng = torch.get_rng_state().clone()
        metrics = train_epoch(
            model,
            loader,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            "cpu",
            nn.CrossEntropyLoss(),
            gate_regularization=0.1,
        )
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        with patch(
            "hccr.training.trainer.GateTrainingDiagnostics",
            side_effect=lambda m: GateTrainingDiagnostics(m, max_steps=0),
        ):
            train_epoch(
                baseline,
                loader,
                torch.optim.AdamW(baseline.parameters(), lr=1e-3),
                "cpu",
                nn.CrossEntropyLoss(),
                gate_regularization=0.1,
            )
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, baseline.state_dict()[key], rtol=0, atol=0
            )
        report = metrics["gate_training"]["ranks"][0]
        self.assertIn("including_regularization", report["gradient_source"])
        self.assertIn("history_and_decay", report["update_source"])
        self.assertEqual(len(report["observations"]), 3)
        self.assertGreater(report["observations"][0]["gradient_norm"], 0)
        self.assertGreater(report["observations"][0]["update_norm"], 0)
