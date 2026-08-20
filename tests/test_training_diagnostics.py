from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hccr.training.diagnostics import (
    profile_model,
    recalibrate_batch_norm,
    summarize_batch_norm_state,
    summarize_validation_stability,
    write_training_diagnostics,
)
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
                ((images, targets, {}) for images, targets in loader),
                optimizer,
                "cpu",
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
