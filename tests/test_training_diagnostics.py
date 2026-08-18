from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hccr.training.diagnostics import profile_model, write_training_diagnostics
from hccr.training.trainer import train_epoch


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_profile_and_epoch_diagnostics_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 2))
            profile = profile_model(
                model, image_size=4, device="cpu", output_dir=output
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
            self.assertGreater(profile["estimated_macs"], 0)
            self.assertGreater(epoch["gradient_norm_max"], 0)
            self.assertIn("stages", epoch)
            self.assertTrue((output / "resource_profile.json").exists())
            self.assertIn(
                "epochs", json.loads((output / "training_diagnostics.json").read_text())
            )
