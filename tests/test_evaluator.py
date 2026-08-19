from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from hccr.evaluation.evaluator import evaluate


class EvaluatorTests(unittest.TestCase):
    def test_streaming_evaluation_writes_diagnostics(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2, bias=False)
        )
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([[1.0], [-1.0]]))
        loader = [
            (
                torch.tensor([[[[1.0]]], [[[-1.0]]]]),
                torch.tensor([0, 1]),
                {
                    "sample_id": ["a", "b"],
                    "source_file": ["a.png", "b.png"],
                },
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            metrics = evaluate(model, loader, "cpu", Path(directory))
            self.assertEqual(metrics["top1"], 1.0)
            self.assertEqual(metrics["top5"], 1.0)
            self.assertEqual(metrics["samples"], 2)
            self.assertTrue((Path(directory) / "per_class_metrics.csv").is_file())
            self.assertTrue((Path(directory) / "validation_errors.csv").is_file())
