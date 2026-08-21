from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from hccr.evaluation.diagnostics import (
    summarize_recall_by_support,
    support_tiers,
)
from hccr.evaluation.evaluator import evaluate


class EvaluatorTests(unittest.TestCase):
    def test_streaming_evaluation_writes_diagnostics(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2, bias=False)
        )
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
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
            output = Path(directory)
            metrics = evaluate(
                model,
                loader,
                "cpu",
                output,
                labels={0: "A", 1: "B"},
            )
            self.assertEqual(metrics["top1"], 0.0)
            self.assertEqual(metrics["top5"], 1.0)
            self.assertEqual(metrics["samples"], 2)
            self.assertTrue((output / "per_class_metrics.csv").is_file())
            with (output / "validation_errors.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                errors = list(csv.DictReader(file))
            self.assertEqual(errors[0]["target_label"], "A")
            self.assertEqual(errors[0]["prediction_label"], "B")

    def test_support_tiers_and_macro_recall_use_training_support(self) -> None:
        training_support = {0: 5, 1: 10, 2: 20, 3: 40, 4: 80}
        self.assertEqual(
            support_tiers(training_support),
            {0: "tail", 1: "mid", 2: "mid", 3: "mid", 4: "head"},
        )
        recalls = summarize_recall_by_support(
            {class_id: 2 for class_id in training_support},
            {0: 0, 1: 1, 2: 2, 3: 1, 4: 2},
            training_support,
        )
        self.assertEqual(recalls["macro_recall"], 0.6)
        self.assertEqual(recalls["tail_recall"], 0.0)
        self.assertEqual(recalls["mid_recall"], 2 / 3)
        self.assertEqual(recalls["head_recall"], 1.0)

    def test_streaming_diagnostics_write_support_tiers_and_confusion_pairs(
        self,
    ) -> None:
        model = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2, bias=False)
        )
        with torch.no_grad():
            model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        loader = [
            (
                torch.tensor([[[[1.0]]], [[[-1.0]]]]),
                torch.tensor([0, 1]),
                {"sample_id": ["a", "b"], "source_file": ["a.png", "b.png"]},
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = evaluate(
                model,
                loader,
                "cpu",
                output,
                labels={0: "A", 1: "B"},
                class_support={0: 1, 1: 10},
            )
            self.assertEqual(metrics["macro_recall"], 0.0)
            self.assertEqual(metrics["tail_recall"], 0.0)
            self.assertEqual(metrics["head_recall"], 0.0)
            self.assertTrue((output / "class_tiers.json").is_file())
            with (output / "confusion_pairs.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                pairs = list(csv.DictReader(file))
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["count"], "1")
