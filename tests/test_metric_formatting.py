from __future__ import annotations

import unittest

from hccr.training.workflow import (
    BatchTrainingPlan,
    _format_epoch_metrics,
    _format_final_metrics,
)


class MetricFormattingTests(unittest.TestCase):
    def test_epoch_summary_names_test_split_and_core_metrics(self) -> None:
        plan = BatchTrainingPlan(
            configured_epochs=2,
            resolved_epochs=8,
            batch_size=256,
            steps_per_epoch=4,
            reference_steps_per_epoch=16,
            total_optimizer_steps=32,
            reference_batch_size=64,
            optimizer_step_policy="reference_batch",
            learning_rate_scaling="sqrt",
            base_learning_rate=3e-4,
            resolved_learning_rate=6e-4,
            resolved_scheduler_min_lr=2e-6,
            warmup_steps=2,
        )
        summary = _format_epoch_metrics(
            2,
            plan,
            {
                "train_loss": 0.125,
                "train_samples_per_second": 123.4,
            },
            {
                "top1": 0.91,
                "top5": 0.99,
                "macro_recall": 0.9,
                "tail_recall": 0.88,
            },
            4e-4,
            8,
            0.91,
            "test",
        )

        self.assertIn("updates=8/32", summary)
        self.assertIn("test_top1=91.00%", summary)
        self.assertIn("macro_recall=90.00%", summary)
        self.assertIn("tail_recall=88.00%", summary)

    def test_final_summary_includes_calibration_when_available(self) -> None:
        summary = _format_final_metrics(
            {
                "top1": 0.91,
                "top5": 0.99,
                "macro_recall": 0.9,
                "tail_recall": 0.88,
                "expected_calibration_error": 0.0125,
            },
            "test",
        )

        self.assertEqual(
            summary,
            "test_top1=91.00% | test_top5=99.00% | macro_recall=90.00% | "
            "tail_recall=88.00% | ece=1.25%",
        )
