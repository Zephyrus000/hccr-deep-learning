from __future__ import annotations

import unittest
from pathlib import Path

from hccr.training.workflow import (
    TrainingConfig,
    _build_batch_training_plan,
    _cosine_warmup_factor,
    _resolved_early_stopping_patience,
)


class BatchTrainingPlanTests(unittest.TestCase):
    def _config(self, **overrides) -> TrainingConfig:
        values = {
            "manifest_path": Path("manifest.csv"),
            "output_dir": Path("experiments"),
            "num_classes": 1000,
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "scheduler_min_lr": 1e-6,
            "early_stopping_patience": 8,
        }
        values.update(overrides)
        return TrainingConfig(**values)

    def test_reference_batch_policy_preserves_optimizer_update_budget(self) -> None:
        plan = _build_batch_training_plan(
            self._config(), train_samples=1024, steps_per_epoch=4
        )

        self.assertEqual(plan.reference_steps_per_epoch, 16)
        self.assertEqual(plan.total_optimizer_steps, 320)
        self.assertEqual(plan.resolved_epochs, 80)
        self.assertEqual(plan.warmup_steps, 16)
        self.assertAlmostEqual(plan.resolved_learning_rate, 6e-4)
        self.assertAlmostEqual(plan.resolved_scheduler_min_lr, 2e-6)
        self.assertEqual(_resolved_early_stopping_patience(self._config(), plan), 32)

    def test_configured_epoch_policy_preserves_legacy_update_budget(self) -> None:
        plan = _build_batch_training_plan(
            self._config(optimizer_step_policy="configured_epochs"),
            train_samples=1024,
            steps_per_epoch=4,
        )

        self.assertEqual(plan.total_optimizer_steps, 80)
        self.assertEqual(plan.resolved_epochs, 20)

    def test_distributed_plan_scales_against_effective_global_batch(self) -> None:
        plan = _build_batch_training_plan(
            self._config(
                optimizer_step_policy="configured_epochs",
                learning_rate_scaling="linear",
                reference_batch_size=256,
            ),
            train_samples=1024,
            steps_per_epoch=2,
            world_size=2,
        )

        self.assertEqual(plan.world_size, 2)
        self.assertEqual(plan.effective_global_batch_size, 512)
        self.assertAlmostEqual(plan.resolved_learning_rate, 6e-4)

    def test_cosine_warmup_reaches_base_then_minimum_learning_rate(self) -> None:
        self.assertAlmostEqual(_cosine_warmup_factor(0, 320, 16, 0.01), 1 / 16)
        self.assertAlmostEqual(_cosine_warmup_factor(16, 320, 16, 0.01), 1.0)
        self.assertAlmostEqual(_cosine_warmup_factor(320, 320, 16, 0.01), 0.01)
