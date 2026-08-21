from __future__ import annotations

import unittest

import torch

from hccr.training.losses import build_classification_loss
from hccr.training.workflow import _margin_multiplier


class ClassificationLossTests(unittest.TestCase):
    def test_cross_entropy_supports_explicit_label_smoothing(self) -> None:
        criterion = build_classification_loss(0.05)
        loss = criterion(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]), torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(criterion.label_smoothing, 0.05)

    def test_invalid_label_smoothing_is_rejected(self) -> None:
        for value in (-0.1, 1.0):
            with self.assertRaisesRegex(ValueError, "label_smoothing"):
                build_classification_loss(value)

    def test_margin_warmup_reaches_full_strength_on_final_epoch(self) -> None:
        self.assertEqual(
            [_margin_multiplier(epoch, 3, "arcface") for epoch in (1, 2, 3)],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(_margin_multiplier(1, 3, "linear"), 1.0)
