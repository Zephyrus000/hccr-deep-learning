from __future__ import annotations

import unittest

import torch

from hccr.models import EfficientHCCRNet, build_model


class EfficientHCCRNetTests(unittest.TestCase):
    def test_logits_shape(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 11))

    def test_factory_and_parameter_budget(self) -> None:
        model = build_model("efficient_hccr", num_classes=11, width=8)
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()), 100_000
        )
