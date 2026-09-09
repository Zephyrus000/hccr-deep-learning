from __future__ import annotations

import unittest

import torch

from hccr.models import EfficientHCCRNet, build_model
from hccr.training.diagnostics import (
    _parameter_bytes_by_component,
    _parameter_counts_by_component,
    estimate_macs_by_component,
    project_classifier_cost,
)


class ReferenceBaselineTests(unittest.TestCase):
    def test_reference_baselines_accept_grayscale_and_return_class_logits(self) -> None:
        for name in ("resnet18", "mobilenet_v3_small"):
            with self.subTest(name=name):
                model = build_model(name, num_classes=11).eval()
                with torch.inference_mode():
                    logits = model(torch.rand(2, 1, 64, 64))
                self.assertEqual(logits.shape, (2, 11))
                self.assertEqual(model.effective_input_channels, 1)
                self.assertEqual(model.classification_head, "softmax")

    def test_resnet_stem_is_one_channel(self) -> None:
        model = build_model("resnet18", num_classes=11)
        self.assertEqual(model.backbone.conv1.in_channels, 1)

    def test_mobilenet_stem_is_one_channel(self) -> None:
        model = build_model("mobilenet_v3_small", num_classes=11)
        self.assertEqual(model.backbone.features[0][0].in_channels, 1)

    def test_mobilenet_full_class_projection_keeps_its_hidden_classifier(self) -> None:
        model = build_model("mobilenet_v3_small", num_classes=11).eval()
        counts = _parameter_counts_by_component(model)
        bytes_by_component = _parameter_bytes_by_component(model)
        macs = estimate_macs_by_component(model, image_size=64, device="cpu")
        projection = project_classifier_cost(
            model, counts, bytes_by_component, macs, 100
        )
        final_linear = model.classifier[-1]
        expected_final_parameters = final_linear.in_features * 100 + 100
        self.assertEqual(
            projection["classifier_parameter_count"], expected_final_parameters
        )
        self.assertEqual(
            projection["total_parameter_count"],
            counts["total"]
            - sum(parameter.numel() for parameter in final_linear.parameters())
            + expected_final_parameters,
        )


class SoftmaxHeadTests(unittest.TestCase):
    def test_proposed_softmax_head_does_not_apply_an_angular_margin(self) -> None:
        model = EfficientHCCRNet(
            num_classes=5,
            width=8,
            dropout=0.0,
            classification_head="softmax",
        ).eval()
        inputs = torch.randn(3, 1, 64, 64)
        targets = torch.tensor([0, 2, 4])
        torch.testing.assert_close(
            model(inputs), model.training_logits(inputs, targets, 1.0)
        )
