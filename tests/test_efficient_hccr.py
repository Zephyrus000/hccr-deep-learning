from __future__ import annotations

import unittest

import torch

from hccr.models import EfficientHCCRNet, build_model
from hccr.models.efficient_hccr import AngularMarginClassifier


class EfficientHCCRNetTests(unittest.TestCase):
    def test_logits_shape(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 11))

    def test_factory_and_parameter_budget(self) -> None:
        model = build_model("efficient_hccr", num_classes=11, width=8)
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()), 100_000
        )


class EfficientHCCRArchitectureTests(unittest.TestCase):
    def test_named_stage_boundaries_preserve_expected_shapes(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8, stage_depths=(1, 2, 3))
        features, stages = model.forward_features(
            torch.randn(2, 1, 64, 64), return_stages=True
        )
        self.assertEqual(model.stage_ranges, ((0, 1), (1, 3), (3, 6)))
        self.assertEqual(stages["stage1"].shape, (2, 8, 32, 32))
        self.assertEqual(stages["stage2"].shape, (2, 16, 16, 16))
        self.assertEqual(stages["stage3"].shape, (2, 32, 8, 8))
        self.assertEqual(features.shape, stages["stage3"].shape)

    def test_default_checkpoint_keys_and_parameter_budget_stay_stable(self) -> None:
        model = EfficientHCCRNet(num_classes=1000, width=64)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()), 433_344
        )
        self.assertIn("features.0.block.0.weight", model.state_dict())
        inputs = torch.randn(2, 1, 64, 64)
        model.eval()
        with torch.inference_mode():
            embeddings = torch.flatten(
                model.pool(model.features(model.stem(inputs))), 1
            )
            legacy_path = model.classifier(model.embedding_dropout(embeddings))
            staged_path = model(inputs)
        torch.testing.assert_close(staged_path, legacy_path)

    def test_model_accepts_only_grayscale_channels(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8)
        self.assertEqual(model.effective_input_channels, 1)
        self.assertEqual(model.stem[0].in_channels, 1)
        with self.assertRaisesRegex(ValueError, "grayscale"):
            EfficientHCCRNet(num_classes=11, width=8, in_channels=3)

    def test_forward_rejects_non_grayscale_tensor(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8)
        with self.assertRaisesRegex(ValueError, "BCHW grayscale"):
            model(torch.randn(2, 3, 64, 64))

    def test_stem_stride_one_preserves_first_stage_resolution(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8, stem_stride=1)
        features, stages = model.forward_features(
            torch.randn(2, 1, 64, 64), return_stages=True
        )
        self.assertEqual(stages["stage1"].shape[-2:], (64, 64))
        self.assertEqual(stages["stage2"].shape[-2:], (32, 32))
        self.assertEqual(stages["stage3"].shape[-2:], (16, 16))
        self.assertEqual(features.shape, stages["stage3"].shape)

    def test_reparameterized_depthwise_keeps_pointwise_projection(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8, reparameterize_depthwise=True)
        first_block = model.features[0]
        self.assertEqual(len(first_block.depthwise_branches), 3)
        self.assertIsInstance(first_block.pointwise[0], torch.nn.Conv2d)
        self.assertEqual(first_block.pointwise[0].kernel_size, (1, 1))

    def test_classifier_embedding_is_decoupled_from_late_backbone_capacity(
        self,
    ) -> None:
        model = EfficientHCCRNet(
            num_classes=11,
            width=8,
            backbone_output_channels=40,
            embedding_dim=12,
        )
        features = model.forward_features(torch.randn(2, 1, 64, 64))
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, (2, 40, 8, 8))
        self.assertEqual(model.backbone_output_channels, 40)
        self.assertEqual(model.embedding_dim, 12)
        self.assertEqual(model.embedding_projection.in_features, 40)
        self.assertEqual(model.embedding_projection.out_features, 12)
        self.assertEqual(model.classifier.in_features, 12)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 11))

    def test_decoupled_dimensions_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "backbone_output_channels"):
            EfficientHCCRNet(num_classes=11, width=8, backbone_output_channels=0)
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            EfficientHCCRNet(num_classes=11, width=8, embedding_dim=0)


class AngularMarginHeadTests(unittest.TestCase):
    def test_angular_logits_stay_float32_inside_cpu_autocast(self) -> None:
        classifier = AngularMarginClassifier(
            embedding_dim=4,
            num_classes=3,
            kind="cosface",
            scale=16.0,
            margin=0.1,
        ).eval()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = classifier(torch.randn(2, 4))
        self.assertEqual(logits.dtype, torch.float32)

    def test_cosface_margin_changes_only_target_logits(self) -> None:
        model = EfficientHCCRNet(
            num_classes=5,
            width=8,
            dropout=0.0,
            classification_head="cosface",
            logit_scale=16.0,
            angular_margin=0.1,
        ).eval()
        inputs = torch.randn(3, 1, 64, 64)
        targets = torch.tensor([0, 2, 4])
        inference_logits = model(inputs)
        training_logits = model.training_logits(inputs, targets)
        ddp_callable_logits = model(inputs, targets, 1.0)
        differences = inference_logits - training_logits
        expected = torch.zeros_like(differences).scatter(1, targets.unsqueeze(1), 1.6)
        torch.testing.assert_close(differences, expected)
        torch.testing.assert_close(ddp_callable_logits, training_logits)

    def test_arcface_keeps_target_free_inference_and_supports_warmup(self) -> None:
        model = EfficientHCCRNet(
            num_classes=5,
            width=8,
            dropout=0.0,
            classification_head="arcface",
        ).eval()
        inputs = torch.randn(3, 1, 64, 64)
        targets = torch.tensor([0, 2, 4])
        inference_logits = model(inputs)
        zero_margin_logits = model.training_logits(inputs, targets, 0.0)
        full_margin_logits = model.training_logits(inputs, targets, 1.0)
        torch.testing.assert_close(inference_logits, zero_margin_logits)
        self.assertTrue(
            torch.all(
                full_margin_logits.gather(1, targets.unsqueeze(1))
                < inference_logits.gather(1, targets.unsqueeze(1))
            )
        )
