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
            sum(parameter.numel() for parameter in model.parameters()), 434_344
        )
        self.assertIn("features.0.block.0.weight", model.state_dict())
        inputs = torch.randn(2, 1, 64, 64)
        model.eval()
        with torch.inference_mode():
            legacy_path = model.classifier(
                model.pool(model.features(model.stem(inputs)))
            )
            staged_path = model(inputs)
        torch.testing.assert_close(staged_path, legacy_path)

    def test_directional_input_modes_preserve_raw_grayscale_channel(self) -> None:
        inputs = torch.rand(2, 1, 64, 64)
        for mode, expected_channels in (
            ("grayscale", 1),
            ("grayscale_sobel", 2),
            ("grayscale_gabor", 5),
        ):
            model = EfficientHCCRNet(num_classes=11, width=8, input_mode=mode).eval()
            adapted = model.input_adapter(inputs)
            self.assertEqual(adapted.shape, (2, expected_channels, 64, 64))
            self.assertEqual(model.effective_input_channels, expected_channels)
            torch.testing.assert_close(adapted[:, :1], inputs)
            torch.testing.assert_close(adapted, model.input_adapter(inputs))
            self.assertEqual(model(inputs).shape, (2, 11))

    def test_directional_filters_do_not_create_false_white_border_edges(self) -> None:
        white = torch.ones(1, 1, 32, 32)
        for mode in ("grayscale_sobel", "grayscale_gabor"):
            adapter = EfficientHCCRNet(
                num_classes=2, width=4, input_mode=mode
            ).input_adapter
            directional = adapter(white)[:, 1:]
            self.assertLess(directional.abs().max().item(), 1e-6)

    def test_directional_filters_are_polarity_invariant(self) -> None:
        black_on_white = torch.ones(1, 1, 32, 32)
        black_on_white[:, :, 8:24, 12:20] = 0
        white_on_black = 1.0 - black_on_white
        for mode in ("grayscale_sobel", "grayscale_gabor"):
            adapter = EfficientHCCRNet(
                num_classes=2, width=4, input_mode=mode
            ).input_adapter
            torch.testing.assert_close(
                adapter(black_on_white)[:, 1:],
                adapter(white_on_black)[:, 1:],
                atol=1e-6,
                rtol=1e-6,
            )

    def test_directional_kernels_are_fixed_nonpersistent_buffers(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8, input_mode="grayscale_gabor")
        self.assertFalse(
            any(
                name.startswith("input_adapter") for name, _ in model.named_parameters()
            )
        )
        self.assertNotIn("input_adapter.kernels", model.state_dict())
        self.assertEqual(model.stem[0].in_channels, 5)

    def test_unknown_input_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_mode"):
            EfficientHCCRNet(num_classes=11, width=8, input_mode="unknown")


class AttentionModuleTests(unittest.TestCase):
    def test_eca_and_se_preserve_logits_shape(self) -> None:
        inputs = torch.randn(2, 1, 64, 64)
        baseline = EfficientHCCRNet(num_classes=11, width=8)
        eca = EfficientHCCRNet(num_classes=11, width=8, attention="eca")
        se = EfficientHCCRNet(num_classes=11, width=8, attention="se")
        self.assertEqual(eca(inputs).shape, (2, 11))
        self.assertEqual(se(inputs).shape, (2, 11))
        baseline_parameters = sum(p.numel() for p in baseline.parameters())
        self.assertEqual(
            sum(p.numel() for p in eca.parameters()), baseline_parameters + 3
        )
        self.assertGreater(sum(p.numel() for p in se.parameters()), baseline_parameters)

    def test_attention_placement_is_explicit_and_validated(self) -> None:
        model = EfficientHCCRNet(
            num_classes=11, width=8, attention="eca", attention_stages=(2, 3)
        )
        self.assertEqual(set(model.stage_attention), {"2", "3"})
        with self.assertRaisesRegex(ValueError, "unique values"):
            EfficientHCCRNet(
                num_classes=11,
                width=8,
                attention="eca",
                attention_stages=(3, 3),
            )


class AngularMarginHeadTests(unittest.TestCase):
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
        differences = inference_logits - training_logits
        expected = torch.zeros_like(differences).scatter(1, targets.unsqueeze(1), 1.6)
        torch.testing.assert_close(differences, expected)

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


class CrossStageArchitectureTests(unittest.TestCase):
    def test_additive_bridge_preserves_stage_and_logits_shapes(self) -> None:
        model = EfficientHCCRNet(
            num_classes=11, width=8, cross_stage="projected_residual"
        )
        features, stages = model.forward_features(
            torch.randn(2, 1, 64, 64), return_stages=True
        )
        self.assertEqual(stages["stage2"].shape, (2, 16, 16, 16))
        self.assertEqual(stages["stage3"].shape, (2, 32, 8, 8))
        self.assertEqual(features.shape, stages["stage3"].shape)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 11))

    def test_c_cbam_uses_parallel_attention_on_stage2_source(self) -> None:
        model = EfficientHCCRNet(num_classes=11, width=8, cross_stage="c_cbam")
        state = model.state_dict()
        self.assertIn("cross_stage_bridge.attention.channel.mlp.0.weight", state)
        self.assertIn("cross_stage_bridge.attention.spatial.convolution.weight", state)
        self.assertIn("cross_stage_bridge.bridge.projection.0.weight", state)
        self.assertEqual(model(torch.randn(2, 1, 64, 64)).shape, (2, 11))

    def test_csp_stage_splits_merges_and_reduces_stage3_parameters(self) -> None:
        baseline = EfficientHCCRNet(num_classes=1000, width=64)
        model = EfficientHCCRNet(
            num_classes=1000, width=64, csp_stages=(3,), csp_split_ratio=0.5
        )
        features, stages = model.forward_features(
            torch.randn(2, 1, 96, 96), return_stages=True
        )
        self.assertEqual(model.stage_ranges, ((0, 1), (1, 3), (3, 4)))
        self.assertEqual(stages["stage3"].shape, (2, 256, 12, 12))
        self.assertEqual(features.shape, stages["stage3"].shape)
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )

    def test_cross_stage_and_csp_options_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross_stage"):
            EfficientHCCRNet(num_classes=11, width=8, cross_stage="unknown")
        with self.assertRaisesRegex(ValueError, "csp_stages"):
            EfficientHCCRNet(num_classes=11, width=8, csp_stages=(1,))
        with self.assertRaisesRegex(ValueError, "csp_split_ratio"):
            EfficientHCCRNet(num_classes=11, width=8, csp_split_ratio=1.0)
