import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from hccr.models import build_model
from hccr.models.artifacts import build_model_from_metadata
from hccr.models.heads import FactorizedClassifier


class ModernBaselineTests(unittest.TestCase):
    def test_optional_dependency_failure_names_install_extra(self):
        with (
            patch.dict(sys.modules, {"timm": None}),
            self.assertRaisesRegex(ImportError, r"\[modern\]"),
        ):
            build_model("mobilenetv4_conv_small", num_classes=7)

    def test_modern_backbones_use_grayscale_shared_head_and_roundtrip(self):
        for name in ("mobilenetv4_conv_small", "repvit_m0_9"):
            with self.subTest(name=name):

                def create_backbone(*args, **kwargs):
                    backbone = nn.Sequential(
                        nn.Conv2d(1, 12, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(1)
                    )
                    backbone.num_features = 12
                    return backbone

                create_model = Mock(side_effect=create_backbone)
                with patch.dict(
                    sys.modules, {"timm": SimpleNamespace(create_model=create_model)}
                ):
                    options = dict(
                        num_classes=7,
                        classification_head="cosface",
                        classifier_structure="dynamic_residual",
                        classifier_rank=3,
                        residual_class_indices=[1, 5],
                    )
                    model = build_model(name, **options).eval()
                    create_model.assert_called_once_with(
                        name,
                        pretrained=False,
                        num_classes=0,
                        in_chans=1,
                        global_pool="avg",
                    )
                    self.assertEqual(model.embedding_dim, 320)
                    self.assertEqual(model.embedding_projection.in_features, 12)
                    self.assertIsInstance(model.classifier, FactorizedClassifier)
                    sample, labels = torch.randn(2, 1, 16, 16), torch.tensor([1, 5])
                    with torch.no_grad():
                        logits = model(sample)
                        margin_logits = model(sample, labels)
                    expected = torch.zeros_like(logits).scatter(1, labels[:, None], 3.2)
                    torch.testing.assert_close(logits - margin_logits, expected)
                    restored = build_model_from_metadata(
                        {"model": {"name": name, **options}}
                    ).eval()
                    restored.load_state_dict(model.state_dict())
                    with torch.no_grad():
                        torch.testing.assert_close(restored(sample), logits)

    @unittest.skipUnless(importlib.util.find_spec("timm"), "optional timm unavailable")
    def test_real_optional_backbones_forward_without_pretrained_download(self):
        for name in ("mobilenetv4_conv_small", "repvit_m0_9"):
            with self.subTest(name=name):
                model = build_model(name, num_classes=7).eval()
                with torch.inference_mode():
                    logits = model(torch.zeros(1, 1, 96, 96))
                self.assertEqual(logits.shape, (1, 7))
                self.assertEqual(model.embedding_dim, 320)
                self.assertTrue(torch.isfinite(logits).all())
