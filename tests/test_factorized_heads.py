import copy
import io
import unittest

import torch
from torch.nn import functional as F

from hccr.models import EfficientHCCRNet, build_model
from hccr.models.artifacts import build_model_from_metadata
from hccr.models.heads import AngularMarginClassifier, FactorizedClassifier


def head(structure="dynamic_residual", **options):
    return FactorizedClassifier(
        8,
        7,
        "cosface",
        16.0,
        0.1,
        classifier_structure=structure,
        classifier_rank=3,
        residual_class_indices=() if structure == "low_rank" else (5, 1, 3),
        **options,
    )


def explicit_logits(model, x, targets=None):
    z = F.normalize(x.float(), dim=1, eps=model.normalization_eps)
    p = model.coefficients @ model.basis
    q = F.normalize(p, dim=1, eps=model.normalization_eps)
    score = z @ q.T
    if model.residual_indices.numel():
        alpha = model.gate_values(
            z, score, p.norm(dim=1).clamp_min(model.normalization_eps)
        )
        weights = (
            q[model.residual_indices][None] + alpha[:, :, None] * model.residuals[None]
        )
        selected = (
            z[:, None] * F.normalize(weights, dim=2, eps=model.normalization_eps)
        ).sum(2)
        score = score.index_copy(1, model.residual_indices, selected)
    score = score.clamp(-1 + 1e-7, 1 - 1e-7)
    if targets is not None:
        score = score.scatter(
            1, targets[:, None], score.gather(1, targets[:, None]) - model.margin
        )
    return score * model.scale


class FactorizedHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(13)

    def test_factorized_values_and_gradients_match_explicit_dynamic_weights(self):
        model = head()
        with torch.no_grad():
            model.residuals.normal_(std=0.3)
            model.gate_weight.normal_(std=0.3)
            model.gate_gamma.fill_(0.2)
        reference = copy.deepcopy(model)
        x = torch.randn(4, 8, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        targets = torch.tensor([5, 0, 3, 2])
        actual = model(x, targets)
        expected = explicit_logits(reference, xr, targets)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=5e-5, rtol=3e-4)
        for (name, p), (name2, pr) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            self.assertEqual(name, name2)
            torch.testing.assert_close(p.grad, pr.grad, atol=8e-5, rtol=5e-4)

    def test_static_dynamic_initial_logits_and_residual_gradients_match(self):
        static, dynamic = head("static_residual"), head()
        dynamic.load_state_dict(static.state_dict(), strict=False)
        x = torch.randn(4, 8)
        target = torch.tensor([1, 2, 3, 4])
        a, b = static(x, target), dynamic(x, target)
        torch.testing.assert_close(a, b)
        F.cross_entropy(a, target).backward()
        F.cross_entropy(b, target).backward()
        torch.testing.assert_close(static.residuals.grad, dynamic.residuals.grad)
        self.assertGreater(dynamic.residuals.grad.norm().item(), 0)
        self.assertEqual(dynamic.gate_weight.grad.count_nonzero().item(), 0)
        with torch.no_grad():
            dynamic.residuals.add_(dynamic.residuals.grad, alpha=-0.1)
        dynamic.zero_grad()
        F.cross_entropy(dynamic(x, target), target).backward()
        self.assertGreater(dynamic.gate_weight.grad.norm().item(), 0)

    def test_zero_margin_and_cosface_target_only_with_amp(self):
        model = head()
        x, labels = torch.randn(3, 8), torch.tensor([1, 0, 5])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inference = model(x)
            train = model(x, labels)
            zero = model(x, labels, 0.0)
        self.assertEqual(train.dtype, torch.float32)
        torch.testing.assert_close(inference, zero)
        expected = torch.zeros_like(train).scatter(1, labels[:, None], 1.6)
        torch.testing.assert_close(inference - train, expected)
        dense = AngularMarginClassifier(8, 7, "normalized_softmax", 16.0, 0.2)
        torch.testing.assert_close(dense(x), dense(x, labels))

    def test_prior_retains_gate_graph_and_deepcopy_discards_transients(self):
        model = head()
        model(torch.randn(2, 8))
        penalty = model.regularization_loss(0.1, 0.2)
        penalty.backward()
        self.assertGreater(model.gate_bias.grad.item(), 0)
        cloned = copy.deepcopy(model)
        self.assertIsNone(cloned._gate_mean)
        self.assertIsNone(cloned._deployment_cache)

    def test_arcface_tail_is_monotonic_in_both_classifier_families(self):
        dense = AngularMarginClassifier(2, 2, "arcface", 16.0, 0.3)
        factorized = FactorizedClassifier(2, 2, "arcface", 16.0, 0.3, classifier_rank=2)
        with torch.no_grad():
            dense.weight.copy_(torch.eye(2))
            factorized.coefficients.copy_(torch.eye(2))
            factorized.basis.copy_(torch.eye(2))
        angles = torch.linspace(2.6, torch.pi, 101)
        inputs = torch.stack((angles.cos(), angles.sin()), dim=1)
        targets = torch.zeros(len(angles), dtype=torch.long)
        for model in (dense, factorized):
            plain = model(inputs)
            logits = model(inputs, targets)
            self.assertTrue(torch.all(logits[1:, 0] <= logits[:-1, 0]))
            self.assertTrue(torch.all(logits[:, 0] < plain[:, 0]))
            torch.testing.assert_close(logits[:, 1], plain[:, 1])
        torch.testing.assert_close(dense(inputs, targets), factorized(inputs, targets))

    def test_deployment_cache_invalidates_on_mutation_loading_and_train(self):
        model = head().eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            first = model(x)
            cache = model._deployment_cache
            self.assertTrue(all(t.ndim == 1 for t in cache[1]))
            model.basis.add_(0.3)
            second = model(x)
            self.assertIsNot(cache, model._deployment_cache)
            self.assertFalse(torch.allclose(first, second))
            torch.testing.assert_close(second, explicit_logits(model, x))
            replacement = head().state_dict()
            model.load_state_dict(replacement)
            torch.testing.assert_close(model(x), explicit_logits(model, x))
        model.train()
        output = model(x)
        output.sum().backward()
        self.assertIsNotNone(model.basis.grad)

    def test_dense_eval_cache_tracks_weight_updates_and_checkpoint_loading(self):
        model = AngularMarginClassifier(8, 7, "cosface", 16.0, 0.1).eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            before = model(x)
            model.weight[0].neg_()
            after = model(x)
            self.assertFalse(torch.allclose(before, after))
            replacement = AngularMarginClassifier(8, 7, "cosface", 16.0, 0.1)
            model.load_state_dict(replacement.state_dict())
            expected = (
                F.normalize(x, dim=1) @ F.normalize(replacement.weight, dim=1).T
            ).clamp(-1 + 1e-7, 1 - 1e-7) * model.scale
            torch.testing.assert_close(model(x), expected)

    def test_serialization_preserves_sparse_indices_without_dense_cache(self):
        model = head().eval()
        x = torch.randn(2, 8)
        with torch.no_grad():
            result = model(x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        self.assertFalse(any("cache" in name for name in state))
        self.assertFalse(any(value.shape == (7, 8) for value in state.values()))
        restored = head().eval()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(x), result)

    def test_svd_initialization_is_row_scale_invariant_and_reconstruction_exact(self):
        weight = torch.randn(7, 8)
        a, b = head(), head()
        a.initialize_from_dense(weight, reconstruct_residual=True)
        b.initialize_from_dense(
            weight * torch.arange(1.0, 8.0)[:, None], reconstruct_residual=True
        )
        x = torch.randn(4, 8)
        torch.testing.assert_close(a(x), b(x), atol=2e-5, rtol=2e-5)
        dense = F.normalize(x, dim=1) @ F.normalize(weight, dim=1).T * a.scale
        torch.testing.assert_close(
            a(x)[:, a.residual_indices],
            dense[:, a.residual_indices],
            atol=3e-6,
            rtol=2e-5,
        )

    def test_norm_floor_and_opposing_residuals_remain_finite_and_match(self):
        model = head("static_residual")
        with torch.no_grad():
            model.coefficients[0].zero_()
            q = F.normalize(
                model.coefficients @ model.basis, dim=1, eps=model.normalization_eps
            )
            model.residuals.copy_(-2 * q[model.residual_indices])
        x = torch.randn(3, 8)
        actual = model(x)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(
            actual, explicit_logits(model, x), atol=2e-5, rtol=2e-5
        )

    def test_counts_and_gate_controls(self):
        model = FactorizedClassifier(
            320,
            3755,
            "cosface",
            32.0,
            0.1,
            classifier_structure="dynamic_residual",
            classifier_rank=64,
            residual_class_indices=range(500),
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), 421122)
        self.assertEqual(model.inference_macs(), 473280)
        for kind in ("hadamard", "cosine", "sample", "global", "class"):
            with self.subTest(kind=kind):
                m = head(gate_kind=kind)
                self.assertEqual(m(torch.randn(2, 8)).shape, (2, 7))

    def test_model_artifact_reconstruction_and_baseline_training_margin(self):
        common = dict(
            num_classes=7,
            embedding_dim=8,
            classification_head="cosface",
            classifier_structure="dynamic_residual",
            classifier_rank=3,
            residual_class_indices=[1, 5],
            gate_kind="hadamard",
            gate_use_gap=True,
            gate_detach_gap=True,
            normalization_eps=1e-6,
        )
        model = EfficientHCCRNet(
            width=4, stage_depths=(1, 1, 1), dropout=0.0, **common
        ).eval()
        metadata = {
            "model": dict(
                name="efficient_hccr",
                width=4,
                stage_depths=[1, 1, 1],
                dropout=0.0,
                **common,
            )
        }
        restored = build_model_from_metadata(metadata).eval()
        restored.load_state_dict(model.state_dict())
        x = torch.randn(2, 1, 32, 32)
        torch.testing.assert_close(model(x), restored(x))
        baseline = build_model("mobilenet_v3_small", **common).eval()
        labels = torch.tensor([1, 5])
        diff = baseline(x) - baseline(x, labels)
        torch.testing.assert_close(
            diff,
            torch.zeros_like(diff).scatter(1, labels[:, None], 3.2),
            atol=2e-6,
            rtol=1e-5,
        )

    def test_invalid_selection_and_geometry_fail(self):
        for indices in ([1, 1], [-1], [7], [1.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                FactorizedClassifier(
                    8,
                    7,
                    "cosface",
                    32.0,
                    0.1,
                    classifier_structure="static_residual",
                    classifier_rank=3,
                    residual_class_indices=indices,
                )
        with self.assertRaises(ValueError):
            FactorizedClassifier(8, 7, "softmax", 32.0, 0.0, classifier_rank=3)


if __name__ == "__main__":
    unittest.main()
