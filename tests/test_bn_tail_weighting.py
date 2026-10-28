from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from hccr.training.trainer import _batch_norm_tail_momentum, train_epoch


class BatchNormTailWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_final_26_samples_do_not_receive_a_full_256_sample_bn_update(self):
        model = nn.Sequential(nn.BatchNorm2d(1), nn.Flatten(), nn.Linear(4, 2))
        images = torch.cat((torch.zeros(256, 1, 2, 2), torch.full((26, 1, 2, 2), 10.0)))
        targets = torch.zeros(282, dtype=torch.int64)
        loader = DataLoader(
            [
                (image, target, "")
                for image, target in zip(images, targets, strict=True)
            ],
            batch_size=256,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        metrics = train_epoch(model, loader, optimizer, "cpu", nn.CrossEntropyLoss())
        expected_momentum = 1 - 0.9 ** (26 / 256)
        self.assertAlmostEqual(
            model[0].running_mean.item(), 10 * expected_momentum, places=6
        )
        self.assertAlmostEqual(
            model[0].running_var.item(), 0.9 * (1 - expected_momentum), places=6
        )
        self.assertEqual(model[0].momentum, 0.1)
        self.assertEqual(int(model[0].num_batches_tracked), 2)
        self.assertEqual(metrics["optimizer_steps"], 2)

    def test_weighted_tail_preserves_training_logits_gradients_and_optimizer_update(
        self,
    ):
        torch.manual_seed(5)
        ordinary = nn.Sequential(nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(8, 3))
        weighted = copy.deepcopy(ordinary)
        x = torch.randn(26, 2, 2, 2)
        targets = torch.arange(26) % 3
        raw_optimizer = torch.optim.SGD(ordinary.parameters(), lr=0.1)
        weighted_optimizer = torch.optim.SGD(weighted.parameters(), lr=0.1)
        expected = ordinary(x)
        nn.functional.cross_entropy(expected, targets).backward()
        with _batch_norm_tail_momentum([(weighted[0], 0.1)], 26, 256):
            actual = weighted(x)
        nn.functional.cross_entropy(actual, targets).backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for raw, adjusted in zip(
            ordinary.parameters(), weighted.parameters(), strict=True
        ):
            torch.testing.assert_close(raw.grad, adjusted.grad, rtol=0, atol=0)
        raw_optimizer.step()
        weighted_optimizer.step()
        for raw, adjusted in zip(
            ordinary.parameters(), weighted.parameters(), strict=True
        ):
            torch.testing.assert_close(raw, adjusted, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(ordinary[0].running_mean, weighted[0].running_mean)
        )
        self.assertEqual(weighted[0].momentum, 0.1)

    def test_full_batch_frozen_norm_and_cumulative_momentum_stay_unchanged(self):
        normal = nn.BatchNorm2d(1)
        frozen = nn.BatchNorm2d(1).eval()
        cumulative = nn.BatchNorm2d(1, momentum=None)
        with _batch_norm_tail_momentum([(normal, 0.1), (frozen, 0.1)], 256, 256):
            self.assertEqual(normal.momentum, 0.1)
            self.assertEqual(frozen.momentum, 0.1)
        with _batch_norm_tail_momentum([(frozen, 0.1)], 26, 256):
            self.assertEqual(frozen.momentum, 0.1)
        with _batch_norm_tail_momentum([], 26, 256):
            cumulative(torch.ones(26, 1, 2, 2))
        self.assertIsNone(cumulative.momentum)
        self.assertEqual(cumulative.running_mean.item(), 1.0)
        with _batch_norm_tail_momentum([(normal, 0.1)], 26, None):
            self.assertEqual(normal.momentum, 0.1)

    def test_original_momentum_is_restored_if_forward_raises(self):
        norm = nn.BatchNorm2d(1, momentum=0.2)
        with (
            self.assertRaisesRegex(RuntimeError, "forward failed"),
            _batch_norm_tail_momentum([(norm, 0.2)], 26, 256),
        ):
            self.assertNotEqual(norm.momentum, 0.2)
            raise RuntimeError("forward failed")
        self.assertEqual(norm.momentum, 0.2)


if __name__ == "__main__":
    unittest.main()
