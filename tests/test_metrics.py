import unittest

import torch

from hccr.evaluation import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_top_metrics(self) -> None:
        metrics = classification_metrics(
            torch.tensor([[3.0, 1.0], [0.0, 2.0]]), torch.tensor([0, 1])
        )
        self.assertEqual(metrics, {"top1": 1.0, "top5": 1.0})
