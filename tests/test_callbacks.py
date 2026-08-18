from __future__ import annotations

import unittest

from hccr.training.callbacks import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_configured_consecutive_non_improvements(self) -> None:
        callback = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(callback.update(0.50))
        self.assertFalse(callback.update(0.505))
        self.assertTrue(callback.update(0.504))
