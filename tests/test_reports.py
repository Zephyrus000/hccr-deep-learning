import tempfile
import unittest
from pathlib import Path

from hccr.evaluation.reports import save_learning_curves


class ReportTests(unittest.TestCase):
    def test_learning_curve_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = save_learning_curves(
                Path(directory), [{"epoch": 1.0, "train_loss": 1.0, "top1": 0.5}]
            )
            self.assertTrue(output.is_file())
