from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from hccr.evaluation.diagnostics import write_validation_diagnostics


class ValidationDiagnosticsTest(unittest.TestCase):
    def test_writes_debugging_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "raw" / "sample.png"
            image_path.parent.mkdir()
            Image.new("L", (8, 8), color=255).save(image_path)
            output = root / "output"
            health = write_validation_diagnostics(
                output,
                torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
                torch.tensor([1, 1]),
                [
                    {"sample_id": "a", "source_file": "raw/sample.png"},
                    {"sample_id": "b", "source_file": "raw/sample.png"},
                ],
                root,
            )
            self.assertEqual(health["samples"], 2)
            self.assertTrue((output / "per_class_metrics.csv").exists())
            self.assertTrue((output / "validation_errors.csv").exists())
            self.assertTrue((output / "reliability_diagram.png").exists())
            self.assertTrue((output / "validation_error_gallery.png").exists())
            self.assertIn(
                "expected_calibration_error",
                json.loads((output / "validation_health.json").read_text()),
            )
