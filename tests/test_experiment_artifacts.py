import tempfile
import unittest
from pathlib import Path

from hccr.utils.experiment import initialize_run, write_curves


class ExperimentArtifactsTests(unittest.TestCase):
    def test_run_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_id = initialize_run(output, {"seed": 7}, {"manifest_digest": "abc"})
            write_curves(output, [{"epoch": 1.0, "top1": 0.5}])
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "curves.json").is_file())
            self.assertIn(run_id, (output / "metadata.json").read_text())
