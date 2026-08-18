from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from hccr.utils.logging import close_logging, configure_logging


class LoggingTests(unittest.TestCase):
    def test_progress_logs_are_kept_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = configure_logging(Path(directory))
            logging.getLogger("hccr.train").info("batch=100 loss=1.0")
            close_logging(logger)
            self.assertIn("batch=100", (Path(directory) / "run.log").read_text())
