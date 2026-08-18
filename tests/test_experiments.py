from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hccr.experiments import compare_runs


class ExperimentComparisonTests(unittest.TestCase):
    def test_quality_gate_accepts_accuracy_gain_within_latency_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "experiment_summary.csv"
            summary.write_text(
                "run_id,top1,latency_p95_ms\nbase,0.7,10\ncandidate,0.72,11\n",
                encoding="utf-8",
            )
            verdict = compare_runs(summary, "base", "candidate", 0.01, 1.2)
            self.assertTrue(verdict["accepted"])
