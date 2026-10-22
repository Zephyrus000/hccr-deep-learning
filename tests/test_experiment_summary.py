from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hccr.artifacts.summary import EXPERIMENT_SUMMARY_FIELDS, SummaryTable


class ExperimentSummaryTests(unittest.TestCase):
    def test_summary_writes_measured_values_and_keeps_missing_test_blank(self):
        with TemporaryDirectory() as directory:
            table = SummaryTable(Path(directory))
            path = table.append(
                {"run_id": "run-1", "validation_top1": 0.8}, _resource_profile()
            )
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), EXPERIMENT_SUMMARY_FIELDS)
            self.assertEqual(rows[0]["top1"], "")
            self.assertEqual(rows[0]["validation_top1"], "0.8")
            self.assertEqual(rows[0]["peak_inference_cuda_memory_mib"], "12.5")
            self.assertEqual(rows[0]["latency_p95_ms"], "2.0")
            self.assertNotIn("latency_p99_ms", rows[0])
            self.assertNotIn("end_to_end_latency_p95_ms", rows[0])

    def test_old_summary_is_preserved_byte_for_byte(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = b"run_id,legacy_metric\r\nold,123\r\n"
            (root / "experiment_summary.csv").write_bytes(original)
            table = SummaryTable(root)
            path = table.append({"run_id": "new-1"}, _resource_profile())
            table.append({"run_id": "new-2"}, _resource_profile())
            self.assertEqual(path.name, "experiment_summary.v2.csv")
            self.assertEqual((root / "experiment_summary.csv").read_bytes(), original)
            with path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(
                    [r["run_id"] for r in csv.DictReader(file)], ["new-1", "new-2"]
                )

    def test_incompatible_versioned_summary_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("experiment_summary.csv", "experiment_summary.v2.csv"):
                (root / name).write_text("unknown\noriginal\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible summary"):
                SummaryTable(root).append({"run_id": "new"}, _resource_profile())
            self.assertEqual(
                (root / "experiment_summary.v2.csv").read_text(), "unknown\noriginal\n"
            )


def _resource_profile():
    return {
        "parameter_count": 10,
        "estimated_macs": 20,
        "estimated_flops": 40,
        "mac_coverage": {"complete": True, "unsupported_operator_types": []},
        "flop_coverage": {"complete": True, "unsupported_operator_types": []},
        "inference_benchmarks": [
            {
                "batch_size": 1,
                "latency_mean_ms": 1.5,
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
                "samples_per_second": 1000.0,
                "peak_cuda_memory_mib": 12.5,
            }
        ],
    }
