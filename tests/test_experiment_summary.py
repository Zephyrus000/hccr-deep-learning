from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hccr.training.summary import (
    EXPERIMENT_SUMMARY_FIELDS,
    merge_resource_profile_into_summary,
    write_experiment_summary_rows,
)


class ExperimentSummaryTests(unittest.TestCase):
    def test_profile_migration_preserves_unmeasured_end_to_end_values(self) -> None:
        migrated = merge_resource_profile_into_summary(
            {
                "run_id": "run-1",
                "end_to_end_latency_p50_ms": "9.0",
                "end_to_end_latency_p95_ms": "10.0",
                "end_to_end_latency_p99_ms": "11.0",
            },
            _resource_profile(end_to_end=None),
        )
        self.assertEqual(migrated["end_to_end_latency_p50_ms"], "9.0")
        self.assertEqual(migrated["end_to_end_latency_p95_ms"], "10.0")
        self.assertEqual(migrated["end_to_end_latency_p99_ms"], "11.0")

    def test_profile_migration_reads_peak_memory_from_batch_benchmark(self) -> None:
        migrated = merge_resource_profile_into_summary(
            {"run_id": "run-1"}, _resource_profile(end_to_end={})
        )
        self.assertEqual(migrated["peak_inference_cuda_memory_mib"], 12.5)

    def test_writer_uses_one_canonical_header(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_summary.csv"
            write_experiment_summary_rows(path, [{"run_id": "run-1"}])
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), EXPERIMENT_SUMMARY_FIELDS)
            self.assertEqual(rows[0]["run_id"], "run-1")


def _resource_profile(end_to_end: dict | None) -> dict:
    return {
        "effective_input_channels": 1,
        "parameter_count": 10,
        "estimated_macs": 20,
        "estimated_flops": 40,
        "mac_coverage": {"complete": True, "unsupported_operator_types": []},
        "flop_coverage": {"complete": True, "unsupported_operator_types": []},
        "full_class_projection": {},
        "inference_benchmarks": [
            {
                "batch_size": 1,
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
                "latency_p99_ms": 3.0,
                "samples_per_second": 1000.0,
                "peak_cuda_memory_mib": 12.5,
            }
        ],
        "end_to_end_batch1_benchmark": end_to_end,
    }
