from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from scripts.benchmark_cpu_lmdb import (
    _model_from_run,
    _prepare_model_for_mode,
    benchmark_model,
)
from scripts.benchmark_cpu_lmdb import (
    build_parser as build_cpu_benchmark_parser,
)

from hccr.benchmarking import (
    BENCHMARK_SCHEMA_VERSION,
    aggregate_timing_summaries,
    build_benchmark_protocol,
    summarize_timings,
)


class BenchmarkSchemaTests(unittest.TestCase):
    def test_timing_summary_uses_canonical_metric_names(self) -> None:
        summary = summarize_timings([1.0, 2.0, 3.0], batch_size=1)
        self.assertEqual(
            set(summary),
            {
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "samples_per_second",
            },
        )
        self.assertEqual(summary["latency_mean_ms"], 2.0)
        self.assertEqual(summary["latency_p50_ms"], 2.0)

    def test_repetition_aggregation_matches_resource_profile_shape(self) -> None:
        first = summarize_timings([1.0, 2.0])
        second = summarize_timings([2.0, 3.0])
        aggregate = aggregate_timing_summaries([first, second])
        self.assertEqual(len(aggregate["repeat_summaries"]), 2)
        self.assertEqual(aggregate["latency_mean_ms"], 2.0)
        self.assertIn("latency_p99_ms", aggregate)
        self.assertIn("samples_per_second", aggregate)

    def test_protocol_declares_schema_relevant_execution_details(self) -> None:
        protocol = build_benchmark_protocol(
            device="cpu",
            batch_size=1,
            warmup_iterations=2,
            timed_iterations=3,
            repetitions=4,
            scope="model_forward",
            metadata={"intra_op_threads": 1},
        )
        self.assertEqual(BENCHMARK_SCHEMA_VERSION, 1)
        self.assertEqual(protocol["aggregation"], "median_of_repetition_summaries")
        self.assertEqual(protocol["timed_iterations"], 3)
        self.assertEqual(protocol["scope"], "model_forward")

    def test_cpu_script_benchmark_uses_shared_repetition_schema(self) -> None:
        result = benchmark_model(
            torch.nn.Identity(),
            torch.zeros(1, 1, 4, 4),
            warmup_iterations=1,
            timed_iterations=2,
            repetitions=2,
        )
        self.assertEqual(len(result["repeat_summaries"]), 2)
        self.assertIn("latency_p95_ms", result)
        self.assertIn("samples_per_second", result)

    def test_cpu_script_defaults_to_eager_mode(self) -> None:
        arguments = build_cpu_benchmark_parser().parse_args([])
        self.assertEqual(arguments.mode, "eager")

    def test_cpu_script_modes_control_model_optimization(self) -> None:
        model = torch.nn.Identity()
        with patch(
            "scripts.benchmark_cpu_lmdb.optimize_model_for_inference",
            return_value=torch.nn.Identity(),
        ) as optimize:
            eager = _prepare_model_for_mode(model, "eager")
            optimize.assert_not_called()
            optimized = _prepare_model_for_mode(model, "optimized")
        optimize.assert_called_once_with(model)
        self.assertIs(eager, model)
        self.assertIsNot(optimized, model)
        self.assertFalse(eager.training)
        self.assertFalse(optimized.training)

    def test_cpu_script_reconstructs_legacy_run_from_checkpoint_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoint_metadata.json").write_text(
                """{
  "schema_version": 3,
  "model": {
    "name": "efficient_hccr",
    "num_classes": 3,
    "width": 4,
    "stage_depths": [1, 1, 1],
    "stem_stride": 2,
    "reparameterize_depthwise": false,
    "dropout": 0.1,
    "classification_head": "softmax"
  }
}
""",
                encoding="utf-8",
            )
            model, model_name = _model_from_run(run_dir)
        self.assertEqual(model_name, "efficient_hccr")
        self.assertEqual(model.name, "efficient_hccr")
