from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from scripts.benchmark_cpu_lmdb import (
    _prepare_model_for_mode,
    _resolve_run_manifest,
    _shared_preprocess,
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
from hccr.models import build_model_from_metadata, read_checkpoint_metadata


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

    def test_cpu_script_allows_one_timed_iteration(self) -> None:
        result = benchmark_model(
            torch.nn.Identity(),
            torch.zeros(1, 1, 4, 4),
            warmup_iterations=1,
            timed_iterations=1,
            repetitions=1,
        )
        self.assertEqual(result["repeat_summaries"][0]["latency_std_ms"], 0.0)

    def test_cpu_script_defaults_to_eager_mode(self) -> None:
        arguments = build_cpu_benchmark_parser().parse_args([])
        self.assertEqual(arguments.mode, "eager")

    def test_cpu_script_honors_explicit_run_directories(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run-1"
            run_dir.mkdir()
            (run_dir / "config.json").write_text('{"seed": 23}\n', encoding="utf-8")
            (run_dir / "checkpoint_metadata.json").write_text(
                """{
  "model": {"name": "resnet18"},
  "preprocess": {"image_size": 96, "margin": 4}
}
""",
                encoding="utf-8",
            )
            (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
            arguments = build_cpu_benchmark_parser().parse_args(
                ["--runs", str(run_dir)]
            )
            manifest = _resolve_run_manifest(arguments)
        self.assertEqual(manifest[0]["run_id"], "run-1")
        self.assertEqual(manifest[0]["seed"], 23)
        self.assertEqual(manifest[0]["selection"], "explicit")

    def test_cpu_script_rejects_mixed_preprocessing(self) -> None:
        with self.assertRaisesRegex(ValueError, "must share preprocessing"):
            _shared_preprocess(
                [
                    {"run_id": "a", "preprocess": {"image_size": 64}},
                    {"run_id": "b", "preprocess": {"image_size": 96}},
                ]
            )

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

    def test_public_builder_reconstructs_legacy_checkpoint_metadata(self) -> None:
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
            metadata = read_checkpoint_metadata(run_dir)
            model = build_model_from_metadata(metadata)
        self.assertEqual(model.name, "efficient_hccr")
