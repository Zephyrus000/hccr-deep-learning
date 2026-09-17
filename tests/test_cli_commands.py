from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import torch

from hccr.cli import build_parser, main
from hccr.commands import benchmark, deploy, train, validate


class CommandRoutingTests(unittest.TestCase):
    def test_workflow_boundaries_are_separate_modules(self) -> None:
        handlers = {
            name: build_parser().parse_args([name]).command_handler
            for name in ("validate", "deploy")
        }
        self.assertIs(handlers["validate"], validate.run)
        self.assertIs(handlers["deploy"], deploy.run)

    def test_train_parser_and_handler_are_owned_by_train_module(self) -> None:
        arguments = build_parser().parse_args(["train", "--manifest", "manifest.csv"])
        self.assertIs(arguments.command_handler, train.run)
        self.assertEqual(arguments.evaluation_policy, "final_test")

    def test_train_parser_accepts_validation_only_policy(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--evaluation-policy",
                "validation_only",
            ]
        )
        self.assertEqual(arguments.evaluation_policy, "validation_only")

    def test_train_parser_selects_reference_model(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--model",
                "mobilenet_v3_small",
            ]
        )
        config = train.config_from_arguments(arguments)
        self.assertEqual(config.model, "mobilenet_v3_small")

    def test_train_parser_accepts_strict_reproducibility(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--reproducibility-mode",
                "strict",
            ]
        )
        self.assertEqual(arguments.reproducibility_mode, "strict")

    def test_train_parser_exposes_distributed_configuration(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--distributed",
                "--distributed-backend",
                "gloo",
            ]
        )
        config = train.config_from_arguments(arguments)
        self.assertTrue(config.distributed)
        self.assertEqual(config.distributed_backend, "gloo")

    def test_train_parser_exposes_mixed_precision_configuration(self) -> None:
        arguments = build_parser().parse_args(
            ["train", "--manifest", "manifest.csv", "--precision", "bfloat16"]
        )
        config = train.config_from_arguments(arguments)
        self.assertEqual(config.precision, "bfloat16")

    def test_train_parser_exposes_decoupled_backbone_and_embedding_dimensions(
        self,
    ) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--backbone-output-channels",
                "320",
                "--embedding-dim",
                "160",
            ]
        )
        config = train.config_from_arguments(arguments)
        self.assertEqual(config.backbone_output_channels, 320)
        self.assertEqual(config.embedding_dim, 160)

    def test_benchmark_parser_and_handler_are_owned_by_benchmark_module(self) -> None:
        arguments = build_parser().parse_args(
            ["benchmark", "--device", "cpu", "--iterations", "1"]
        )
        self.assertIs(arguments.command_handler, benchmark.run)
        self.assertEqual(arguments.device, "cpu")
        self.assertEqual(arguments.iterations, 1)
        self.assertEqual(arguments.mode, "eager")
        self.assertEqual(arguments.classification_head, "cosface")

    def test_scaffold_commands_dispatch_without_training_arguments(self) -> None:
        self.assertEqual(main(["validate"]), 0)
        self.assertEqual(main(["deploy"]), 0)


class BenchmarkCompatibilityTests(unittest.TestCase):
    def test_standalone_benchmark_parser_matches_top_level_options(self) -> None:
        arguments = benchmark.build_parser().parse_args(
            ["--device", "cpu", "--warmup-iterations", "2", "--iterations", "3"]
        )
        self.assertEqual(arguments.device, "cpu")
        self.assertEqual(arguments.warmup_iterations, 2)
        self.assertEqual(arguments.iterations, 3)
        self.assertEqual(arguments.mode, "eager")

    def test_benchmark_parser_accepts_optimized_mode(self) -> None:
        arguments = benchmark.build_parser().parse_args(["--mode", "optimized"])
        self.assertEqual(arguments.mode, "optimized")

    def test_eager_mode_does_not_call_model_optimizer(self) -> None:
        arguments = benchmark.build_parser().parse_args(
            [
                "--device",
                "cpu",
                "--num-classes",
                "3",
                "--width",
                "4",
                "--stage-depths",
                "1",
                "1",
                "1",
                "--warmup-iterations",
                "1",
                "--iterations",
                "1",
            ]
        )
        timing = {
            "latency_mean_ms": 1.0,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_p99_ms": 1.0,
            "samples_per_second": 1000.0,
        }
        output = io.StringIO()
        with (
            patch("hccr.commands.benchmark.optimize_model_for_inference") as optimize,
            patch("hccr.commands.benchmark.benchmark", return_value=timing),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(benchmark.run(arguments), 0)
        optimize.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["mode"], "eager")
        self.assertEqual(report["optimization_transforms"], [])
        self.assertEqual(report["deploy_transforms"], [])
        self.assertIn("torch_inference_mode", report["shared_inference_behavior"])

    def test_optimized_mode_calls_model_optimizer(self) -> None:
        model = torch.nn.Identity()
        prepared = torch.nn.Identity()
        with patch(
            "hccr.commands.benchmark.optimize_model_for_inference",
            return_value=prepared,
        ) as optimize:
            result = benchmark._prepare_benchmark_model(
                model, "optimized", torch.float32
            )
        optimize.assert_called_once_with(model)
        self.assertIs(result, prepared)
        self.assertFalse(result.training)

    def test_eager_mode_rejects_cuda_graph(self) -> None:
        arguments = benchmark.build_parser().parse_args(
            ["--device", "cpu", "--cuda-graph"]
        )
        with self.assertRaisesRegex(ValueError, "--mode optimized"):
            benchmark.run(arguments)
