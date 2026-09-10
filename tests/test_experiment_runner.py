from __future__ import annotations

import tempfile
import unittest
from json import dumps
from pathlib import Path
from unittest.mock import patch

import torch

from hccr.experiment_runner import (
    _distribution,
    _model_from_run,
    _paired_metric_delta,
    _summarize,
    build_jobs,
    build_parser,
    load_experiment_spec,
    run_experiments,
)
from hccr.models import EfficientHCCRNet, build_model


class ExperimentRunnerTests(unittest.TestCase):
    def test_distributed_experiment_command_uses_torchrun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "distributed",
                    "--set",
                    "distributed=true",
                    "--torchrun-nproc-per-node",
                    "2",
                ]
            )
            spec = load_experiment_spec(arguments, Path(directory))
            job = build_jobs(spec)[0]

        self.assertEqual(spec.torchrun_nproc_per_node, 2)
        self.assertEqual(
            job.command[1:8],
            (
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "hccr",
                "train",
            ),
        )
        self.assertIn("--distributed", job.command)

    def test_distributed_experiment_rejects_a_variant_that_disables_ddp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "invalid-distributed",
                    "--set",
                    "distributed=true",
                    "--variant",
                    '{"name":"invalid","args":{"distributed":false}}',
                    "--torchrun-nproc-per-node",
                    "2",
                ]
            )

            with self.assertRaisesRegex(ValueError, "cannot disable"):
                load_experiment_spec(arguments, Path(directory))

    def test_checked_in_thesis_comparison_configs_build_expected_commands(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        legacy_arguments = build_parser().parse_args(
            ["--config", str(root / "configs/experiment/baseline.yaml")]
        )
        legacy_jobs = build_jobs(load_experiment_spec(legacy_arguments, root))
        self.assertEqual(
            [job.key for job in legacy_jobs], ["legacy_efficient_hccr/seed-7"]
        )
        self.assertIn("--no-reparameterize-depthwise", legacy_jobs[0].command)

        arguments = build_parser().parse_args(
            ["--config", str(root / "configs/experiment/thesis_model_comparison.yaml")]
        )
        jobs = build_jobs(load_experiment_spec(arguments, root))
        commands = {job.key: job.command for job in jobs}
        self.assertEqual(
            list(commands),
            [
                "new_efficient_hccr/seed-7",
                "resnet18/seed-7",
                "mobilenet_v3_small/seed-7",
                "efficient_hccr_without_reparameterization/seed-7",
                "efficient_hccr_without_cosface/seed-7",
            ],
        )
        new_command = commands["new_efficient_hccr/seed-7"]
        self.assertIn("efficient_hccr", new_command)
        self.assertEqual(new_command[new_command.index("--image-size") + 1], "96")
        self.assertEqual(new_command[new_command.index("--batch-size") + 1], "256")
        self.assertEqual(new_command[new_command.index("--width") + 1], "80")
        self.assertEqual(
            new_command[
                new_command.index("--stage-depths")
                + 1 : new_command.index("--stage-depths")
                + 4
            ],
            ("2", "3", "3"),
        )
        self.assertEqual(
            new_command[new_command.index("--backbone-output-channels") + 1], "320"
        )
        self.assertEqual(new_command[new_command.index("--embedding-dim") + 1], "256")
        self.assertEqual(new_command[new_command.index("--num-workers") + 1], "16")
        self.assertIn("--reparameterize-depthwise", new_command)
        self.assertIn("cosface", new_command)
        self.assertIn("resnet18", commands["resnet18/seed-7"])
        self.assertIn("mobilenet_v3_small", commands["mobilenet_v3_small/seed-7"])
        self.assertIn(
            "--no-reparameterize-depthwise",
            commands["efficient_hccr_without_reparameterization/seed-7"],
        )
        self.assertIn("softmax", commands["efficient_hccr_without_cosface/seed-7"])

    def test_model_from_run_restores_mobilenet_and_replaces_only_final_logits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            source = build_model("mobilenet_v3_small", num_classes=11).eval()
            torch.save(source.state_dict(), run_dir / "checkpoint.pt")
            (run_dir / "checkpoint_metadata.json").write_text(
                dumps(
                    {
                        "model": {
                            "name": "mobilenet_v3_small",
                            "num_classes": 11,
                            "in_channels": 1,
                            "classification_head": "softmax",
                        }
                    }
                ),
                encoding="utf-8",
            )
            restored = _model_from_run(run_dir, "cpu").eval()
            full_head = _model_from_run(run_dir, "cpu", num_classes_override=100)

        self.assertEqual(restored.classifier[-1].out_features, 11)
        self.assertEqual(full_head.classifier[-1].out_features, 100)
        inputs = torch.rand(2, 1, 32, 32)
        torch.testing.assert_close(source(inputs), restored(inputs))

    def test_model_from_run_restores_decoupled_projection_and_full_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            source = EfficientHCCRNet(
                num_classes=11,
                width=8,
                backbone_output_channels=40,
                embedding_dim=12,
                reparameterize_depthwise=False,
            ).eval()
            torch.save(source.state_dict(), run_dir / "checkpoint.pt")
            (run_dir / "checkpoint_metadata.json").write_text(
                dumps(
                    {
                        "model": {
                            "name": "efficient_hccr",
                            "num_classes": 11,
                            "in_channels": 1,
                            "width": 8,
                            "backbone_output_channels": 40,
                            "embedding_dim": 12,
                            "stage_depths": [1, 2, 2],
                            "stem_stride": 2,
                            "reparameterize_depthwise": False,
                            "dropout": 0.1,
                            "classification_head": "cosface",
                            "logit_scale": 32.0,
                            "angular_margin": 0.1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            restored = _model_from_run(run_dir, "cpu").eval()
            full_head = _model_from_run(run_dir, "cpu", num_classes_override=100)

        self.assertEqual(restored.backbone_output_channels, 40)
        self.assertEqual(restored.embedding_dim, 12)
        self.assertEqual(full_head.classifier.out_features, 100)
        inputs = torch.rand(2, 1, 32, 32)
        torch.testing.assert_close(source(inputs), restored(inputs))

    def test_custom_variants_expand_in_declared_order_without_cartesian_product(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "polarity",
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "experiments",
                    "--seeds",
                    "7",
                    "17",
                    "--set",
                    "max_classes=1000",
                    "--set",
                    "stage_depths=[1, 2, 3]",
                    "--variant",
                    '{"name":"cosface","args":{"classification_head":"cosface"}}',
                    "--variant",
                    '{"name":"arcface","args":{"classification_head":"arcface"}}',
                ]
            )
            spec = load_experiment_spec(arguments, root)
            jobs = build_jobs(spec)
        self.assertEqual(
            [job.key for job in jobs],
            [
                "cosface/seed-7",
                "cosface/seed-17",
                "arcface/seed-7",
                "arcface/seed-17",
            ],
        )
        self.assertEqual(jobs[0].train_args["stage_depths"], [1, 2, 3])
        self.assertEqual(jobs[0].train_args["evaluation_policy"], "validation_only")
        self.assertIn("--classification-head", jobs[-1].command)
        self.assertIn("arcface", jobs[-1].command)

    def test_dry_run_validates_commands_and_writes_plan_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "three-seeds",
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "experiments",
                    "--seeds",
                    "7",
                    "17",
                    "29",
                    "--set",
                    "max_classes=1000",
                ]
            )
            spec = load_experiment_spec(arguments, root)
            result = run_experiments(spec, dry_run=True)
            experiment_dir = root / "experiments" / "sweeps" / "three-seeds"
            self.assertEqual(result, {"status": "dry_run_complete", "jobs": 3})
            self.assertTrue((experiment_dir / "plan.json").is_file())
            self.assertTrue((experiment_dir / "status.json").is_file())

    def test_reserved_train_arguments_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "invalid",
                    "--set",
                    "seed=99",
                ]
            )
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_experiment_spec(arguments, Path(directory))

    def test_completed_run_consumes_nested_validation_and_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "experiments" / "20260909T000000Z-test"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                dumps(
                    {
                        "validation": {"best": {"top1": 0.8, "top5": 0.95}},
                        "test": {
                            "status": "completed",
                            "metrics": {"top1": 0.75, "top5": 0.93},
                        },
                    }
                ),
                encoding="utf-8",
            )
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "completed-run",
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "experiments",
                    "--seeds",
                    "7",
                ]
            )
            spec = load_experiment_spec(arguments, root)

            with (
                patch("hccr.experiment_runner._run_training_job", return_value=run_dir),
                patch("hccr.experiment_runner._profile_run", return_value={}),
            ):
                result = run_experiments(spec)

        record = result["records"][0]
        self.assertEqual(record["selection_metrics"]["top1"], 0.8)
        self.assertEqual(record["test_metrics"]["top1"], 0.75)
        self.assertEqual(result["variants"]["default"]["metrics"]["top1"]["mean"], 0.8)

    def test_current_metrics_prefer_selected_validation_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "experiments" / "20260909T000000Z-selected"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                dumps(
                    {
                        "validation": {
                            "split": "train_overfit",
                            "best": {"top1": 0.7},
                            "selected": {"top1": 0.72},
                        },
                        "test": {"status": "not_run", "metrics": None},
                    }
                ),
                encoding="utf-8",
            )
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "selected-run",
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "experiments",
                    "--seeds",
                    "7",
                ]
            )
            spec = load_experiment_spec(arguments, root)
            with (
                patch("hccr.experiment_runner._run_training_job", return_value=run_dir),
                patch("hccr.experiment_runner._profile_run", return_value={}),
            ):
                result = run_experiments(spec)

        self.assertEqual(result["records"][0]["selection_metrics"]["top1"], 0.72)
        self.assertEqual(result["records"][0]["selection_split"], "train_overfit")

    def test_distribution_reports_sample_uncertainty(self) -> None:
        distribution = _distribution([1.0, 2.0, 3.0])
        self.assertEqual(distribution["count"], 3)
        self.assertAlmostEqual(distribution["mean"], 2.0)
        self.assertAlmostEqual(distribution["sample_std"], 1.0)
        self.assertAlmostEqual(distribution["std"], 1.0)
        self.assertAlmostEqual(distribution["standard_error"], 1 / 3**0.5)
        self.assertAlmostEqual(distribution["ci95_low"], -0.4841377117)
        self.assertAlmostEqual(distribution["ci95_high"], 4.4841377117)

    def test_paired_delta_marks_mismatched_seed_sets(self) -> None:
        baseline = {"seed_metrics": {"7": {"top1": 0.7}, "17": {"top1": 0.8}}}
        candidate = {"seed_metrics": {"7": {"top1": 0.75}}}
        delta = _paired_metric_delta(baseline, candidate, "top1")
        self.assertFalse(delta["complete"])
        self.assertEqual(delta["seeds"], [7])
        self.assertEqual(delta["baseline_only_seeds"], [17])
        self.assertAlmostEqual(delta["mean"], 0.05)

    def test_summary_marks_missing_jobs_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = build_parser().parse_args(
                [
                    "--experiment-id",
                    "partial-run",
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "experiments",
                    "--seeds",
                    "7",
                    "17",
                ]
            )
            spec = load_experiment_spec(arguments, root)
            summary = _summarize(
                spec,
                [
                    {
                        "key": "default/seed-7",
                        "variant": "default",
                        "seed": 7,
                        "status": "completed",
                        "selection_metrics": {"top1": 0.8},
                        "profiles": {},
                    }
                ],
                "2026-09-09T00:00:00+00:00",
            )
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(
            summary["jobs"],
            {"planned": 2, "completed": 1, "failed": 0, "missing": 1},
        )
