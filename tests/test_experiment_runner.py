from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hccr.experiment_runner import (
    build_jobs,
    build_parser,
    load_experiment_spec,
    run_experiments,
)


class ExperimentRunnerTests(unittest.TestCase):
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
                    '{"name":"black","args":{"input_polarity":"black_on_white"}}',
                    "--variant",
                    '{"name":"white","args":{"input_polarity":"white_on_black"}}',
                ]
            )
            spec = load_experiment_spec(arguments, root)
            jobs = build_jobs(spec)
        self.assertEqual(
            [job.key for job in jobs],
            ["black/seed-7", "black/seed-17", "white/seed-7", "white/seed-17"],
        )
        self.assertEqual(jobs[0].train_args["stage_depths"], [1, 2, 3])
        self.assertIn("--input-polarity", jobs[-1].command)
        self.assertIn("white_on_black", jobs[-1].command)

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
