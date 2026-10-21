"""Validate the runnable pilot commands against the existing image LMDB setup."""

from pathlib import Path
from unittest import TestCase

from hccr.experiment_runner import build_jobs, build_parser, load_experiment_spec
from hccr.tda.experiment import load_config

REPO = Path(__file__).resolve().parents[1]


class PilotConfigTests(TestCase):
    def test_stage2_config_requires_existing_image_lmdb_not_gnt(self) -> None:
        config = load_config(REPO / "configs/experiment/stage2_ph.yaml")
        self.assertEqual(config.manifest, "data/processed/casia_hwdb/manifest.csv")
        self.assertEqual(config.dataset_backend, "lmdb")
        self.assertEqual(config.max_classes, 1000)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.run_dir, "experiments/20260913T114748Z-d2e5e0a9")

    def test_loss_config_builds_three_matched_seed7_lmdb_training_jobs(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--config",
                str(REPO / "configs/experiment/loss_ablation_seed7.yaml"),
                "--dry-run",
            ]
        )
        spec = load_experiment_spec(arguments, REPO)
        jobs = build_jobs(spec)
        self.assertEqual(spec.manifest, REPO / "data/processed/casia_hwdb/manifest.csv")
        self.assertEqual(spec.full_class_num_classes, 7186)
        self.assertEqual([job.seed for job in jobs], [7, 7, 7])
        self.assertEqual(
            [job.train_args["classification_head"] for job in jobs],
            ["cosface", "softmax", "arcface"],
        )
        for job in jobs:
            self.assertEqual(job.train_args["dataset_backend"], "lmdb")
            self.assertEqual(job.train_args["num_classes"], 7186)
            self.assertEqual(job.train_args["max_classes"], 1000)
            self.assertEqual(job.train_args["class_subset_seed"], 7)
            self.assertEqual(job.train_args["evaluation_policy"], "validation_only")
            self.assertIn(str(spec.manifest), job.command)
        varying = {"classification_head", "angular_margin", "margin_warmup_ratio"}
        common = [
            {key: value for key, value in job.train_args.items() if key not in varying}
            for job in jobs
        ]
        self.assertEqual(common[0], common[1])
        self.assertEqual(common[0], common[2])
