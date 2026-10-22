from pathlib import Path
from unittest import TestCase

from hccr.experiments.cli import build_parser
from hccr.experiments.runner import build_jobs
from hccr.experiments.spec import load_experiment_spec
from hccr.tda.experiment import load_config

REPO = Path(__file__).resolve().parents[1]


class FullClassSuiteConfigTests(TestCase):
    def test_full_loss_jobs_are_only_cross_entropy_and_arcface(self):
        args = build_parser().parse_args(
            ["--config", str(REPO / "configs/experiment/loss_ablation_full_seed7.yaml")]
        )
        spec = load_experiment_spec(args, REPO)
        jobs = build_jobs(spec)
        self.assertEqual([job.variant for job in jobs], ["cross_entropy", "arcface"])
        self.assertEqual([job.seed for job in jobs], [7, 7])
        self.assertEqual(
            [job.train_args["classification_head"] for job in jobs],
            ["softmax", "arcface"],
        )
        for job in jobs:
            self.assertEqual(job.train_args["max_classes"], 7186)
            self.assertEqual(job.train_args["num_classes"], 7186)
            self.assertEqual(job.train_args["evaluation_policy"], "final_test")
            self.assertEqual(job.train_args["dataset_backend"], "lmdb")
            self.assertEqual(job.train_args["num_workers"], 4)
            self.assertEqual(job.train_args["device"], "cuda")
            self.assertEqual(job.train_args["epochs"], 20)
        varying = {"classification_head", "angular_margin", "margin_warmup_ratio"}
        self.assertEqual(
            {
                key: value
                for key, value in jobs[0].train_args.items()
                if key not in varying
            },
            {
                key: value
                for key, value in jobs[1].train_args.items()
                if key not in varying
            },
        )

    def test_full_ph_has_all_controls_and_no_sample_limits(self):
        config = load_config(REPO / "configs/experiment/stage2_ph_full_seed7.yaml")
        self.assertEqual(config.max_classes, 7186)
        self.assertEqual(config.seeds, (7,))
        self.assertEqual(config.dataset_backend, "lmdb")
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.run_dir, "experiments/20260913T114748Z-d2e5e0a9")
        self.assertEqual(
            config.variants,
            (
                "original",
                "cnn",
                "gap",
                "flat",
                "statistics",
                "h0",
                "h0_h1",
                "mean_h0_h1",
            ),
        )
        self.assertEqual(config.projection_epochs, 20)
        self.assertEqual(config.head_epochs, 8)
        self.assertIsNone(config.max_train_samples)
        self.assertIsNone(config.max_validation_samples)
        self.assertIsNone(config.max_holdout_samples)
