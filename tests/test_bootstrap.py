from __future__ import annotations

import unittest
from pathlib import Path

from hccr import __version__
from hccr.cli import build_parser, main
from hccr.config import DataConfig, ModelConfig, load_yaml
from hccr.utils import resolve_device


class BootstrapTests(unittest.TestCase):
    def test_package_version_is_exposed(self) -> None:
        self.assertEqual(__version__, "0.1.0")

    def test_default_config_is_loadable(self) -> None:
        config = load_yaml(Path("configs/data/default.yaml"))
        data = DataConfig(**config)
        self.assertEqual(data.image_size, 64)
        model = ModelConfig()
        self.assertEqual(model.name, "efficient_hccr")
        self.assertEqual(model.num_classes, 7186)

    def test_cli_scaffold_accepts_predict(self) -> None:
        self.assertEqual(main(["predict"]), 0)

    def test_train_cli_accepts_measurement_controls(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--benchmark-warmup-iterations",
                "25",
                "--benchmark-iterations",
                "300",
                "--benchmark-repetitions",
                "7",
                "--bn-recalibration-batches",
                "64",
                "--validation-drop-threshold",
                "0.03",
            ]
        )
        self.assertEqual(arguments.benchmark_warmup_iterations, 25)
        self.assertEqual(arguments.benchmark_iterations, 300)
        self.assertEqual(arguments.benchmark_repetitions, 7)
        self.assertEqual(arguments.bn_recalibration_batches, 64)
        self.assertEqual(arguments.validation_drop_threshold, 0.03)

    def test_cpu_device_is_always_available(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")


class ArchitectureCliTests(unittest.TestCase):
    def test_train_cli_promotes_multibranch_and_keeps_control_switch(self) -> None:
        promoted = build_parser().parse_args(["train", "--manifest", "manifest.csv"])
        control = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--no-reparameterize-depthwise",
            ]
        )
        self.assertTrue(promoted.reparameterize_depthwise)
        self.assertEqual(promoted.margin_warmup_ratio, 0.2)
        self.assertFalse(control.reparameterize_depthwise)

    def test_train_cli_accepts_three_stage_depths(self) -> None:
        arguments = build_parser().parse_args(
            ["train", "--manifest", "manifest.csv", "--stage-depths", "1", "2", "3"]
        )
        self.assertEqual(tuple(arguments.stage_depths), (1, 2, 3))

    def test_train_cli_accepts_deployable_depthwise_and_stem_ablation(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--stem-stride",
                "1",
                "--reparameterize-depthwise",
            ]
        )
        self.assertEqual(arguments.stem_stride, 1)
        self.assertTrue(arguments.reparameterize_depthwise)


class LossCliTests(unittest.TestCase):
    def test_train_cli_accepts_margin_head_options(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--classification-head",
                "arcface",
                "--label-smoothing",
                "0.05",
                "--logit-scale",
                "16",
                "--angular-margin",
                "0.1",
                "--margin-warmup-ratio",
                "0.2",
            ]
        )
        self.assertEqual(arguments.classification_head, "arcface")
        self.assertEqual(arguments.label_smoothing, 0.05)
        self.assertEqual(arguments.logit_scale, 16.0)
        self.assertEqual(arguments.angular_margin, 0.1)
        self.assertEqual(arguments.margin_warmup_ratio, 0.2)


class RetainedModelCliTests(unittest.TestCase):
    def test_train_cli_accepts_dropout(self) -> None:
        arguments = build_parser().parse_args(
            ["train", "--manifest", "manifest.csv", "--dropout", "0.2"]
        )
        self.assertEqual(arguments.dropout, 0.2)


class BatchSizeCliTests(unittest.TestCase):
    def test_train_cli_accepts_batch_aware_training_controls(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--reference-batch-size",
                "64",
                "--optimizer-step-policy",
                "reference_batch",
                "--learning-rate-scaling",
                "sqrt",
                "--lr-warmup-ratio",
                "0.1",
            ]
        )
        self.assertEqual(arguments.reference_batch_size, 64)
        self.assertEqual(arguments.optimizer_step_policy, "reference_batch")
        self.assertEqual(arguments.learning_rate_scaling, "sqrt")
        self.assertEqual(arguments.lr_warmup_ratio, 0.1)
