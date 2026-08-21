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
        self.assertEqual(ModelConfig().num_classes, 7186)

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
    def test_train_cli_accepts_three_stage_depths(self) -> None:
        arguments = build_parser().parse_args(
            ["train", "--manifest", "manifest.csv", "--stage-depths", "1", "2", "3"]
        )
        self.assertEqual(tuple(arguments.stage_depths), (1, 2, 3))


class AttentionCliTests(unittest.TestCase):
    def test_train_cli_accepts_attention_and_placement(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--attention",
                "eca",
                "--attention-stages",
                "2",
                "3",
            ]
        )
        self.assertEqual(arguments.attention, "eca")
        self.assertEqual(tuple(arguments.attention_stages), (2, 3))


class CrossStageCliTests(unittest.TestCase):
    def test_train_cli_accepts_cross_stage_and_csp_options(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--cross-stage",
                "c_cbam",
                "--csp-stages",
                "3",
                "--csp-split-ratio",
                "0.5",
            ]
        )
        self.assertEqual(arguments.cross_stage, "c_cbam")
        self.assertEqual(tuple(arguments.csp_stages), (3,))
        self.assertEqual(arguments.csp_split_ratio, 0.5)
        residual_arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--cross-stage",
                "projected_residual",
            ]
        )
        self.assertEqual(residual_arguments.cross_stage, "projected_residual")


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
                "--margin-warmup-epochs",
                "3",
            ]
        )
        self.assertEqual(arguments.classification_head, "arcface")
        self.assertEqual(arguments.label_smoothing, 0.05)
        self.assertEqual(arguments.logit_scale, 16.0)
        self.assertEqual(arguments.angular_margin, 0.1)
        self.assertEqual(arguments.margin_warmup_epochs, 3)


class AugmentationCliTests(unittest.TestCase):
    def test_train_cli_accepts_deformation_probabilities(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--elastic-probability",
                "0.2",
                "--elastic-displacement-ratio",
                "0.01",
                "--erosion-probability",
                "0.05",
                "--dilation-probability",
                "0.05",
            ]
        )
        self.assertEqual(arguments.elastic_probability, 0.2)
        self.assertEqual(arguments.elastic_displacement_ratio, 0.01)
        self.assertEqual(arguments.erosion_probability, 0.05)
        self.assertEqual(arguments.dilation_probability, 0.05)


class DeterministicInputCliTests(unittest.TestCase):
    def test_train_cli_accepts_directional_input_mode(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--input-mode",
                "grayscale_gabor",
            ]
        )
        self.assertEqual(arguments.input_mode, "grayscale_gabor")

    def test_train_cli_accepts_input_polarity(self) -> None:
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "manifest.csv",
                "--input-polarity",
                "white_on_black",
            ]
        )
        self.assertEqual(arguments.input_polarity, "white_on_black")

    def test_train_cli_accepts_dropout(self) -> None:
        arguments = build_parser().parse_args(
            ["train", "--manifest", "manifest.csv", "--dropout", "0.2"]
        )
        self.assertEqual(arguments.dropout, 0.2)
