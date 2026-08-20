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
