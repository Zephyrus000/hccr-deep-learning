"""Thin command-line entry point for HCCR workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hccr import __version__
from hccr.experiments import compare_runs
from hccr.training import TrainingConfig, run_training
from hccr.utils.experiment import write_json

COMMANDS = ("prepare-data", "train", "evaluate", "predict", "tune", "compare-runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hccr")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command_parser = subparsers.add_parser(
            command, help=f"Run the {command} workflow."
        )
        if command == "train":
            command_parser.add_argument("--manifest", type=Path, required=True)
            command_parser.add_argument(
                "--output-dir", type=Path, default=Path("experiments")
            )
            command_parser.add_argument("--num-classes", type=int, default=7186)
            command_parser.add_argument("--epochs", type=int, default=10)
            command_parser.add_argument("--batch-size", type=int, default=64)
            command_parser.add_argument("--learning-rate", type=float, default=1e-3)
            command_parser.add_argument("--weight-decay", type=float, default=1e-4)
            command_parser.add_argument("--image-size", type=int, default=64)
            command_parser.add_argument("--width", type=int, default=32)
            command_parser.add_argument(
                "--device", choices=("auto", "cpu", "cuda"), default="auto"
            )
            command_parser.add_argument("--seed", type=int, default=7)
            command_parser.add_argument("--num-workers", type=int, default=0)
            command_parser.add_argument(
                "--scheduler", choices=("none", "cosine", "plateau"), default="cosine"
            )
            command_parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
            command_parser.add_argument("--scheduler-patience", type=int, default=3)
            command_parser.add_argument(
                "--early-stopping-patience", type=int, default=8
            )
            command_parser.add_argument(
                "--early-stopping-min-delta", type=float, default=0.0
            )
            command_parser.add_argument("--overfit-samples", type=int)
            command_parser.add_argument("--overfit-check", action="store_true")
            command_parser.add_argument("--max-classes", type=int)
            command_parser.add_argument("--class-subset-seed", type=int, default=7)
            command_parser.add_argument("--center-by-centroid", action="store_true")
            command_parser.add_argument("--otsu-binarize", action="store_true")
            command_parser.add_argument("--median-filter-size", type=int)
            command_parser.add_argument(
                "--benchmark-warmup-iterations", type=int, default=20
            )
            command_parser.add_argument("--benchmark-iterations", type=int, default=200)
            command_parser.add_argument("--benchmark-repetitions", type=int, default=5)
            command_parser.add_argument(
                "--bn-recalibration-batches", type=int, default=0
            )
            command_parser.add_argument(
                "--validation-drop-threshold", type=float, default=0.05
            )
        if command == "compare-runs":
            command_parser.add_argument("--summary", type=Path, required=True)
            command_parser.add_argument("--baseline", required=True)
            command_parser.add_argument("--candidate", required=True)
            command_parser.add_argument("--min-top1-gain", type=float, default=0.0)
            command_parser.add_argument(
                "--max-p95-latency-ratio", type=float, default=1.0
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a bootstrap command without embedding workflow business logic."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "train":
        metrics = run_training(
            TrainingConfig(
                manifest_path=arguments.manifest,
                output_dir=arguments.output_dir,
                num_classes=arguments.num_classes,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                image_size=arguments.image_size,
                width=arguments.width,
                device=arguments.device,
                seed=arguments.seed,
                num_workers=arguments.num_workers,
                scheduler=arguments.scheduler,
                scheduler_min_lr=arguments.scheduler_min_lr,
                scheduler_patience=arguments.scheduler_patience,
                early_stopping_patience=(
                    arguments.early_stopping_patience
                    if arguments.early_stopping_patience > 0
                    else None
                ),
                early_stopping_min_delta=arguments.early_stopping_min_delta,
                max_train_samples=arguments.overfit_samples,
                overfit_check=arguments.overfit_check,
                max_classes=arguments.max_classes,
                class_subset_seed=arguments.class_subset_seed,
                center_by_centroid=arguments.center_by_centroid,
                otsu_binarize=arguments.otsu_binarize,
                median_filter_size=arguments.median_filter_size,
                benchmark_warmup_iterations=arguments.benchmark_warmup_iterations,
                benchmark_iterations=arguments.benchmark_iterations,
                benchmark_repetitions=arguments.benchmark_repetitions,
                bn_recalibration_batches=arguments.bn_recalibration_batches,
                validation_drop_threshold=arguments.validation_drop_threshold,
            )
        )
        print(metrics)
        return 0
    if arguments.command == "compare-runs":
        verdict = compare_runs(
            arguments.summary,
            arguments.baseline,
            arguments.candidate,
            arguments.min_top1_gain,
            arguments.max_p95_latency_ratio,
        )
        write_json(arguments.summary.with_name("comparison.json"), verdict)
        print(verdict)
        return 0 if verdict["accepted"] else 1
    print(f"hccr {arguments.command}: workflow scaffold is ready for implementation.")
    return 0
