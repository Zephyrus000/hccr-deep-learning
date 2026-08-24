"""Training command parser and handler."""

from __future__ import annotations

import argparse
from pathlib import Path

from hccr.training import TrainingConfig, run_training

NAME = "train"
HELP = "Train an HCCR model from a dataset manifest."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments owned by the training workflow."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--num-classes", type=int, default=7186)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--stage-depths",
        type=int,
        nargs=3,
        metavar=("STAGE1", "STAGE2", "STAGE3"),
        default=(1, 2, 2),
    )
    parser.add_argument("--stem-stride", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--reparameterize-depthwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train 3x3, 1x3 and 3x1 depthwise branches and fuse them for deploy "
            "(enabled by default; use --no-reparameterize-depthwise for control)."
        ),
    )
    parser.add_argument(
        "--classification-head", choices=("cosface", "arcface"), default="cosface"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=32.0)
    parser.add_argument("--angular-margin", type=float, default=0.1)
    parser.add_argument("--margin-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dataset-backend",
        choices=("auto", "filesystem", "lmdb"),
        default="auto",
        help="Use images.lmdb automatically when present, or require a backend.",
    )
    parser.add_argument("--lmdb-path", type=Path)
    parser.add_argument(
        "--dataloader-start-method",
        choices=("auto", "spawn"),
        default="auto",
        help="Use spawn automatically for CUDA workers to avoid poison fork.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--scheduler", choices=("none", "cosine", "plateau"), default="cosine"
    )
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--overfit-samples", type=int)
    parser.add_argument("--overfit-check", action="store_true")
    parser.add_argument("--max-classes", type=int)
    parser.add_argument("--class-subset-seed", type=int, default=7)
    parser.add_argument("--benchmark-warmup-iterations", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    parser.add_argument("--bn-recalibration-batches", type=int, default=0)
    parser.add_argument("--validation-drop-threshold", type=float, default=0.05)


def config_from_arguments(arguments: argparse.Namespace) -> TrainingConfig:
    """Convert parsed arguments into the training domain configuration."""
    return TrainingConfig(
        manifest_path=arguments.manifest,
        output_dir=arguments.output_dir,
        num_classes=arguments.num_classes,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        image_size=arguments.image_size,
        width=arguments.width,
        dropout=arguments.dropout,
        stage_depths=tuple(arguments.stage_depths),
        stem_stride=arguments.stem_stride,
        reparameterize_depthwise=arguments.reparameterize_depthwise,
        classification_head=arguments.classification_head,
        label_smoothing=arguments.label_smoothing,
        logit_scale=arguments.logit_scale,
        angular_margin=arguments.angular_margin,
        margin_warmup_ratio=arguments.margin_warmup_ratio,
        device=arguments.device,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        dataset_backend=arguments.dataset_backend,
        lmdb_path=arguments.lmdb_path,
        dataloader_start_method=arguments.dataloader_start_method,
        prefetch_factor=arguments.prefetch_factor,
        persistent_workers=arguments.persistent_workers,
        worker_timeout_seconds=arguments.worker_timeout_seconds,
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
        benchmark_warmup_iterations=arguments.benchmark_warmup_iterations,
        benchmark_iterations=arguments.benchmark_iterations,
        benchmark_repetitions=arguments.benchmark_repetitions,
        bn_recalibration_batches=arguments.bn_recalibration_batches,
        validation_drop_threshold=arguments.validation_drop_threshold,
    )


def run(arguments: argparse.Namespace) -> int:
    """Run training from parsed CLI arguments."""
    print(run_training(config_from_arguments(arguments)))
    return 0
