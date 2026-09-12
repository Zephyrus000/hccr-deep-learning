"""Training command parser and handler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hccr.models import MODEL_NAMES
from hccr.training import TrainingConfig, run_training

NAME = "train"
HELP = "Train an HCCR model from a dataset manifest."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments owned by the training workflow."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--num-classes", type=int, default=7186)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="efficient_hccr",
        help="Model family: proposed EfficientHCCRNet or a fixed reference baseline.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=64,
        help=(
            "Batch size whose optimizer-update budget and learning rate are the "
            "baseline."
        ),
    )
    parser.add_argument(
        "--optimizer-step-policy",
        choices=("reference_batch", "configured_epochs"),
        default="reference_batch",
        help=(
            "Match the reference batch's optimizer-update budget (default), or "
            "preserve the configured physical epoch count."
        ),
    )
    parser.add_argument(
        "--learning-rate-scaling",
        choices=("none", "sqrt", "linear"),
        default="sqrt",
        help="Scale AdamW learning rate from the reference batch size.",
    )
    parser.add_argument(
        "--lr-warmup-ratio",
        type=float,
        default=0.05,
        help="Cosine scheduler fraction used for linear learning-rate warm-up.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--backbone-output-channels",
        type=int,
        help=(
            "Optional 1x1 late-stage backbone expansion before pooling. "
            "Keep unset for the width*4 baseline."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help=(
            "CosFace/ArcFace embedding dimension. Keep unset for the legacy "
            "width*4 value; set independently to cap full-class head growth."
        ),
    )
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
        "--classification-head",
        choices=("cosface", "arcface", "softmax"),
        default="cosface",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=32.0)
    parser.add_argument("--angular-margin", type=float, default=0.1)
    parser.add_argument("--margin-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
        help=(
            "Training/evaluation arithmetic. FP32 is reproducible default; "
            "use bfloat16 on supported CUDA GPUs or auto to select it."
        ),
    )
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one process per GPU under torchrun. Batch size is per GPU; "
            "the run metadata records the effective global batch."
        ),
    )
    parser.add_argument(
        "--distributed-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="DDP collective backend; auto selects NCCL for CUDA.",
    )
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
    parser.add_argument(
        "--evaluation-policy",
        choices=("validation_only", "final_test"),
        default="final_test",
        help=(
            "Skip held-out test data during selection, or evaluate it once at the end."
        ),
    )
    parser.add_argument(
        "--reproducibility-mode",
        choices=("seeded", "strict"),
        default="seeded",
        help="Seed stochastic state, or additionally require deterministic algorithms.",
    )
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
        model=arguments.model,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        reference_batch_size=arguments.reference_batch_size,
        optimizer_step_policy=arguments.optimizer_step_policy,
        learning_rate_scaling=arguments.learning_rate_scaling,
        lr_warmup_ratio=arguments.lr_warmup_ratio,
        weight_decay=arguments.weight_decay,
        image_size=arguments.image_size,
        width=arguments.width,
        backbone_output_channels=arguments.backbone_output_channels,
        embedding_dim=arguments.embedding_dim,
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
        precision=arguments.precision,
        distributed=arguments.distributed,
        distributed_backend=arguments.distributed_backend,
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
        evaluation_policy=arguments.evaluation_policy,
        reproducibility_mode=arguments.reproducibility_mode,
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
    metrics = run_training(config_from_arguments(arguments))
    evaluation_name = (
        "train_overfit"
        if arguments.overfit_check
        else (
            "validation" if arguments.evaluation_policy == "validation_only" else "test"
        )
    )
    summary = [
        f"{evaluation_name}_top1={metrics['top1']:.2%}",
        f"{evaluation_name}_top5={metrics['top5']:.2%}",
        f"macro_recall={metrics['macro_recall']:.2%}",
        f"tail_recall={metrics['tail_recall']:.2%}",
    ]
    if "expected_calibration_error" in metrics:
        summary.append(f"ece={metrics['expected_calibration_error']:.2%}")
    if os.environ.get("RANK", "0") == "0":
        print("final metrics | " + " | ".join(summary))
    return 0
