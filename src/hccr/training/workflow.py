"""Config-driven train workflow used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import seed as random_seed

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from hccr.data.dataset import HCCRDataset, select_class_subset
from hccr.data.manifest import read_manifest
from hccr.evaluation.evaluator import evaluate
from hccr.evaluation.reports import save_learning_curves
from hccr.models import build_model
from hccr.preprocessing import EvalPreprocessor, TrainPreprocessor
from hccr.preprocessing.gallery import save_gallery
from hccr.training.artifacts import file_digest, save_checkpoint
from hccr.training.callbacks import EarlyStopping
from hccr.training.diagnostics import profile_model, write_training_diagnostics
from hccr.training.trainer import train_epoch
from hccr.utils import close_logging, configure_logging, resolve_device
from hccr.utils.experiment import initialize_run, new_run_id, write_curves, write_json


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: Path
    output_dir: Path
    num_classes: int
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 64
    width: int = 32
    device: str = "auto"
    seed: int = 7
    num_workers: int = 0
    max_train_samples: int | None = None
    overfit_check: bool = False
    max_classes: int | None = None
    class_subset_seed: int = 7
    center_by_centroid: bool = False
    otsu_binarize: bool = False
    median_filter_size: int | None = None
    scheduler: str = "cosine"
    scheduler_min_lr: float = 1e-6
    scheduler_patience: int = 3
    early_stopping_patience: int | None = 8
    early_stopping_min_delta: float = 0.0


def run_training(config: TrainingConfig) -> dict[str, float]:
    device = resolve_device(config.device)
    random_seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    run_id = new_run_id()
    output_dir = config.output_dir / run_id
    logger = configure_logging(output_dir)
    logger.info("training started config=%s", config)
    initialize_run(
        output_dir,
        config,
        {
            "manifest_digest": file_digest(config.manifest_path),
            "resolved_device": device,
        },
        run_id,
    )
    class_id_map = (
        select_class_subset(
            config.manifest_path, config.max_classes, config.class_subset_seed
        )
        if config.max_classes is not None
        else None
    )
    active_num_classes = config.max_classes or config.num_classes
    preprocess_options = {
        "center_by_centroid": config.center_by_centroid,
        "otsu_binarize": config.otsu_binarize,
        "median_filter_size": config.median_filter_size,
    }
    train_set = HCCRDataset(
        config.manifest_path,
        "train",
        TrainPreprocessor(config.image_size, **preprocess_options),
        class_id_map,
    )
    valid_set = HCCRDataset(
        config.manifest_path,
        "validation",
        EvalPreprocessor(config.image_size, **preprocess_options),
        class_id_map,
    )
    if class_id_map is not None:
        write_json(output_dir / "class_subset.json", {"class_id_map": class_id_map})
    _write_label_mapping(output_dir, config.manifest_path, class_id_map)
    _save_preprocessing_gallery(
        train_set, EvalPreprocessor(config.image_size, **preprocess_options), output_dir
    )
    if config.max_train_samples is not None:
        train_set = Subset(
            train_set, range(min(config.max_train_samples, len(train_set)))
        )
    if config.overfit_check:
        if config.max_train_samples is None:
            raise ValueError("overfit_check requires max_train_samples")
        valid_set = Subset(
            HCCRDataset(
                config.manifest_path,
                "train",
                EvalPreprocessor(config.image_size, **preprocess_options),
                class_id_map,
            ),
            range(len(train_set)),
        )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    model = build_model(
        "efficient_hccr", num_classes=active_num_classes, width=config.width
    ).to(device)
    resource_profile = profile_model(model, config.image_size, device, output_dir)
    logger.info("resource profile=%s", resource_profile)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _build_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        config.early_stopping_patience, config.early_stopping_min_delta
    )
    best_metrics: dict[str, float] = {"top1": 0.0, "top5": 0.0}
    curves: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(
            model,
            valid_loader,
            device,
            output_dir,
            config.manifest_path.parents[2],
        )
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(metrics["top1"])
            else:
                scheduler.step()
        current_learning_rate = optimizer.param_groups[0]["lr"]
        curves.append(
            {
                "epoch": float(epoch),
                **train_metrics,
                **metrics,
                "next_learning_rate": current_learning_rate,
            }
        )
        write_curves(output_dir, curves)
        write_training_diagnostics(output_dir, curves)
        save_learning_curves(output_dir, curves)
        write_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "best": best_metrics, "latest": metrics},
        )
        logger.info(
            "epoch=%s train_loss=%.6f grad_norm=%.6f throughput=%.2f validation=%s",
            epoch,
            train_metrics["train_loss"],
            train_metrics["gradient_norm_mean"],
            train_metrics["train_samples_per_second"],
            metrics,
        )
        if metrics["top1"] >= best_metrics["top1"]:
            best_metrics = metrics
            save_checkpoint(
                model,
                output_dir,
                {
                    "schema_version": 1,
                    "model": {
                        "name": "efficient_hccr",
                        "width": config.width,
                        "num_classes": active_num_classes,
                    },
                    "preprocess": {"image_size": config.image_size},
                    "manifest_digest": file_digest(config.manifest_path),
                    "metrics": metrics,
                },
            )
            logger.info("best checkpoint saved epoch=%s", epoch)
        if early_stopping.update(metrics["top1"]):
            write_json(
                output_dir / "early_stopping.json",
                {
                    "stopped": True,
                    "stopped_epoch": epoch,
                    "best_top1": early_stopping.best_score,
                    "bad_epochs": early_stopping.bad_epochs,
                    "patience": config.early_stopping_patience,
                    "min_delta": config.early_stopping_min_delta,
                },
            )
            logger.info("early stopping at epoch=%s", epoch)
            break
    _append_experiment_summary(
        config.output_dir, run_id, config, best_metrics, resource_profile
    )
    logger.info("training completed run_id=%s best_metrics=%s", run_id, best_metrics)
    close_logging(logger)
    return best_metrics


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.scheduler_min_lr
        )
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
    raise ValueError("scheduler must be one of: none, cosine, plateau")


def _save_preprocessing_gallery(
    dataset: HCCRDataset,
    transform: EvalPreprocessor,
    output_dir: Path,
) -> None:
    images = []
    for row in dataset.rows[:8]:
        with Image.open(dataset.root / row["source_file"]) as image:
            images.append(image.convert("L"))
    if images:
        save_gallery(images, transform, output_dir / "preprocessing_gallery.png")


def _write_label_mapping(
    output_dir: Path, manifest_path: Path, class_id_map: dict[int, int] | None
) -> None:
    labels = {
        int(row["class_id"]): row["unicode_label"]
        for row in read_manifest(manifest_path)
    }
    output_labels = (
        {model_id: labels[class_id] for class_id, model_id in class_id_map.items()}
        if class_id_map is not None
        else labels
    )
    write_json(output_dir / "labels.json", {"labels": output_labels})


def _append_experiment_summary(
    experiments_dir: Path,
    run_id: str,
    config: TrainingConfig,
    metrics: dict[str, float],
    resource_profile: dict,
) -> None:
    benchmark = resource_profile["inference_benchmarks"][0]
    summary_path = experiments_dir / "experiment_summary.csv"
    import csv

    fields = [
        "run_id",
        "model",
        "width",
        "epochs",
        "image_size",
        "top1",
        "top5",
        "parameter_count",
        "estimated_macs",
        "latency_p50_ms",
        "latency_p95_ms",
        "samples_per_second",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "seed",
        "max_classes",
    ]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "model": "efficient_hccr",
                "width": config.width,
                "epochs": config.epochs,
                "image_size": config.image_size,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "parameter_count": resource_profile["parameter_count"],
                "estimated_macs": resource_profile["estimated_macs"],
                "latency_p50_ms": benchmark["latency_p50_ms"],
                "latency_p95_ms": benchmark["latency_p95_ms"],
                "samples_per_second": benchmark["samples_per_second"],
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": config.scheduler,
                "seed": config.seed,
                "max_classes": config.max_classes or config.num_classes,
            }
        )
