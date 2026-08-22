"""Config-driven train workflow used by the CLI."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import getstate as random_state
from random import seed as random_seed
from random import setstate as restore_random_state

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
from hccr.training.artifacts import (
    file_digest,
    save_checkpoint,
    save_recalibrated_checkpoint,
    update_checkpoint_metrics,
)
from hccr.training.callbacks import EarlyStopping
from hccr.training.diagnostics import (
    profile_model,
    recalibrate_batch_norm,
    summarize_batch_norm_state,
    summarize_validation_stability,
    write_training_diagnostics,
)
from hccr.training.losses import build_classification_loss
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
    width: int = 64
    dropout: float = 0.1
    stage_depths: tuple[int, int, int] = (1, 2, 2)
    classification_head: str = "cosface"
    label_smoothing: float = 0.0
    logit_scale: float = 32.0
    angular_margin: float = 0.1
    margin_warmup_epochs: int = 3
    device: str = "auto"
    seed: int = 7
    num_workers: int = 0
    dataloader_start_method: str = "auto"
    prefetch_factor: int = 1
    persistent_workers: bool = True
    worker_timeout_seconds: float = 120.0
    max_train_samples: int | None = None
    overfit_check: bool = False
    max_classes: int | None = None
    class_subset_seed: int = 7
    scheduler: str = "cosine"
    scheduler_min_lr: float = 1e-6
    scheduler_patience: int = 3
    early_stopping_patience: int | None = 8
    early_stopping_min_delta: float = 0.0
    benchmark_warmup_iterations: int = 20
    benchmark_iterations: int = 200
    benchmark_repetitions: int = 5
    bn_recalibration_batches: int = 0
    validation_drop_threshold: float = 0.05


def run_training(config: TrainingConfig) -> dict[str, float]:
    _validate_training_config(config)
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
    train_transform = TrainPreprocessor(config.image_size)
    train_set = HCCRDataset(
        config.manifest_path,
        "train",
        train_transform,
        class_id_map,
    )
    training_class_support = _training_class_support(train_set)
    valid_set = HCCRDataset(
        config.manifest_path,
        "validation",
        EvalPreprocessor(config.image_size),
        class_id_map,
    )
    data_root = train_set.root
    if class_id_map is not None:
        write_json(output_dir / "class_subset.json", {"class_id_map": class_id_map})
    _write_label_mapping(output_dir, config.manifest_path, class_id_map)
    label_mapping = {
        int(class_id): label
        for class_id, label in json.loads(
            (output_dir / "labels.json").read_text(encoding="utf-8")
        )["labels"].items()
    }
    _save_preprocessing_gallery(
        train_set,
        EvalPreprocessor(config.image_size),
        output_dir,
        "preprocessing_gallery.png",
    )
    saved_random_state = random_state()
    random_seed(config.seed)
    _save_preprocessing_gallery(
        train_set,
        train_transform,
        output_dir,
        "augmentation_gallery.png",
    )
    restore_random_state(saved_random_state)
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
                EvalPreprocessor(config.image_size),
                class_id_map,
            ),
            range(len(train_set)),
        )
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = _data_loader_options(config, device)
    logger.info(
        "data loader workers=%s start_method=%s persistent=%s prefetch=%s "
        "timeout_seconds=%s pin_memory=%s",
        config.num_workers,
        loader_options.get("multiprocessing_context", "platform_default"),
        loader_options.get("persistent_workers", False),
        loader_options.get("prefetch_factor"),
        loader_options.get("timeout", 0),
        loader_options["pin_memory"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.batch_size,
        **loader_options,
    )
    calibration_loader = None
    if config.bn_recalibration_batches:
        calibration_set = HCCRDataset(
            config.manifest_path,
            "train",
            EvalPreprocessor(config.image_size),
            class_id_map,
        )
        if config.max_train_samples is not None:
            calibration_set = Subset(
                calibration_set,
                range(min(config.max_train_samples, len(calibration_set))),
            )
        calibration_samples = min(
            len(calibration_set), config.bn_recalibration_batches * config.batch_size
        )
        calibration_indices = torch.randperm(
            len(calibration_set),
            generator=torch.Generator().manual_seed(config.seed + 1),
        )[:calibration_samples].tolist()
        calibration_set = Subset(calibration_set, calibration_indices)
        calibration_loader = DataLoader(
            calibration_set,
            batch_size=config.batch_size,
            **loader_options,
        )
    model = build_model(
        "efficient_hccr",
        num_classes=active_num_classes,
        width=config.width,
        dropout=config.dropout,
        stage_depths=config.stage_depths,
        classification_head=config.classification_head,
        logit_scale=config.logit_scale,
        angular_margin=config.angular_margin,
    ).to(device)
    resource_profile = profile_model(
        model,
        config.image_size,
        device,
        output_dir,
        config.benchmark_warmup_iterations,
        config.benchmark_iterations,
        config.benchmark_repetitions,
        EvalPreprocessor(config.image_size),
    )
    logger.info("resource profile=%s", resource_profile)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = build_classification_loss(config.label_smoothing)
    scheduler = _build_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        config.early_stopping_patience, config.early_stopping_min_delta
    )
    best_metrics: dict[str, float] = {"top1": 0.0, "top5": 0.0}
    curves: list[dict] = []
    validation_stability = None
    for epoch in range(1, config.epochs + 1):
        margin_multiplier = _margin_multiplier(
            epoch, config.margin_warmup_epochs, config.classification_head
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            margin_multiplier,
        )
        batch_norm = summarize_batch_norm_state(model)
        metrics = evaluate(
            model, valid_loader, device, class_support=training_class_support
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
                "batch_norm": batch_norm,
                "next_learning_rate": current_learning_rate,
            }
        )
        validation_stability = summarize_validation_stability(
            curves, config.validation_drop_threshold
        )
        write_curves(output_dir, curves)
        write_training_diagnostics(output_dir, curves)
        write_json(output_dir / "validation_stability.json", validation_stability)
        save_learning_curves(output_dir, curves)
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
                    "schema_version": 2,
                    "model": {
                        "name": "efficient_hccr",
                        "in_channels": 1,
                        "effective_input_channels": model.effective_input_channels,
                        "width": config.width,
                        "stage_depths": list(config.stage_depths),
                        "dropout": config.dropout,
                        "classification_head": config.classification_head,
                        "logit_scale": config.logit_scale,
                        "angular_margin": config.angular_margin,
                        "num_classes": active_num_classes,
                    },
                    "training": {
                        "loss": "cross_entropy",
                        "label_smoothing": config.label_smoothing,
                        "margin_warmup_epochs": config.margin_warmup_epochs,
                        "augmentation": "random_affine_and_blur",
                    },
                    "preprocess": {
                        "image_size": config.image_size,
                        "margin": 4,
                    },
                    "manifest_digest": file_digest(config.manifest_path),
                    "labels_digest": file_digest(output_dir / "labels.json"),
                    "class_subset_digest": (
                        file_digest(output_dir / "class_subset.json")
                        if class_id_map is not None
                        else None
                    ),
                    "metrics": metrics,
                },
            )
            logger.info("best checkpoint saved epoch=%s", epoch)
        write_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "best": best_metrics,
                "latest": metrics,
                "validation_stability": validation_stability,
            },
        )
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
    model.load_state_dict(
        torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=True)
    )
    diagnostic_metrics = evaluate(
        model,
        valid_loader,
        device,
        output_dir,
        data_root,
        label_mapping,
        training_class_support,
    )
    if diagnostic_metrics["top1"] != best_metrics["top1"]:
        raise RuntimeError("best checkpoint metrics do not match diagnostic evaluation")
    best_metrics = {**best_metrics, **diagnostic_metrics}
    update_checkpoint_metrics(output_dir, best_metrics)
    recalibration_report = None
    if calibration_loader is not None:
        recalibrated_model = deepcopy(model)
        recalibration = recalibrate_batch_norm(
            recalibrated_model,
            calibration_loader,
            device,
            config.bn_recalibration_batches,
        )
        recalibrated_metrics = evaluate(
            recalibrated_model,
            valid_loader,
            device,
            output_dir / "bn_recalibrated",
            data_root,
            label_mapping,
            training_class_support,
        )
        recalibration_report = {
            "source_checkpoint": "checkpoint.pt",
            "checkpoint": "checkpoint_recalibrated.pt",
            "top1_delta": recalibrated_metrics["top1"] - best_metrics["top1"],
            "top5_delta": recalibrated_metrics["top5"] - best_metrics["top5"],
            "metrics": recalibrated_metrics,
            "batch_norm": recalibration,
        }
        save_recalibrated_checkpoint(
            recalibrated_model,
            output_dir,
            recalibrated_metrics,
            recalibration_report,
        )
        write_json(output_dir / "bn_recalibration.json", recalibration_report)
        save_learning_curves(
            output_dir,
            curves,
            recalibrated={
                "epoch": validation_stability["best_epoch"],
                "top1": recalibrated_metrics["top1"],
            },
        )
        logger.info(
            "BN recalibration completed batches=%s samples=%s top1=%.6f delta=%.6f",
            recalibration["batches"],
            recalibration["samples"],
            recalibrated_metrics["top1"],
            recalibration_report["top1_delta"],
        )
        del recalibrated_model
    write_json(
        output_dir / "metrics.json",
        {
            "run_id": run_id,
            "best": best_metrics,
            "latest": metrics,
            "validation_stability": validation_stability,
            "bn_recalibrated": (
                recalibration_report["metrics"] if recalibration_report else None
            ),
            "bn_recalibration": recalibration_report,
        },
    )
    _append_experiment_summary(
        config.output_dir, run_id, config, best_metrics, resource_profile, curves
    )
    logger.info("training completed run_id=%s best_metrics=%s", run_id, best_metrics)
    close_logging(logger)
    return best_metrics


def _validate_training_config(config: TrainingConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if (
        min(
            config.benchmark_warmup_iterations,
            config.benchmark_iterations,
            config.benchmark_repetitions,
        )
        < 1
    ):
        raise ValueError(
            "benchmark warm-up, iterations and repetitions must be positive"
        )
    if config.bn_recalibration_batches < 0:
        raise ValueError("bn_recalibration_batches must be non-negative")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.dataloader_start_method not in {"auto", "spawn"}:
        raise ValueError("dataloader_start_method must be one of: auto, spawn")
    if config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if config.worker_timeout_seconds < 0:
        raise ValueError("worker_timeout_seconds must be non-negative")
    if config.validation_drop_threshold < 0:
        raise ValueError("validation_drop_threshold must be non-negative")
    if len(config.stage_depths) != 3 or any(depth < 1 for depth in config.stage_depths):
        raise ValueError("stage_depths must contain three positive values")
    if config.width < 1:
        raise ValueError("width must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.classification_head not in {"cosface", "arcface"}:
        raise ValueError("classification_head must be one of: cosface, arcface")
    if not 0 <= config.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if config.logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    if not 0 <= config.angular_margin < torch.pi / 2:
        raise ValueError("angular_margin must be in [0, pi/2)")
    if config.margin_warmup_epochs < 0:
        raise ValueError("margin_warmup_epochs must be non-negative")


def _initialize_data_worker(_worker_id: int) -> None:
    """Prevent each loader process from creating a full CPU thread pool."""
    torch.set_num_threads(1)


def _data_loader_options(config: TrainingConfig, device: str) -> dict:
    options = {
        "num_workers": config.num_workers,
        "pin_memory": device.startswith("cuda"),
    }
    if config.num_workers == 0:
        return options
    start_method = config.dataloader_start_method
    if start_method == "auto" and device.startswith("cuda"):
        start_method = "spawn"
    options.update(
        {
            "persistent_workers": config.persistent_workers,
            "prefetch_factor": config.prefetch_factor,
            "timeout": config.worker_timeout_seconds,
            "worker_init_fn": _initialize_data_worker,
        }
    )
    if start_method != "auto":
        options["multiprocessing_context"] = start_method
    return options


def _margin_multiplier(epoch: int, warmup_epochs: int, head: str) -> float:
    if head == "linear" or warmup_epochs == 0:
        return 1.0
    if warmup_epochs == 1:
        return 1.0
    return min(1.0, (epoch - 1) / (warmup_epochs - 1))


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
    filename: str,
) -> None:
    rows_by_label: dict[str, dict[str, str]] = {}
    for row in dataset.rows:
        rows_by_label.setdefault(row["unicode_label"], row)
        if len(rows_by_label) == 8:
            break
    images, labels = [], []
    for label, row in rows_by_label.items():
        with Image.open(dataset.root / row["source_file"]) as image:
            images.append(image.convert("L"))
            labels.append(_unicode_codepoint(label))
    if images:
        save_gallery(
            images,
            transform,
            output_dir / filename,
            labels,
        )


def _unicode_codepoint(label: str) -> str:
    return " ".join(f"U+{ord(character):04X}" for character in label)


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


def _training_class_support(dataset: HCCRDataset) -> dict[int, int]:
    support: dict[int, int] = {}
    for row in dataset.rows:
        class_id = int(row["class_id"])
        model_id = dataset.class_id_map[class_id] if dataset.class_id_map else class_id
        support[model_id] = support.get(model_id, 0) + 1
    return support


def _append_experiment_summary(
    experiments_dir: Path,
    run_id: str,
    config: TrainingConfig,
    metrics: dict[str, float],
    resource_profile: dict,
    curves: list[dict],
) -> None:
    benchmark = resource_profile["inference_benchmarks"][0]
    end_to_end = resource_profile["end_to_end_batch1_benchmark"]
    full_class = resource_profile["full_class_projection"]
    mac_coverage = resource_profile["mac_coverage"]
    summary_path = experiments_dir / "experiment_summary.csv"
    import csv

    fields = [
        "run_id",
        "model",
        "effective_input_channels",
        "width",
        "dropout",
        "stage_depths",
        "classification_head",
        "label_smoothing",
        "logit_scale",
        "angular_margin",
        "margin_warmup_epochs",
        "epochs",
        "image_size",
        "top1",
        "top5",
        "macro_recall",
        "head_recall",
        "mid_recall",
        "tail_recall",
        "expected_calibration_error",
        "parameter_count",
        "backbone_parameter_count",
        "head_parameter_count",
        "estimated_macs",
        "estimated_backbone_macs",
        "estimated_head_macs",
        "mac_coverage_complete",
        "unsupported_operator_types",
        "full_class_num_classes",
        "full_class_parameter_count",
        "full_class_estimated_macs",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "end_to_end_latency_p50_ms",
        "end_to_end_latency_p95_ms",
        "end_to_end_latency_p99_ms",
        "samples_per_second",
        "peak_inference_cuda_memory_mib",
        "peak_training_cuda_memory_mib",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "seed",
        "max_classes",
    ]
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
        if existing_fields != fields:
            with summary_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(existing_rows)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "model": "efficient_hccr",
                "effective_input_channels": resource_profile[
                    "effective_input_channels"
                ],
                "width": config.width,
                "dropout": config.dropout,
                "stage_depths": ",".join(map(str, config.stage_depths)),
                "classification_head": config.classification_head,
                "label_smoothing": config.label_smoothing,
                "logit_scale": config.logit_scale,
                "angular_margin": config.angular_margin,
                "margin_warmup_epochs": config.margin_warmup_epochs,
                "epochs": config.epochs,
                "image_size": config.image_size,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "macro_recall": metrics.get("macro_recall"),
                "head_recall": metrics.get("head_recall"),
                "mid_recall": metrics.get("mid_recall"),
                "tail_recall": metrics.get("tail_recall"),
                "expected_calibration_error": metrics.get("expected_calibration_error"),
                "parameter_count": resource_profile["parameter_count"],
                "backbone_parameter_count": resource_profile[
                    "backbone_parameter_count"
                ],
                "head_parameter_count": resource_profile["head_parameter_count"],
                "estimated_macs": resource_profile["estimated_macs"],
                "estimated_backbone_macs": resource_profile["estimated_backbone_macs"],
                "estimated_head_macs": resource_profile["estimated_head_macs"],
                "mac_coverage_complete": mac_coverage["complete"],
                "unsupported_operator_types": json.dumps(
                    mac_coverage["unsupported_operator_types"]
                ),
                "full_class_num_classes": full_class.get("num_classes"),
                "full_class_parameter_count": full_class.get("total_parameter_count"),
                "full_class_estimated_macs": full_class.get("total_macs"),
                "latency_p50_ms": benchmark["latency_p50_ms"],
                "latency_p95_ms": benchmark["latency_p95_ms"],
                "latency_p99_ms": benchmark["latency_p99_ms"],
                "end_to_end_latency_p50_ms": end_to_end["latency_p50_ms"],
                "end_to_end_latency_p95_ms": end_to_end["latency_p95_ms"],
                "end_to_end_latency_p99_ms": end_to_end["latency_p99_ms"],
                "samples_per_second": benchmark["samples_per_second"],
                "peak_inference_cuda_memory_mib": benchmark["peak_cuda_memory_mib"],
                "peak_training_cuda_memory_mib": max(
                    epoch.get("peak_cuda_memory_mib", 0.0) for epoch in curves
                ),
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": config.scheduler,
                "seed": config.seed,
                "max_classes": config.max_classes or config.num_classes,
            }
        )
