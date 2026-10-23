"""Config-driven train workflow used by the CLI."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import getstate as random_state
from random import seed as random_seed
from random import setstate as restore_random_state

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from hccr.artifacts import RunArtifacts, new_run_id, write_json
from hccr.artifacts.checkpoints import (
    file_digest,
)
from hccr.artifacts.summary import SummaryTable
from hccr.benchmarking.profiling import profile_model
from hccr.data.dataset import HCCRDataset, select_class_subset
from hccr.data.manifest import read_manifest, validate_training_manifest
from hccr.evaluation.evaluator import evaluate
from hccr.evaluation.reports import save_learning_curves
from hccr.models import MODEL_NAMES, build_model
from hccr.preprocessing import EvalPreprocessor, TrainPreprocessor
from hccr.preprocessing.batch import TensorBatchAugmenter
from hccr.preprocessing.gallery import save_gallery
from hccr.training.callbacks import EarlyStopping
from hccr.training.diagnostics import (
    recalibrate_batch_norm,
    summarize_batch_norm_state,
    summarize_validation_stability,
    write_training_diagnostics,
)
from hccr.training.distributed import (
    DistributedContext,
    destroy_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_model_for_training,
)
from hccr.training.input_cache import build_training_cache
from hccr.training.losses import build_classification_loss
from hccr.training.precision import PRECISION_NAMES, PrecisionContext, resolve_precision
from hccr.training.proposal import (
    classifier_options,
    initialize_training_model,
    manifest_class_labels,
    resolve_residual_mask,
    validate_allocation_initialization,
    validate_official_source,
)
from hccr.training.trainer import train_epoch
from hccr.utils import close_logging, configure_logging, resolve_device


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: Path
    output_dir: Path
    num_classes: int
    model: str = "efficient_hccr"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    reference_batch_size: int = 64
    optimizer_step_policy: str = "reference_batch"
    learning_rate_scaling: str = "sqrt"
    lr_warmup_ratio: float = 0.05
    weight_decay: float = 1e-4
    image_size: int = 64
    width: int = 64
    backbone_output_channels: int | None = None
    embedding_dim: int | None = None
    dropout: float = 0.1
    stage_depths: tuple[int, int, int] = (1, 2, 2)
    stem_stride: int = 2
    reparameterize_depthwise: bool = True
    classification_head: str = "cosface"
    classifier_structure: str = "dense"
    classifier_rank: int = 64
    residual_mask: Path | None = None
    residual_class_indices: tuple[int, ...] = ()
    gate_kind: str = "hadamard"
    gate_use_gap: bool = True
    gate_detach_gap: bool = True
    normalization_eps: float = 1e-6
    residual_regularization: float = 0.0
    gate_regularization: float = 0.0
    initialize_from_run: Path | None = None
    checkpoint_selection: str = "best_validation"
    freeze_backbone: bool = False
    require_official_data: bool = False
    train_rotation_degrees: float = 8.0
    train_translate_ratio: float = 0.08
    train_blur_probability: float = 0.1
    label_smoothing: float = 0.0
    logit_scale: float = 32.0
    angular_margin: float = 0.1
    margin_warmup_ratio: float = 0.2
    device: str = "auto"
    precision: str = "float32"
    distributed: bool = False
    distributed_backend: str = "auto"
    seed: int = 7
    num_workers: int = 0
    augmentation_backend: str = "cpu"
    training_cache: str = "none"
    training_cache_max_gib: float = 32.0
    training_cache_reserve_gib: float = 16.0
    dataset_backend: str = "auto"
    lmdb_path: Path | None = None
    dataloader_start_method: str = "auto"
    prefetch_factor: int = 1
    persistent_workers: bool = True
    worker_timeout_seconds: float = 120.0
    max_train_samples: int | None = None
    overfit_check: bool = False
    evaluation_policy: str = "final_test"
    reproducibility_mode: str = "seeded"
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


@dataclass(frozen=True)
class BatchTrainingPlan:
    """Resolved update and learning-rate plan for one training run."""

    configured_epochs: int
    resolved_epochs: int
    batch_size: int
    steps_per_epoch: int
    reference_steps_per_epoch: int
    total_optimizer_steps: int
    reference_batch_size: int
    optimizer_step_policy: str
    learning_rate_scaling: str
    base_learning_rate: float
    resolved_learning_rate: float
    resolved_scheduler_min_lr: float
    warmup_steps: int
    world_size: int = 1
    effective_global_batch_size: int = 0


def run_training(config: TrainingConfig) -> dict[str, float]:
    """Run one training job, optionally under a torchrun DDP process group."""
    _validate_training_config(config)
    distributed = initialize_distributed(
        config.distributed, resolve_device(config.device), config.distributed_backend
    )
    try:
        precision = resolve_precision(config.precision, distributed.device)
        return _run_training(config, distributed, precision)
    finally:
        destroy_distributed(distributed)


def _run_training(
    config: TrainingConfig,
    distributed: DistributedContext,
    precision: PrecisionContext,
) -> dict[str, float]:
    reproducibility = _configure_reproducibility(config, distributed.rank)
    device = distributed.device
    if (
        config.augmentation_backend == "gpu"
        and not config.freeze_backbone
        and not device.startswith("cuda")
    ):
        raise ValueError("GPU augmentation requires a CUDA training device")
    manifest_rows = read_manifest(config.manifest_path)
    writer_info = validate_training_manifest(config.manifest_path, manifest_rows)
    if len(manifest_class_labels(manifest_rows)) != config.num_classes:
        raise ValueError("num_classes differs from the manifest vocabulary")
    if config.require_official_data:
        validate_official_source(config.manifest_path, manifest_rows)
    allocation = None
    if config.residual_mask is not None:
        allocation = resolve_residual_mask(
            config.residual_mask, config.manifest_path, manifest_rows
        )
        config = replace(
            config, residual_class_indices=tuple(allocation["residual_class_indices"])
        )
        if (
            config.initialize_from_run is None
            or file_digest(config.initialize_from_run / "checkpoint.pt")
            != allocation["provenance"]["low_rank"]["checkpoint_digest"]
        ):
            raise ValueError(
                "residual training must initialize from its allocation's low-rank pilot"
            )
    run_id = distributed.broadcast_object(
        new_run_id() if distributed.is_main_process else None
    )
    output_dir = config.output_dir / run_id
    artifacts = RunArtifacts(output_dir)
    logger = (
        configure_logging(output_dir)
        if distributed.is_main_process
        else logging.getLogger(f"hccr.rank{distributed.rank}")
    )
    if distributed.is_main_process:
        logger.info("training started config=%s", config)
        artifacts.initialize(
            config,
            {
                "manifest_digest": file_digest(config.manifest_path),
                "resolved_device": device,
                "reproducibility": reproducibility,
                "writer_provenance": writer_info,
                "distributed": _distributed_metadata(distributed),
                "precision": precision.metadata(),
            },
            run_id,
        )
        if allocation is not None:
            write_json(output_dir / "residual_allocation.json", allocation)
    distributed.barrier()
    class_id_map = (
        select_class_subset(
            config.manifest_path, config.max_classes, config.class_subset_seed
        )
        if config.max_classes is not None
        else None
    )
    active_num_classes = config.max_classes or config.num_classes
    train_transform = (
        EvalPreprocessor(config.image_size)
        if config.freeze_backbone or config.augmentation_backend == "gpu"
        else TrainPreprocessor(
            config.image_size,
            rotation_degrees=config.train_rotation_degrees,
            translate_ratio=config.train_translate_ratio,
            blur_probability=config.train_blur_probability,
        )
    )
    train_set = HCCRDataset(
        config.manifest_path,
        "train",
        train_transform,
        class_id_map,
        storage_backend=config.dataset_backend,
        lmdb_path=config.lmdb_path,
        metadata_mode="augmentations",
    )
    training_class_support = _training_class_support(train_set)
    selection_split = "validation"
    validation_set = HCCRDataset(
        config.manifest_path,
        selection_split,
        EvalPreprocessor(config.image_size),
        class_id_map,
        storage_backend=config.dataset_backend,
        lmdb_path=config.lmdb_path,
        metadata_mode="index",
    )
    test_set = None
    icdar_set = None
    data_root = None
    if not config.overfit_check and config.evaluation_policy == "final_test":
        test_set = HCCRDataset(
            config.manifest_path,
            "test",
            EvalPreprocessor(config.image_size),
            class_id_map,
            storage_backend=config.dataset_backend,
            lmdb_path=config.lmdb_path,
            metadata_mode="index",
        )
        data_root = (
            test_set.root
            if test_set.rows
            and (test_set.root / test_set.rows[0]["source_file"]).is_file()
            else None
        )
        if "icdar" in writer_info["available_splits"]:
            icdar_set = HCCRDataset(
                config.manifest_path,
                "icdar",
                EvalPreprocessor(config.image_size),
                class_id_map,
                storage_backend=config.dataset_backend,
                lmdb_path=config.lmdb_path,
                metadata_mode="index",
            )
    label_mapping: dict[int, str] = {}
    if distributed.is_main_process:
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
        if config.augmentation_backend == "gpu" and not config.freeze_backbone:
            _save_tensor_augmentation_gallery(train_set, config, device, output_dir)
        else:
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
        selection_split = "train_overfit"
        validation_set = Subset(
            HCCRDataset(
                config.manifest_path,
                "train",
                EvalPreprocessor(config.image_size),
                class_id_map,
                storage_backend=config.dataset_backend,
                lmdb_path=config.lmdb_path,
                metadata_mode="index",
            ),
            range(len(train_set)),
        )
    if not validation_set:
        raise ValueError(
            f"{selection_split} split must contain at least one sample; rebuild the "
            "manifest with a validation split before training"
        )
    if test_set is not None and not test_set:
        raise ValueError("test split must contain at least one sample")
    if icdar_set is not None and not icdar_set:
        raise ValueError("icdar split must contain at least one sample")
    generator = torch.Generator().manual_seed(config.seed + distributed.rank)
    loader_options = _data_loader_options(config, device)
    train_sampler = _train_sampler(train_set, config, distributed)
    order_generator = None
    if not distributed.enabled and (
        config.augmentation_backend == "gpu"
        or config.training_cache != "none"
        or config.freeze_backbone
    ):
        order_generator = torch.Generator()
        train_sampler = RandomSampler(train_set, generator=order_generator)
    if distributed.is_main_process:
        logger.info(
            "data loader workers=%s start_method=%s persistent=%s prefetch=%s "
            "timeout_seconds=%s pin_memory=%s distributed_world_size=%s",
            config.num_workers,
            loader_options.get("multiprocessing_context", "platform_default"),
            loader_options.get("persistent_workers", False),
            loader_options.get("prefetch_factor"),
            loader_options.get("timeout", 0),
            loader_options["pin_memory"],
            distributed.world_size,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **loader_options,
    )
    training_plan = _build_batch_training_plan(
        config, len(train_set), len(train_loader), distributed.world_size
    )
    if distributed.is_main_process:
        write_json(output_dir / "batch_training_plan.json", asdict(training_plan))
        logger.info("batch training plan=%s", training_plan)
    validation_loader = (
        DataLoader(
            validation_set,
            batch_size=config.batch_size,
            **loader_options,
        )
        if distributed.is_main_process
        else None
    )
    if distributed.is_main_process:
        logger.info(
            "selection validation split=%s samples=%s",
            selection_split,
            len(validation_set),
        )
    test_loader = None
    if test_set is not None and distributed.is_main_process:
        test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            **loader_options,
        )
        logger.info(
            "final test split=test samples=%s writer_provenance=%s",
            len(test_set),
            writer_info["availability"],
        )
    icdar_loader = None
    if icdar_set is not None and distributed.is_main_process:
        icdar_loader = DataLoader(
            icdar_set, batch_size=config.batch_size, **loader_options
        )
        logger.info("final external holdout split=icdar samples=%s", len(icdar_set))
    calibration_loader = None
    if config.bn_recalibration_batches and distributed.is_main_process:
        calibration_set = HCCRDataset(
            config.manifest_path,
            "train",
            EvalPreprocessor(config.image_size),
            class_id_map,
            storage_backend=config.dataset_backend,
            lmdb_path=config.lmdb_path,
            metadata_mode="index",
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
    model = _build_training_model(config, active_num_classes).to(device)
    if config.initialize_from_run is not None:
        initialization = initialize_training_model(
            model,
            config.initialize_from_run,
            config.manifest_path,
            manifest_class_labels(manifest_rows),
        )
        if allocation is not None:
            validate_allocation_initialization(
                allocation, config.initialize_from_run, initialization
            )
        if distributed.is_main_process:
            write_json(output_dir / "initialization.json", initialization)
    if config.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("classifier."):
                parameter.requires_grad_(False)
    cache_info = {"requested_placement": "none", "status": "disabled"}
    if config.training_cache != "none":
        cached_loader, cache_info = build_training_cache(
            train_set,
            config.batch_size,
            device,
            placement=config.training_cache,
            max_gib=config.training_cache_max_gib,
            reserve_gib=config.training_cache_reserve_gib,
            seed=config.seed,
            num_workers=min(config.num_workers, 4),
            model=model,
            image_size=config.image_size,
        )
        if cached_loader is not None:
            train_loader = cached_loader
    batch_augmenter = (
        TensorBatchAugmenter(
            config.image_size,
            rotation_degrees=config.train_rotation_degrees,
            translate_ratio=config.train_translate_ratio,
            blur_probability=config.train_blur_probability,
            seed=config.seed,
        )
        if config.augmentation_backend == "gpu" and not config.freeze_backbone
        else None
    )
    if distributed.is_main_process:
        write_json(
            output_dir / "input_pipeline.json",
            {
                "schema_version": 1,
                "cache": cache_info,
                "augmentation": batch_augmenter.metadata()
                if batch_augmenter is not None
                else {
                    "backend": "deterministic_eval"
                    if config.freeze_backbone
                    else "pil_affine_v1"
                },
                "fit_only": True,
                "manifest_digest": file_digest(config.manifest_path),
            },
        )
        logger.info("training input cache=%s", cache_info)
    distributed.barrier()
    training_model = wrap_model_for_training(model, distributed)
    base_model = unwrap_model(training_model)
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in training_model.parameters()
            if parameter.requires_grad
        ),
        lr=training_plan.resolved_learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = build_classification_loss(config.label_smoothing)
    scheduler = _build_scheduler(optimizer, config, training_plan)
    early_stopping = EarlyStopping(
        _resolved_early_stopping_patience(config, training_plan),
        config.early_stopping_min_delta,
    )
    best_validation_metrics: dict[str, float] = {"top1": 0.0, "top5": 0.0}
    curves: list[dict] = []
    validation_history: list[dict] = []
    validation_stability = None
    latest_validation_metrics: dict[str, float] | None = None
    global_step = 0
    for epoch in range(1, training_plan.resolved_epochs + 1):
        if distributed.enabled and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if order_generator is not None:
            order_generator.manual_seed(config.seed + epoch)
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        if batch_augmenter is not None:
            batch_augmenter.set_epoch(epoch)
        remaining_steps = training_plan.total_optimizer_steps - global_step
        max_batches = min(training_plan.steps_per_epoch, remaining_steps)
        margin_multiplier = _margin_multiplier(
            epoch,
            training_plan.resolved_epochs,
            config.margin_warmup_ratio,
            getattr(base_model, "classification_head", "softmax"),
        )
        train_metrics = train_epoch(
            training_model,
            train_loader,
            optimizer,
            device,
            criterion,
            margin_multiplier,
            scheduler if config.scheduler == "cosine" else None,
            max_batches,
            distributed,
            precision,
            residual_regularization=config.residual_regularization,
            gate_regularization=config.gate_regularization,
            freeze_backbone=config.freeze_backbone,
            batch_augmenter=batch_augmenter,
        )
        optimizer_step_start = global_step
        global_step += int(train_metrics["optimizer_steps"])
        validation_metrics = None
        if distributed.is_main_process:
            assert validation_loader is not None
            validation_metrics = evaluate(
                base_model,
                validation_loader,
                device,
                class_support=training_class_support,
                evaluation_name=selection_split,
                precision=precision,
            )
        validation_metrics = distributed.broadcast_object(validation_metrics)
        assert isinstance(validation_metrics, dict)
        if scheduler is not None and config.scheduler == "plateau":
            scheduler.step(validation_metrics["top1"])
        current_learning_rate = optimizer.param_groups[0]["lr"]
        should_stop = False
        if distributed.is_main_process:
            batch_norm = summarize_batch_norm_state(base_model)
            latest_validation_metrics = validation_metrics
            validation_history.append({"epoch": float(epoch), **validation_metrics})
            curve = {
                "epoch": float(epoch),
                "configured_epochs": config.epochs,
                "optimizer_step_start": optimizer_step_start,
                "optimizer_step_end": global_step,
                "selection_split": selection_split,
                **train_metrics,
                **validation_metrics,
                "batch_norm": batch_norm,
                "next_learning_rate": current_learning_rate,
            }
            validation_stability = summarize_validation_stability(
                validation_history, config.validation_drop_threshold
            )
            curves.append(curve)
            artifacts.write_curves(curves)
            write_training_diagnostics(output_dir, curves)
            write_json(output_dir / "validation_stability.json", validation_stability)
            save_learning_curves(output_dir, curves, evaluation_name=selection_split)
            if (
                config.checkpoint_selection == "final"
                or epoch == 1
                or validation_metrics["top1"] > best_validation_metrics["top1"]
            ):
                best_validation_metrics = validation_metrics
                artifacts.save_checkpoint(
                    base_model,
                    _checkpoint_metadata(
                        config,
                        training_plan,
                        base_model,
                        active_num_classes,
                        output_dir,
                        validation_metrics,
                        selection_split,
                        test_set is not None,
                        writer_info,
                        class_id_map,
                        distributed,
                        precision,
                        epoch=epoch,
                        completed_optimizer_steps=global_step,
                    ),
                )
                logger.info("best checkpoint saved epoch=%s", epoch)
            logger.info(
                _format_epoch_metrics(
                    epoch,
                    training_plan,
                    train_metrics,
                    validation_metrics,
                    current_learning_rate,
                    global_step,
                    best_validation_metrics["top1"],
                    selection_split,
                )
            )
            write_json(
                output_dir / "metrics.json",
                _metrics_payload(
                    run_id,
                    selection_split,
                    best_validation_metrics,
                    latest_validation_metrics,
                    validation_stability,
                    test_set is not None,
                    config,
                    precision,
                    writer_info,
                ),
            )
            should_stop = early_stopping.update(validation_metrics["top1"])
            if should_stop:
                write_json(
                    output_dir / "early_stopping.json",
                    {
                        "stopped": True,
                        "stopped_epoch": epoch,
                        "best_top1": early_stopping.best_score,
                        "bad_epochs": early_stopping.bad_epochs,
                        "patience": config.early_stopping_patience,
                        "resolved_patience": _resolved_early_stopping_patience(
                            config, training_plan
                        ),
                        "min_delta": config.early_stopping_min_delta,
                    },
                )
                logger.info("early stopping at epoch=%s", epoch)
        should_stop = bool(distributed.broadcast_object(should_stop))
        if should_stop:
            break
    if config.training_cache != "none":
        del train_loader, cached_loader
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    distributed.barrier()
    final_metrics = None
    if distributed.is_main_process:
        base_model.load_state_dict(
            torch.load(
                output_dir / "checkpoint.pt", map_location=device, weights_only=True
            )
        )
        selected_model = base_model
        selected_checkpoint_name = "checkpoint"
        selected_validation_metrics = best_validation_metrics
        recalibration_report = None
        if calibration_loader is not None:
            assert validation_loader is not None
            recalibrated_model = deepcopy(base_model)
            recalibration = recalibrate_batch_norm(
                recalibrated_model,
                calibration_loader,
                device,
                config.bn_recalibration_batches,
            )
            recalibrated_validation_metrics = evaluate(
                recalibrated_model,
                validation_loader,
                device,
                class_support=training_class_support,
                evaluation_name=selection_split,
                precision=precision,
            )
            recalibration_report = {
                "source_checkpoint": "checkpoint.pt",
                "checkpoint": "checkpoint_recalibrated.pt",
                "validation_top1_delta": (
                    recalibrated_validation_metrics["top1"]
                    - best_validation_metrics["top1"]
                ),
                "validation_top5_delta": (
                    recalibrated_validation_metrics["top5"]
                    - best_validation_metrics["top5"]
                ),
                "validation_metrics": recalibrated_validation_metrics,
                "batch_norm": recalibration,
            }
            if (
                recalibrated_validation_metrics["top1"]
                > best_validation_metrics["top1"]
            ):
                selected_model = recalibrated_model
                selected_checkpoint_name = "checkpoint_recalibrated"
                selected_validation_metrics = recalibrated_validation_metrics
                recalibration_report["selected"] = True
                artifacts.save_recalibrated_checkpoint(
                    recalibrated_model,
                    selected_validation_metrics,
                    recalibration_report,
                )
            else:
                recalibration_report["selected"] = False
                del recalibrated_model
            write_json(output_dir / "bn_recalibration.json", recalibration_report)
            logger.info(
                "BN recalibration validation batches=%s samples=%s top1=%.6f "
                "delta=%.6f selected=%s",
                recalibration["batches"],
                recalibration["samples"],
                recalibrated_validation_metrics["top1"],
                recalibration_report["validation_top1_delta"],
                recalibration_report["selected"],
            )
        final_test_metrics = None
        if test_loader is not None:
            final_test_metrics = evaluate(
                selected_model,
                test_loader,
                device,
                output_dir,
                data_root,
                label_mapping,
                training_class_support,
                evaluation_name="test",
                precision=precision,
            )
            artifacts.record_final_test_metrics(
                final_test_metrics, selected_checkpoint_name
            )
        final_metrics = final_test_metrics or selected_validation_metrics
        final_icdar_metrics = None
        if icdar_loader is not None:
            final_icdar_metrics = evaluate(
                selected_model,
                icdar_loader,
                device,
                output_dir / "icdar",
                data_root,
                label_mapping,
                training_class_support,
                evaluation_name="icdar",
                precision=precision,
            )
            artifacts.record_final_test_metrics(
                final_icdar_metrics, selected_checkpoint_name, split="icdar"
            )
            logger.info(
                "external holdout | %s",
                _format_final_metrics(final_icdar_metrics, "icdar"),
            )
        write_json(
            output_dir / "metrics.json",
            _metrics_payload(
                run_id,
                selection_split,
                best_validation_metrics,
                latest_validation_metrics,
                validation_stability,
                test_set is not None,
                config,
                precision,
                writer_info,
                selected_validation_metrics,
                final_test_metrics,
                selected_checkpoint_name,
                recalibration_report,
                final_icdar_metrics,
            ),
        )
        optimizer.state.clear()
        base_model.zero_grad(set_to_none=True)
        selected_model.zero_grad(set_to_none=True)
        del optimizer, scheduler, training_model, model, base_model
        resource_profile = profile_model(
            selected_model,
            config.image_size,
            device,
            output_dir,
            config.benchmark_warmup_iterations,
            config.benchmark_iterations,
            config.benchmark_repetitions,
            active_num_classes
            if config.require_official_data or config.classifier_structure != "dense"
            else 7186,
        )
        resource_profile["checkpoint_name"] = selected_checkpoint_name
        resource_profile["measurement_stage"] = "selected_checkpoint"
        resource_profile["checkpoint_sha256"] = file_digest(
            output_dir / f"{selected_checkpoint_name}.pt"
        )
        artifacts.write_json("resource_profile.json", resource_profile)
        _append_experiment_summary(
            config.output_dir,
            run_id,
            config,
            precision,
            training_plan,
            selection_split,
            selected_validation_metrics,
            final_test_metrics,
            resource_profile,
            curves,
            final_icdar_metrics,
        )
        logger.info(
            "training completed run_id=%s | %s",
            run_id,
            _format_final_metrics(
                final_metrics,
                "test" if final_test_metrics is not None else selection_split,
            ),
        )
        close_logging(logger)
    final_metrics = distributed.broadcast_object(final_metrics)
    assert isinstance(final_metrics, dict)
    return final_metrics


def _distributed_metadata(distributed: DistributedContext) -> dict[str, object]:
    return {
        "enabled": distributed.enabled,
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "local_rank": distributed.local_rank,
        "backend": distributed.backend,
        "device": distributed.device,
    }


def _checkpoint_metadata(
    config: TrainingConfig,
    training_plan: BatchTrainingPlan,
    model: torch.nn.Module,
    active_num_classes: int,
    output_dir: Path,
    validation_metrics: dict[str, float],
    selection_split: str,
    has_test_set: bool,
    writer_info: dict[str, object],
    class_id_map: dict[int, int] | None,
    distributed: DistributedContext,
    precision: PrecisionContext,
    *,
    epoch: int | None = None,
    completed_optimizer_steps: int | None = None,
) -> dict[str, object]:
    head = getattr(model, "classification_head", "softmax")
    has_margin_head = head in {"cosface", "arcface"}
    return {
        "schema_version": 9,
        "model": _model_metadata(model, config, active_num_classes),
        "training": {
            "loss": "cross_entropy",
            "seed": config.seed,
            "optimizer": "adamw",
            "weight_decay": config.weight_decay,
            "completed_optimizer_steps": completed_optimizer_steps,
            "freeze_backbone": config.freeze_backbone,
            "max_train_samples": config.max_train_samples,
            "overfit_check": config.overfit_check,
            "residual_regularization": config.residual_regularization,
            "gate_regularization": config.gate_regularization,
            "initialization_run": str(config.initialize_from_run)
            if config.initialize_from_run
            else None,
            "allocation_digest": file_digest(output_dir / "residual_allocation.json")
            if config.residual_mask
            else None,
            "label_smoothing": config.label_smoothing,
            "margin_schedule": "linear_warmup" if has_margin_head else "disabled",
            "margin_warmup_ratio": (
                config.margin_warmup_ratio if has_margin_head else 0.0
            ),
            "resolved_margin_warmup_epochs": (
                _margin_warmup_epochs(
                    training_plan.resolved_epochs, config.margin_warmup_ratio
                )
                if has_margin_head
                else 0
            ),
            "optimizer_step_policy": config.optimizer_step_policy,
            "reference_batch_size": config.reference_batch_size,
            "batch_size": config.batch_size,
            "effective_global_batch_size": training_plan.effective_global_batch_size,
            "distributed": _distributed_metadata(distributed),
            "precision": precision.metadata(),
            "configured_epochs": config.epochs,
            "resolved_epochs": training_plan.resolved_epochs,
            "steps_per_epoch": training_plan.steps_per_epoch,
            "reference_steps_per_epoch": training_plan.reference_steps_per_epoch,
            "total_optimizer_steps": training_plan.total_optimizer_steps,
            "learning_rate_scaling": config.learning_rate_scaling,
            "base_learning_rate": config.learning_rate,
            "resolved_learning_rate": training_plan.resolved_learning_rate,
            "lr_warmup_ratio": config.lr_warmup_ratio,
            "warmup_steps": training_plan.warmup_steps,
            "augmentation": "deterministic_eval"
            if config.freeze_backbone
            else (
                "tensor_affine_v1"
                if config.augmentation_backend == "gpu"
                else "random_affine_and_blur"
            ),
            "augmentation_backend": config.augmentation_backend,
            "training_cache": config.training_cache,
            "training_cache_max_gib": config.training_cache_max_gib,
            "training_cache_reserve_gib": config.training_cache_reserve_gib,
            "augmentation_parameters": {
                "rotation_degrees": config.train_rotation_degrees,
                "translate_ratio": config.train_translate_ratio,
                "blur_probability": config.train_blur_probability,
            },
        },
        "preprocess": {"image_size": config.image_size, "margin": 4},
        "selection": {
            "split": selection_split,
            "metrics": validation_metrics,
            "epoch": epoch,
            "policy": config.checkpoint_selection,
        },
        "test": {
            "split": "test" if has_test_set else None,
            "status": "not_run",
            "evaluation_policy": config.evaluation_policy,
            "writer_provenance": writer_info,
            "metrics": None,
        },
        "icdar": {
            "split": "icdar" if "icdar" in writer_info["available_splits"] else None,
            "status": "not_run",
            "evaluation_policy": config.evaluation_policy,
            "writer_provenance": writer_info,
            "metrics": None,
        },
        "manifest_digest": file_digest(config.manifest_path),
        "labels_digest": file_digest(output_dir / "labels.json"),
        "class_subset_digest": (
            file_digest(output_dir / "class_subset.json")
            if class_id_map is not None
            else None
        ),
    }


def _metrics_payload(
    run_id: str,
    selection_split: str,
    best_validation_metrics: dict[str, float],
    latest_validation_metrics: dict[str, float] | None,
    validation_stability: dict | None,
    has_test_set: bool,
    config: TrainingConfig,
    precision: PrecisionContext,
    writer_info: dict[str, object],
    selected_validation_metrics: dict[str, float] | None = None,
    final_test_metrics: dict[str, float] | None = None,
    selected_checkpoint_name: str | None = None,
    recalibration_report: dict | None = None,
    final_icdar_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    validation: dict[str, object] = {
        "split": selection_split,
        "best": best_validation_metrics,
        "latest": latest_validation_metrics,
        "stability": validation_stability,
    }
    if selected_validation_metrics is not None:
        validation["selected"] = selected_validation_metrics
        validation["selected_checkpoint"] = f"{selected_checkpoint_name}.pt"
    return {
        "schema_version": 2,
        "run_id": run_id,
        "precision": precision.metadata(),
        "validation": validation,
        "icdar": {
            "split": "icdar" if "icdar" in writer_info["available_splits"] else None,
            "status": "completed" if final_icdar_metrics is not None else "not_run",
            "reason": (
                None
                if final_icdar_metrics is not None
                else (
                    "split_unavailable"
                    if "icdar" not in writer_info["available_splits"]
                    else (
                        "validation_only_policy"
                        if config.evaluation_policy == "validation_only"
                        else (
                            "overfit_check"
                            if config.overfit_check
                            else "pending_final_evaluation"
                        )
                    )
                )
            ),
            "available": "icdar" in writer_info["available_splits"],
            "evaluation_policy": config.evaluation_policy,
            "writer_provenance": writer_info,
            "checkpoint": (
                f"{selected_checkpoint_name}.pt"
                if final_icdar_metrics is not None
                else None
            ),
            "metrics": final_icdar_metrics,
        },
        "test": {
            "split": "test" if final_test_metrics is not None else None,
            "status": "completed" if final_test_metrics is not None else "not_run",
            "reason": (
                None
                if final_test_metrics is not None
                else (
                    "validation_only_policy"
                    if config.evaluation_policy == "validation_only"
                    else "overfit_check"
                )
            ),
            "evaluation_policy": config.evaluation_policy,
            "writer_provenance": writer_info,
            "checkpoint": (
                f"{selected_checkpoint_name}.pt"
                if final_test_metrics is not None
                and selected_checkpoint_name is not None
                else None
            ),
            "metrics": final_test_metrics,
            "bn_recalibration": recalibration_report,
            "available": has_test_set,
        },
    }


def _validate_training_config(config: TrainingConfig) -> None:
    if config.model not in MODEL_NAMES:
        raise ValueError(f"model must be one of: {', '.join(MODEL_NAMES)}")
    if config.evaluation_policy not in {"validation_only", "final_test"}:
        raise ValueError(
            "evaluation_policy must be one of: validation_only, final_test"
        )
    if config.reproducibility_mode not in {"seeded", "strict"}:
        raise ValueError("reproducibility_mode must be one of: seeded, strict")
    if config.distributed_backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("distributed_backend must be one of: auto, nccl, gloo")
    if config.precision not in PRECISION_NAMES:
        raise ValueError(f"precision must be one of: {', '.join(PRECISION_NAMES)}")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if config.optimizer_step_policy not in {"configured_epochs", "reference_batch"}:
        raise ValueError(
            "optimizer_step_policy must be one of: configured_epochs, reference_batch"
        )
    if config.learning_rate_scaling not in {"none", "sqrt", "linear"}:
        raise ValueError("learning_rate_scaling must be one of: none, sqrt, linear")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= config.scheduler_min_lr <= config.learning_rate:
        raise ValueError("scheduler_min_lr must be in [0, learning_rate]")
    if not 0 <= config.lr_warmup_ratio < 1:
        raise ValueError("lr_warmup_ratio must be in [0, 1)")
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
    if config.augmentation_backend not in {"cpu", "gpu"}:
        raise ValueError("augmentation_backend must be cpu or gpu")
    if config.training_cache not in {"none", "auto", "cpu", "cuda"}:
        raise ValueError("unsupported training_cache placement")
    if (
        not math.isfinite(config.training_cache_max_gib)
        or config.training_cache_max_gib <= 0
        or not math.isfinite(config.training_cache_reserve_gib)
        or config.training_cache_reserve_gib < 0
    ):
        raise ValueError(
            "training cache budget must be positive and reserve nonnegative"
        )
    if config.training_cache != "none":
        if config.distributed:
            raise ValueError("training cache supports one process per GPU; disable DDP")
        if config.augmentation_backend == "cpu" and not config.freeze_backbone:
            raise ValueError(
                "training cache requires GPU augmentation or a frozen pilot"
            )
    if config.dataset_backend not in {"auto", "filesystem", "lmdb"}:
        raise ValueError("dataset_backend must be one of: auto, filesystem, lmdb")
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
    if config.stem_stride not in {1, 2}:
        raise ValueError("stem_stride must be 1 or 2")
    if config.width < 1:
        raise ValueError("width must be positive")
    if (
        config.backbone_output_channels is not None
        and config.backbone_output_channels < 1
    ):
        raise ValueError("backbone_output_channels must be positive when set")
    if config.embedding_dim is not None and config.embedding_dim < 1:
        raise ValueError("embedding_dim must be positive when set")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.classification_head not in {
        "cosface",
        "arcface",
        "softmax",
        "normalized_softmax",
    }:
        raise ValueError("unsupported classification_head")
    if config.classifier_structure not in {
        "dense",
        "low_rank",
        "static_residual",
        "dynamic_residual",
    }:
        raise ValueError("unsupported classifier_structure")
    if (
        config.classifier_structure != "dense"
        and config.classification_head == "softmax"
    ):
        raise ValueError(
            "factorized heads require normalized_softmax, cosface or arcface"
        )
    if config.classifier_rank < 1:
        raise ValueError("classifier_rank must be positive")
    if config.normalization_eps <= 0:
        raise ValueError("normalization_eps must be positive")
    if min(config.residual_regularization, config.gate_regularization) < 0:
        raise ValueError("regularization coefficients must be non-negative")
    if config.gate_kind not in {"hadamard", "cosine", "sample", "global", "class"}:
        raise ValueError("unsupported gate_kind")
    if config.classifier_structure == "dense" and (
        config.residual_regularization or config.gate_regularization
    ):
        raise ValueError("dense heads do not have residual/gate regularization")
    if config.freeze_backbone and config.bn_recalibration_batches:
        raise ValueError("frozen pilots cannot recalibrate backbone BatchNorm")
    if config.initialize_from_run is not None and config.max_classes is not None:
        raise ValueError("checkpoint initialization requires the full manifest mapping")
    if (
        config.require_official_data
        and config.classifier_structure.endswith("residual")
        and config.residual_mask is None
    ):
        raise ValueError("official residual runs require an audited allocation mask")
    if config.residual_mask is not None and config.max_classes is not None:
        raise ValueError("residual masks require the full manifest class mapping")
    if config.residual_mask is not None and config.residual_class_indices:
        raise ValueError("choose residual_mask or residual_class_indices, not both")
    if (
        config.classifier_structure.endswith("residual")
        and config.residual_mask is None
        and not config.residual_class_indices
    ):
        raise ValueError(
            "residual heads require a residual mask or explicit class indices"
        )
    if config.classifier_structure in {"dense", "low_rank"} and (
        config.residual_mask or config.residual_class_indices
    ):
        raise ValueError("only residual heads accept a residual mask")
    if config.require_official_data and config.max_classes is not None:
        raise ValueError(
            "official proposal runs require the complete audited vocabulary"
        )
    if config.require_official_data and (
        config.max_train_samples is not None or config.overfit_check
    ):
        raise ValueError(
            "official proposal runs cannot use limited/overfit training samples"
        )
    if config.freeze_backbone and config.initialize_from_run is None:
        raise ValueError("frozen-backbone pilots require initialize_from_run")
    if config.checkpoint_selection not in {"best_validation", "final"}:
        raise ValueError("invalid checkpoint_selection")
    if config.freeze_backbone and (
        config.checkpoint_selection != "final"
        or config.early_stopping_patience is not None
    ):
        raise ValueError(
            "paired frozen pilots require final checkpoints and disabled early stopping"
        )
    if (
        config.train_rotation_degrees < 0
        or not 0 <= config.train_translate_ratio <= 1
        or not 0 <= config.train_blur_probability <= 1
    ):
        raise ValueError("invalid training augmentation settings")
    if not 0 <= config.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if config.logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    if not 0 <= config.angular_margin < torch.pi / 2:
        raise ValueError("angular_margin must be in [0, pi/2)")
    if not 0 <= config.margin_warmup_ratio <= 1:
        raise ValueError("margin_warmup_ratio must be in [0, 1]")


def _build_training_model(config: TrainingConfig, num_classes: int) -> torch.nn.Module:
    """Build the requested family without leaking proposed-only knobs to baselines."""
    if config.model != "efficient_hccr":
        return build_model(
            config.model,
            num_classes=num_classes,
            in_channels=1,
            classification_head=config.classification_head,
            embedding_dim=config.embedding_dim,
            logit_scale=config.logit_scale,
            angular_margin=config.angular_margin,
            **classifier_options(config),
        )
    return build_model(
        config.model,
        num_classes=num_classes,
        width=config.width,
        backbone_output_channels=config.backbone_output_channels,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        stage_depths=config.stage_depths,
        stem_stride=config.stem_stride,
        reparameterize_depthwise=config.reparameterize_depthwise,
        classification_head=config.classification_head,
        logit_scale=config.logit_scale,
        angular_margin=config.angular_margin,
        **classifier_options(config),
    )


def _model_metadata(
    model: torch.nn.Module, config: TrainingConfig, num_classes: int
) -> dict[str, object]:
    """Persist exactly the constructor data needed to reconstruct a checkpoint."""
    metadata: dict[str, object] = {
        "name": config.model,
        "in_channels": 1,
        "effective_input_channels": getattr(model, "effective_input_channels", 1),
        "classification_head": getattr(model, "classification_head", "softmax"),
        "num_classes": num_classes,
        **classifier_options(config),
    }
    if config.model != "efficient_hccr":
        metadata.update(
            {
                "backbone_output_channels": getattr(
                    model, "backbone_output_channels", None
                ),
                "embedding_dim": config.embedding_dim,
                "resolved_embedding_dim": getattr(model, "embedding_dim", None),
                "logit_scale": config.logit_scale,
                "angular_margin": config.angular_margin,
            }
        )
        return metadata
    metadata.update(
        {
            "width": config.width,
            "backbone_output_channels": model.backbone_output_channels,
            "embedding_dim": model.embedding_dim,
            "stage_depths": list(config.stage_depths),
            "stem_stride": config.stem_stride,
            "reparameterize_depthwise": config.reparameterize_depthwise,
            "dropout": config.dropout,
            "logit_scale": config.logit_scale,
            "angular_margin": config.angular_margin,
        }
    )
    return metadata


def _configure_reproducibility(
    config: TrainingConfig, rank: int = 0
) -> dict[str, object]:
    strict = config.reproducibility_mode == "strict"
    if strict:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    resolved_seed = config.seed + rank
    random_seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)
    torch.use_deterministic_algorithms(strict)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = strict
        torch.backends.cudnn.benchmark = False
    return {
        "mode": config.reproducibility_mode,
        "seed": config.seed,
        "rank_seed": resolved_seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "scope": (
            "recorded environment; cross-version and cross-device identity "
            "not guaranteed"
        ),
    }


def _build_batch_training_plan(
    config: TrainingConfig,
    train_samples: int,
    steps_per_epoch: int,
    world_size: int = 1,
) -> BatchTrainingPlan:
    if train_samples < 1 or steps_per_epoch < 1 or world_size < 1:
        raise ValueError("training split must contain at least one batch")
    effective_global_batch_size = config.batch_size * world_size
    reference_steps_per_epoch = math.ceil(train_samples / config.reference_batch_size)
    configured_steps = config.epochs * steps_per_epoch
    reference_steps = config.epochs * reference_steps_per_epoch
    total_optimizer_steps = (
        max(configured_steps, reference_steps)
        if config.optimizer_step_policy == "reference_batch"
        else configured_steps
    )
    scale = _learning_rate_scale(
        effective_global_batch_size,
        config.reference_batch_size,
        config.learning_rate_scaling,
    )
    return BatchTrainingPlan(
        configured_epochs=config.epochs,
        resolved_epochs=math.ceil(total_optimizer_steps / steps_per_epoch),
        batch_size=config.batch_size,
        steps_per_epoch=steps_per_epoch,
        reference_steps_per_epoch=reference_steps_per_epoch,
        total_optimizer_steps=total_optimizer_steps,
        reference_batch_size=config.reference_batch_size,
        optimizer_step_policy=config.optimizer_step_policy,
        learning_rate_scaling=config.learning_rate_scaling,
        base_learning_rate=config.learning_rate,
        resolved_learning_rate=config.learning_rate * scale,
        resolved_scheduler_min_lr=config.scheduler_min_lr * scale,
        warmup_steps=math.ceil(total_optimizer_steps * config.lr_warmup_ratio),
        world_size=world_size,
        effective_global_batch_size=effective_global_batch_size,
    )


def _learning_rate_scale(
    batch_size: int, reference_batch_size: int, policy: str
) -> float:
    ratio = batch_size / reference_batch_size
    if policy == "none":
        return 1.0
    if policy == "sqrt":
        return math.sqrt(ratio)
    if policy == "linear":
        return ratio
    raise ValueError("learning_rate_scaling must be one of: none, sqrt, linear")


def _resolved_early_stopping_patience(
    config: TrainingConfig, plan: BatchTrainingPlan
) -> int | None:
    if config.early_stopping_patience is None:
        return None
    return math.ceil(
        config.early_stopping_patience * plan.resolved_epochs / config.epochs
    )


def _format_epoch_metrics(
    epoch: int,
    plan: BatchTrainingPlan,
    train_metrics: dict[str, float],
    evaluation_metrics: dict[str, float],
    learning_rate: float,
    completed_steps: int,
    best_top1: float,
    evaluation_name: str,
) -> str:
    """Format the metrics most useful while monitoring a training run."""
    prefix = (
        f"epoch={epoch}/{plan.resolved_epochs} "
        f"updates={completed_steps}/{plan.total_optimizer_steps} "
        f"train_loss={train_metrics['train_loss']:.4f} "
        f"lr={learning_rate:.2e} "
    )
    suffix = f"throughput={train_metrics['train_samples_per_second']:.1f}img/s"
    return (
        prefix
        + (
            f"{evaluation_name}_top1={evaluation_metrics['top1']:.2%} "
            f"{evaluation_name}_top5={evaluation_metrics['top5']:.2%} "
            f"macro_recall={evaluation_metrics['macro_recall']:.2%} "
            f"tail_recall={evaluation_metrics['tail_recall']:.2%} "
            f"best_{evaluation_name}_top1={best_top1:.2%} "
        )
        + suffix
    )


def _format_final_metrics(metrics: dict[str, float], evaluation_name: str) -> str:
    """Format final selected-checkpoint metrics for the run log and CLI."""
    parts = [
        f"{evaluation_name}_top1={metrics['top1']:.2%}",
        f"{evaluation_name}_top5={metrics['top5']:.2%}",
        f"macro_recall={metrics['macro_recall']:.2%}",
        f"tail_recall={metrics['tail_recall']:.2%}",
    ]
    if "expected_calibration_error" in metrics:
        parts.append(f"ece={metrics['expected_calibration_error']:.2%}")
    return " | ".join(parts)


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


def _train_sampler(
    train_set, config: TrainingConfig, distributed: DistributedContext
) -> DistributedSampler | None:
    if not distributed.enabled:
        return None
    return DistributedSampler(
        train_set,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=config.seed,
        drop_last=False,
    )


def _margin_warmup_epochs(total_epochs: int, warmup_ratio: float) -> int:
    if warmup_ratio == 0:
        return 0
    return max(1, math.ceil(total_epochs * warmup_ratio))


def _margin_multiplier(
    epoch: int, total_epochs: int, warmup_ratio: float, head: str
) -> float:
    warmup_epochs = _margin_warmup_epochs(total_epochs, warmup_ratio)
    if head not in {"cosface", "arcface"} or warmup_epochs == 0:
        return 1.0
    if warmup_epochs == 1:
        return 1.0
    return min(1.0, (epoch - 1) / (warmup_epochs - 1))


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    plan: BatchTrainingPlan,
):
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        minimum_factor = plan.resolved_scheduler_min_lr / plan.resolved_learning_rate
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_warmup_factor(
                step,
                plan.total_optimizer_steps,
                plan.warmup_steps,
                minimum_factor,
            ),
        )
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=plan.resolved_scheduler_min_lr,
        )
    raise ValueError("scheduler must be one of: none, cosine, plateau")


def _cosine_warmup_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_factor: float,
) -> float:
    """Return a per-optimizer-step linear-warmup cosine decay multiplier."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return minimum_factor + (1.0 - minimum_factor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _save_tensor_augmentation_gallery(
    dataset: HCCRDataset, config: TrainingConfig, device: str, output_dir: Path
) -> None:
    from torchvision.transforms.v2 import functional as vision_functional

    rows = list({row["unicode_label"]: row for row in dataset.rows[:64]}.values())[:8]
    if not rows:
        return
    images = [dataset.load_image(row) for row in rows]
    transform = EvalPreprocessor(config.image_size)
    inputs = torch.stack(
        [
            vision_functional.to_dtype(
                vision_functional.to_image(transform(image)), torch.float32, scale=True
            )
            for image in images
        ]
    ).to(device)
    augmenter = TensorBatchAugmenter(
        config.image_size,
        rotation_degrees=config.train_rotation_degrees,
        translate_ratio=config.train_translate_ratio,
        blur_probability=config.train_blur_probability,
        seed=config.seed,
    )
    outputs, _ = augmenter(inputs)
    processed = iter(vision_functional.to_pil_image(tensor.cpu()) for tensor in outputs)
    save_gallery(
        images,
        lambda _image: next(processed),
        output_dir / "augmentation_gallery.png",
        [_unicode_codepoint(row["unicode_label"]) for row in rows],
    )


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
        images.append(dataset.load_image(row))
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
    precision: PrecisionContext,
    training_plan: BatchTrainingPlan,
    selection_split: str,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
    resource_profile: dict,
    curves: list[dict],
    icdar_metrics: dict[str, float] | None = None,
) -> None:
    row = {
        **asdict(config),
        **asdict(training_plan),
        "run_id": run_id,
        "selection_split": selection_split,
        "test_split": "test" if test_metrics is not None else None,
        "icdar_split": "icdar" if icdar_metrics is not None else None,
        "precision_requested": precision.requested,
        "precision_resolved": precision.resolved,
        "stage_depths": ",".join(map(str, config.stage_depths)),
        "max_classes": config.max_classes or config.num_classes,
        "peak_training_cuda_memory_mib": max(
            (epoch.get("peak_cuda_memory_mib", 0.0) for epoch in curves), default=0.0
        ),
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **(test_metrics or {}),
        **{f"icdar_{key}": value for key, value in (icdar_metrics or {}).items()},
    }
    if config.model != "efficient_hccr":
        for key in (
            "width",
            "dropout",
            "stage_depths",
            "stem_stride",
            "reparameterize_depthwise",
        ):
            row[key] = None
    if config.classification_head not in {"cosface", "arcface"}:
        row["margin_warmup_ratio"] = 0.0
    SummaryTable(experiments_dir).append(row, resource_profile)
