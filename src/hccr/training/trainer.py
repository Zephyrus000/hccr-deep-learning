"""Minimal train loop; orchestration remains in the CLI layer."""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from hccr.preprocessing.batch import TensorBatchAugmenter
from hccr.training.diagnostics import ArchitectureDiagnostics
from hccr.training.distributed import DistributedContext, unwrap_model
from hccr.training.gate_diagnostics import GateTrainingDiagnostics
from hccr.training.precision import PrecisionContext


@contextmanager
def _batch_norm_tail_momentum(
    layers, batch_size: int, reference_batch_size: int | None
):
    """Give a partial batch the same running-stat decay per image as full batches."""
    changed = []
    if reference_batch_size and batch_size < reference_batch_size:
        ratio = batch_size / reference_batch_size
        for module, momentum in layers:
            if module.training:
                changed.append((module, momentum))
                module.momentum = 1 - (1 - momentum) ** ratio
    try:
        yield
    finally:
        for module, momentum in changed:
            module.momentum = momentum


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
    margin_multiplier: float = 1.0,
    batch_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_batches: int | None = None,
    distributed: DistributedContext | None = None,
    precision: PrecisionContext | None = None,
    *,
    residual_regularization: float = 0.0,
    gate_regularization: float = 0.0,
    freeze_backbone: bool = False,
    batch_augmenter: TensorBatchAugmenter | None = None,
) -> dict[str, Any]:
    model.train()
    base_model = unwrap_model(model)
    if freeze_backbone:
        base_model.eval()
        base_model.classifier.train()
    batch_norm_layers = [
        (module, module.momentum)
        for module in base_model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.training
        and module.track_running_stats
        and module.momentum is not None
    ]
    reference_batch_size = getattr(loader, "batch_size", None)
    is_main_process = distributed is None or distributed.is_main_process
    total_loss = 0.0
    total_classification_loss = 0.0
    total_regularization_loss = 0.0
    total_samples = 0
    loss_values: list[float] = []
    gradient_norms: list[float] = []
    augmentation_counts: dict[str, int] = {}
    started_at = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    data_loading_seconds = 0.0
    forward_backward_seconds = 0.0
    previous_batch_finished = started_at
    learning_rate_start = optimizer.param_groups[0]["lr"]
    precision = precision or PrecisionContext("float32", "float32", "cpu", None, None)
    optimizer_steps = 0
    gate_diagnostics = GateTrainingDiagnostics(base_model)
    with ArchitectureDiagnostics(base_model) as architecture_diagnostics:
        progress = tqdm(
            loader,
            desc="train",
            unit="batch",
            leave=is_main_process,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not is_main_process,
        )
        for batch_index, (images, targets, metadata) in enumerate(progress, start=1):
            batch_started_at = time.perf_counter()
            data_loading_seconds += batch_started_at - previous_batch_finished
            images, targets = images.to(device), targets.to(device)
            if batch_augmenter is not None:
                images, metadata = batch_augmenter(images.float())
            optimizer.zero_grad(set_to_none=True)
            training_logits = getattr(base_model, "training_logits", None)
            with (
                _batch_norm_tail_momentum(
                    batch_norm_layers, len(images), reference_batch_size
                ),
                precision.autocast(),
            ):
                logits = (
                    model(images, targets, margin_multiplier)
                    if training_logits is not None
                    else model(images)
                )
            classification_loss = criterion(logits.float(), targets)
            regularization_loss = classification_loss.new_zeros(())
            if residual_regularization or gate_regularization:
                regularize = getattr(base_model.classifier, "regularization_loss", None)
                if regularize is None:
                    raise ValueError("regularization requires a factorized classifier")
                regularization_loss = regularize(
                    residual_regularization, gate_regularization
                )
            loss = classification_loss + regularization_loss
            precision.backward(loss)
            precision.unscale_(optimizer)
            architecture_diagnostics.record_gradients()
            squared_norm = sum(
                parameter.grad.detach().pow(2).sum().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient_norms.append(math.sqrt(squared_norm))
            gate_diagnostics.before_step(batch_index)
            optimizer_stepped = precision.step(optimizer)
            gate_diagnostics.after_step(optimizer_stepped)
            if batch_scheduler is not None and optimizer_stepped:
                batch_scheduler.step()
            optimizer_steps += int(optimizer_stepped)
            forward_backward_seconds += time.perf_counter() - batch_started_at
            total_loss += loss.item() * targets.numel()
            total_classification_loss += classification_loss.item() * targets.numel()
            total_regularization_loss += regularization_loss.item() * targets.numel()
            total_samples += targets.numel()
            loss_values.append(loss.item())
            encoded_augmentations = (
                metadata.get("applied_augmentations", ())
                if isinstance(metadata, dict)
                else metadata
            )
            for encoded in encoded_augmentations:
                for augmentation in filter(None, encoded.split(",")):
                    augmentation_counts[augmentation] = (
                        augmentation_counts.get(augmentation, 0) + 1
                    )
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if is_main_process and batch_index % 100 == 0:
                logging.getLogger("hccr.train").info(
                    "batch=%s loss=%.6f", batch_index, loss.item()
                )
            previous_batch_finished = time.perf_counter()
            if max_batches is not None and batch_index >= max_batches:
                break
    elapsed_seconds = time.perf_counter() - started_at
    if total_samples == 0:
        raise ValueError("training loader produced no samples")
    local_metrics = {
        "loss_sum": total_loss,
        "classification_loss_sum": total_classification_loss,
        "regularization_loss_sum": total_regularization_loss,
        "sample_count": total_samples,
        "batch_loss_sum": sum(loss_values),
        "batch_loss_squared_sum": sum(value**2 for value in loss_values),
        "gradient_norm_sum": sum(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "optimizer_steps": optimizer_steps,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "learning_rate_start": learning_rate_start,
        "margin_multiplier": margin_multiplier,
        "augmentation_counts": augmentation_counts,
        "train_elapsed_seconds": elapsed_seconds,
        "data_loading_seconds": data_loading_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else 0.0
        ),
        "stages": architecture_diagnostics.summary(),
    }
    gate_summary = gate_diagnostics.summary()
    if gate_summary is not None:
        local_metrics["gate_training"] = gate_summary
    gathered_metrics = (
        distributed.all_gather_object(local_metrics)
        if distributed is not None
        else [local_metrics]
    )
    return _aggregate_epoch_metrics(gathered_metrics)


def _standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _aggregate_epoch_metrics(rank_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine small end-of-epoch metric summaries from all DDP ranks."""
    total_samples = sum(int(metrics["sample_count"]) for metrics in rank_metrics)
    total_steps = sum(int(metrics["optimizer_steps"]) for metrics in rank_metrics)
    total_loss = sum(float(metrics["loss_sum"]) for metrics in rank_metrics)
    batch_loss_sum = sum(float(metrics["batch_loss_sum"]) for metrics in rank_metrics)
    batch_loss_squared_sum = sum(
        float(metrics["batch_loss_squared_sum"]) for metrics in rank_metrics
    )
    gradient_norm_sum = sum(
        float(metrics["gradient_norm_sum"]) for metrics in rank_metrics
    )
    augmentation_counts: dict[str, int] = {}
    for metrics in rank_metrics:
        for name, count in metrics["augmentation_counts"].items():
            augmentation_counts[name] = augmentation_counts.get(name, 0) + int(count)
    batch_loss_mean = batch_loss_sum / total_steps
    batch_loss_variance = max(
        0.0, batch_loss_squared_sum / total_steps - batch_loss_mean**2
    )
    elapsed_seconds = max(
        float(metrics["train_elapsed_seconds"]) for metrics in rank_metrics
    )
    result = {
        "train_loss": total_loss / total_samples,
        "train_classification_loss": sum(
            float(m.get("classification_loss_sum", m["loss_sum"])) for m in rank_metrics
        )
        / total_samples,
        "train_regularization_loss": sum(
            float(m.get("regularization_loss_sum", 0.0)) for m in rank_metrics
        )
        / total_samples,
        "batch_loss_mean": batch_loss_mean,
        "batch_loss_std": math.sqrt(batch_loss_variance),
        "gradient_norm_mean": gradient_norm_sum / total_steps,
        "gradient_norm_max": max(
            float(metrics["gradient_norm_max"]) for metrics in rank_metrics
        ),
        "learning_rate": float(rank_metrics[0]["learning_rate"]),
        "learning_rate_start": float(rank_metrics[0]["learning_rate_start"]),
        "margin_multiplier": float(rank_metrics[0]["margin_multiplier"]),
        "optimizer_steps": max(
            int(metrics["optimizer_steps"]) for metrics in rank_metrics
        ),
        "augmentation_counts": augmentation_counts,
        "augmentation_rates": {
            name: count / total_samples for name, count in augmentation_counts.items()
        },
        "train_elapsed_seconds": elapsed_seconds,
        "train_samples_per_second": total_samples / elapsed_seconds,
        "data_loading_seconds": max(
            float(metrics["data_loading_seconds"]) for metrics in rank_metrics
        ),
        "forward_backward_seconds": max(
            float(metrics["forward_backward_seconds"]) for metrics in rank_metrics
        ),
        "peak_cuda_memory_mib": max(
            float(metrics["peak_cuda_memory_mib"]) for metrics in rank_metrics
        ),
        "stages": rank_metrics[0]["stages"],
    }
    if any("gate_training" in metrics for metrics in rank_metrics):
        result["gate_training"] = {
            "aggregation": "per_rank_observations_after_gradient_synchronization",
            "ranks": [metrics.get("gate_training") for metrics in rank_metrics],
        }
    return result
