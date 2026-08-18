"""Minimal train loop; orchestration remains in the CLI layer."""

from __future__ import annotations

import logging
import math
import time

import torch
from torch import nn
from tqdm.auto import tqdm

from hccr.training.diagnostics import ArchitectureDiagnostics


def train_epoch(
    model: nn.Module, loader, optimizer: torch.optim.Optimizer, device: str
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    loss_values: list[float] = []
    gradient_norms: list[float] = []
    started_at = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    data_loading_seconds = 0.0
    forward_backward_seconds = 0.0
    previous_batch_finished = started_at
    with ArchitectureDiagnostics(model) as architecture_diagnostics:
        progress = tqdm(loader, desc="train", unit="batch", leave=False)
        for batch_index, (images, targets, _) in enumerate(progress, start=1):
            batch_started_at = time.perf_counter()
            data_loading_seconds += batch_started_at - previous_batch_finished
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            architecture_diagnostics.record_gradients()
            squared_norm = sum(
                parameter.grad.detach().pow(2).sum().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient_norms.append(math.sqrt(squared_norm))
            optimizer.step()
            forward_backward_seconds += time.perf_counter() - batch_started_at
            total_loss += loss.item() * targets.numel()
            total_samples += targets.numel()
            loss_values.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if batch_index % 100 == 0:
                logging.getLogger("hccr.train").info(
                    "batch=%s loss=%.6f", batch_index, loss.item()
                )
            previous_batch_finished = time.perf_counter()
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "train_loss": total_loss / total_samples,
        "batch_loss_mean": sum(loss_values) / len(loss_values),
        "batch_loss_std": _standard_deviation(loss_values),
        "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_elapsed_seconds": elapsed_seconds,
        "train_samples_per_second": total_samples / elapsed_seconds,
        "data_loading_seconds": data_loading_seconds,
        "forward_backward_seconds": forward_backward_seconds,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else 0.0
        ),
        "stages": architecture_diagnostics.summary(),
    }


def _standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
