"""Resource and optimization diagnostics collected during training."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hccr.utils.experiment import write_json


def profile_model(
    model: nn.Module, image_size: int, device: str, output_dir: Path
) -> dict[str, Any]:
    """Measure model complexity and batch-1/8/32 forward-pass latency."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    macs = estimate_macs(model, image_size, device)
    profile = {
        "device": device,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameters,
        "parameter_size_mib": model_bytes / (1024**2),
        "estimated_macs": macs,
        "estimated_flops": macs * 2,
        "inference_benchmarks": [
            _benchmark_batch(model, image_size, device, batch_size)
            for batch_size in (1, 8, 32)
        ],
    }
    write_json(output_dir / "resource_profile.json", profile)
    return profile


def estimate_macs(model: nn.Module, image_size: int, device: str) -> int:
    """Estimate Conv2d/Linear MACs with forward hooks; unsupported ops are omitted."""
    total_macs = 0

    def hook(
        module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            output_height, output_width = output.shape[-2:]
            kernel_height, kernel_width = module.kernel_size
            total_macs += (
                output.shape[0]
                * output.shape[1]
                * output_height
                * output_width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
        elif isinstance(module, nn.Linear):
            total_macs += output.numel() * module.in_features

    hooks = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        with torch.inference_mode():
            model.eval()(torch.zeros(1, 1, image_size, image_size, device=device))
    finally:
        for handle in hooks:
            handle.remove()
    return total_macs


def _benchmark_batch(
    model: nn.Module, image_size: int, device: str, batch_size: int
) -> dict[str, float | int | None]:
    sample = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.inference_mode():
        for _ in range(3):
            model(sample)
        timings = []
        for _ in range(20):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started_at = time.perf_counter()
            model(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started_at) * 1000)
    return {
        "batch_size": batch_size,
        "latency_mean_ms": statistics.mean(timings),
        "latency_p50_ms": _percentile(timings, 50),
        "latency_p95_ms": _percentile(timings, 95),
        "latency_p99_ms": _percentile(timings, 99),
        "samples_per_second": batch_size * 1000 / statistics.mean(timings),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
    }


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile / 100)]


class ArchitectureDiagnostics(AbstractContextManager["ArchitectureDiagnostics"]):
    """Collect activation and per-stage gradient norms without changing the model."""

    def __init__(self, model: nn.Module) -> None:
        self.activation_values: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.gradient_values: dict[str, list[float]] = defaultdict(list)
        self._handles = [
            module.register_forward_hook(self._activation_hook(name))
            for name, module in model.named_children()
        ]
        self._model = model

    def _activation_hook(self, name: str) -> Callable[..., None]:
        def capture(
            _module: nn.Module, _inputs: tuple[torch.Tensor], output: Any
        ) -> None:
            if not isinstance(output, torch.Tensor):
                return
            detached = output.detach()
            self.activation_values[name].append(
                {
                    "mean": detached.mean().item(),
                    "std": detached.std().item(),
                    "zero_fraction": (detached == 0).float().mean().item(),
                    "non_finite_fraction": (~torch.isfinite(detached))
                    .float()
                    .mean()
                    .item(),
                }
            )

        return capture

    def record_gradients(self) -> None:
        for name, module in self._model.named_children():
            squared_norm = sum(
                parameter.grad.detach().pow(2).sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            if squared_norm:
                self.gradient_values[name].append(squared_norm**0.5)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "activation_mean": statistics.mean(item["mean"] for item in values),
                "activation_std": statistics.mean(item["std"] for item in values),
                "zero_fraction": statistics.mean(
                    item["zero_fraction"] for item in values
                ),
                "non_finite_fraction": statistics.mean(
                    item["non_finite_fraction"] for item in values
                ),
                "gradient_norm_mean": statistics.mean(
                    self.gradient_values.get(name, [0.0])
                ),
                "gradient_norm_max": max(self.gradient_values.get(name, [0.0])),
            }
            for name, values in self.activation_values.items()
        }

    def __exit__(self, *_args: object) -> None:
        for handle in self._handles:
            handle.remove()


def write_training_diagnostics(output_dir: Path, epochs: list[dict[str, Any]]) -> None:
    """Persist a machine-readable history for loss, gradients and throughput."""
    write_json(output_dir / "training_diagnostics.json", {"epochs": epochs})
