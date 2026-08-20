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
    model: nn.Module,
    image_size: int,
    device: str,
    output_dir: Path,
    warmup_iterations: int = 20,
    benchmark_iterations: int = 200,
    benchmark_repetitions: int = 5,
) -> dict[str, Any]:
    """Measure complexity and robust batch-1/8/32 forward-pass latency."""
    if min(warmup_iterations, benchmark_iterations, benchmark_repetitions) < 1:
        raise ValueError(
            "benchmark warm-up, iterations and repetitions must be positive"
        )
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
        "benchmark_protocol": {
            "warmup_iterations": warmup_iterations,
            "timed_iterations": benchmark_iterations,
            "repetitions": benchmark_repetitions,
            "aggregation": "median_of_repetition_summaries",
        },
        "device_metadata": _device_metadata(device),
        "inference_benchmarks": [
            _benchmark_batch(
                model,
                image_size,
                device,
                batch_size,
                warmup_iterations,
                benchmark_iterations,
                benchmark_repetitions,
            )
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
    model: nn.Module,
    image_size: int,
    device: str,
    batch_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    repetitions: int,
) -> dict[str, Any]:
    sample = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    repeat_summaries = []
    with torch.inference_mode():
        for _ in range(repetitions):
            for _ in range(warmup_iterations):
                model(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings = []
            for _ in range(benchmark_iterations):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started_at = time.perf_counter()
                model(sample)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - started_at) * 1000)
            repeat_summaries.append(_timing_summary(timings, batch_size))
    return {
        "batch_size": batch_size,
        "latency_mean_ms": statistics.median(
            summary["latency_mean_ms"] for summary in repeat_summaries
        ),
        "latency_p50_ms": statistics.median(
            summary["latency_p50_ms"] for summary in repeat_summaries
        ),
        "latency_p95_ms": statistics.median(
            summary["latency_p95_ms"] for summary in repeat_summaries
        ),
        "latency_p99_ms": statistics.median(
            summary["latency_p99_ms"] for summary in repeat_summaries
        ),
        "samples_per_second": statistics.median(
            summary["samples_per_second"] for summary in repeat_summaries
        ),
        "repeat_summaries": repeat_summaries,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
    }


def _timing_summary(timings: list[float], batch_size: int) -> dict[str, float]:
    mean_latency = statistics.mean(timings)
    return {
        "latency_mean_ms": mean_latency,
        "latency_p50_ms": _percentile(timings, 50),
        "latency_p95_ms": _percentile(timings, 95),
        "latency_p99_ms": _percentile(timings, 99),
        "samples_per_second": batch_size * 1000 / mean_latency,
    }


def _device_metadata(device: str) -> dict[str, Any]:
    metadata = {
        "device": device,
        "dtype": "float32",
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }
    if device.startswith("cuda"):
        metadata["device_name"] = torch.cuda.get_device_name(torch.device(device))
        metadata["cuda_version"] = torch.version.cuda
    return metadata


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile / 100)]


def summarize_batch_norm_state(model: nn.Module) -> dict[str, Any]:
    """Return compact health metrics for every tracked BatchNorm layer."""
    layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        running_mean = module.running_mean
        running_var = module.running_var
        if running_mean is None or running_var is None:
            continue
        finite = bool(
            torch.isfinite(running_mean).all() and torch.isfinite(running_var).all()
        )
        layers[name] = {
            "running_mean_abs_mean": running_mean.detach().abs().mean().item(),
            "running_mean_std": (
                running_mean.detach().float().std(unbiased=False).item()
            ),
            "running_var_mean": running_var.detach().mean().item(),
            "running_var_min": running_var.detach().min().item(),
            "running_var_max": running_var.detach().max().item(),
            "num_batches_tracked": int(module.num_batches_tracked.item()),
            "finite": finite,
        }
    return {
        "layer_count": len(layers),
        "non_finite_layer_count": sum(not layer["finite"] for layer in layers.values()),
        "minimum_running_variance": min(
            (layer["running_var_min"] for layer in layers.values()), default=None
        ),
        "maximum_running_variance": max(
            (layer["running_var_max"] for layer in layers.values()), default=None
        ),
        "layers": layers,
    }


@torch.inference_mode()
def recalibrate_batch_norm(
    model: nn.Module, loader, device: str, max_batches: int
) -> dict[str, Any]:
    """Re-estimate BatchNorm statistics while leaving other modules in eval mode."""
    if max_batches < 1:
        raise ValueError("max_batches must be positive")
    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_layers:
        return {"batches": 0, "samples": 0, **summarize_batch_norm_state(model)}
    original_modes = {module: module.training for module in model.modules()}
    original_momentums = {module: module.momentum for module in batch_norm_layers}
    model.eval()
    for module in batch_norm_layers:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    batches = 0
    samples = 0
    try:
        for images, *_ in loader:
            model(images.to(device))
            batches += 1
            samples += len(images)
            if batches >= max_batches:
                break
    finally:
        for module, momentum in original_momentums.items():
            module.momentum = momentum
        for module, training in original_modes.items():
            module.train(training)
    return {"batches": batches, "samples": samples, **summarize_batch_norm_state(model)}


def summarize_validation_stability(
    epochs: list[dict[str, Any]], drop_threshold: float = 0.05
) -> dict[str, float | int]:
    """Summarize abrupt drops and drawdown from the best prior validation score."""
    if drop_threshold < 0:
        raise ValueError("drop_threshold must be non-negative")
    if not epochs:
        raise ValueError("validation stability requires at least one epoch")
    scores = [float(epoch["top1"]) for epoch in epochs]
    largest_single_drop = 0.0
    max_drawdown = 0.0
    unstable_epochs = 0
    best_so_far = scores[0]
    for previous, current in zip(scores, scores[1:], strict=False):
        single_drop = max(0.0, previous - current)
        largest_single_drop = max(largest_single_drop, single_drop)
        drawdown = max(0.0, best_so_far - current)
        max_drawdown = max(max_drawdown, drawdown)
        if single_drop >= drop_threshold:
            unstable_epochs += 1
        best_so_far = max(best_so_far, current)
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "drop_threshold": drop_threshold,
        "largest_single_epoch_drop": largest_single_drop,
        "max_drawdown_from_prior_best": max_drawdown,
        "unstable_epoch_count": unstable_epochs,
        "best_top1": scores[best_index],
        "best_epoch": int(epochs[best_index]["epoch"]),
        "final_top1": scores[-1],
    }


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
