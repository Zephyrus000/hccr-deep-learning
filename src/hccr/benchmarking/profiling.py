"""Model complexity and batch-one inference measurements."""

from __future__ import annotations

import io
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hccr.artifacts import write_json
from hccr.benchmarking.complexity import (
    classifier_inference_flops,
    estimate_flops_by_component,
    flop_counting_conventions,
)


@dataclass(frozen=True)
class ModelProfiler:
    """A repeatable measurement protocol, shared by training and sweep runners."""

    warmup_iterations: int = 20
    benchmark_iterations: int = 200
    benchmark_repetitions: int = 5

    def profile_model(
        self,
        model: nn.Module,
        image_size: int,
        device: str,
        output_dir: Path,
        full_class_num_classes: int = 7186,
    ) -> dict[str, Any]:
        """Profile a model without leaking eval mode into its caller."""
        original_modes = {module: module.training for module in model.modules()}
        try:
            return _profile_model(
                model,
                image_size,
                device,
                output_dir,
                self.warmup_iterations,
                self.benchmark_iterations,
                self.benchmark_repetitions,
                full_class_num_classes,
            )
        finally:
            # Restore parents before children so deliberately mixed submodule modes
            # are preserved.  Calling train(), rather than assigning .training,
            # also lets stateful modules clear or rebuild mode-specific caches.
            for module, was_training in original_modes.items():
                module.train(was_training)


def profile_model(
    model: nn.Module,
    image_size: int,
    device: str,
    output_dir: Path,
    warmup_iterations: int = 20,
    benchmark_iterations: int = 200,
    benchmark_repetitions: int = 5,
    full_class_num_classes: int = 7186,
) -> dict[str, Any]:
    return ModelProfiler(
        warmup_iterations, benchmark_iterations, benchmark_repetitions
    ).profile_model(
        model,
        image_size,
        device,
        output_dir,
        full_class_num_classes,
    )


def _profile_model(
    model: nn.Module,
    image_size: int,
    device: str,
    output_dir: Path,
    warmup_iterations: int,
    benchmark_iterations: int,
    benchmark_repetitions: int,
    full_class_num_classes: int,
) -> dict[str, Any]:
    """Measure complexity and repeated batch-one forward-pass latency."""
    if min(warmup_iterations, benchmark_iterations, benchmark_repetitions) < 1:
        raise ValueError(
            "benchmark warm-up, iterations and repetitions must be positive"
        )
    parameter_counts = _parameter_counts_by_component(model)
    parameter_count = parameter_counts["total"]
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = _parameter_bytes_by_component(model)
    model_bytes = parameter_bytes["total"]
    macs = estimate_macs_by_component(model, image_size, device)
    flops, flop_coverage = estimate_flops_by_component(model, image_size, device)
    full_class_projection = project_classifier_cost(
        model,
        parameter_counts,
        parameter_bytes,
        macs,
        full_class_num_classes,
        flops,
    )
    eager_benchmarks = [
        _benchmark_batch(
            model,
            image_size,
            device,
            batch_size,
            warmup_iterations,
            benchmark_iterations,
            benchmark_repetitions,
        )
        for batch_size in (1,)
    ]
    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    classifier = getattr(model, "classifier", None)
    classifier_cache = getattr(classifier, "_deployment_cache", None)
    cache_bytes = (
        sum(t.numel() * t.element_size() for t in classifier_cache[1])
        if classifier_cache is not None
        else 0
    )
    normalized_weight_cache = getattr(classifier, "_normalized_weight", None)
    if normalized_weight_cache is not None:
        cache_bytes += (
            normalized_weight_cache.numel() * normalized_weight_cache.element_size()
        )
    profile = {
        "schema_version": 2,
        "model_name": getattr(model, "name", type(model).__name__),
        "classification_head": getattr(model, "classification_head", "softmax"),
        "classifier_structure": getattr(model, "classifier_structure", "dense"),
        "serialized_state_dict_bytes": serialized.tell(),
        "persistent_buffer_bytes": sum(
            t.numel() * t.element_size() for t in model.buffers()
        ),
        "classifier_deployment_cache_bytes": cache_bytes,
        "device": device,
        "effective_input_channels": getattr(model, "effective_input_channels", 1),
        "backbone_output_channels": getattr(model, "backbone_output_channels", None),
        "embedding_dim": getattr(model, "embedding_dim", None),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameters,
        "parameter_size_mib": model_bytes / (1024**2),
        "backbone_parameter_count": parameter_counts["backbone"],
        "embedding_projection_parameter_count": parameter_counts[
            "embedding_projection"
        ],
        "classifier_parameter_count": parameter_counts["classifier"],
        "head_parameter_count": parameter_counts["head"],
        "backbone_parameter_size_mib": parameter_bytes["backbone"] / (1024**2),
        "embedding_projection_parameter_size_mib": parameter_bytes[
            "embedding_projection"
        ]
        / (1024**2),
        "classifier_parameter_size_mib": parameter_bytes["classifier"] / (1024**2),
        "head_parameter_size_mib": parameter_bytes["head"] / (1024**2),
        "estimated_macs": macs["total"],
        "estimated_backbone_macs": macs["backbone"],
        "estimated_embedding_projection_macs": macs["embedding_projection"],
        "estimated_classifier_macs": macs["classifier"],
        "estimated_head_macs": macs["head"],
        "estimated_input_adapter_macs": macs["input_adapter"],
        "estimated_flops": flops["total"],
        "estimated_backbone_flops": flops["backbone"],
        "estimated_embedding_projection_flops": flops["embedding_projection"],
        "estimated_classifier_flops": flops["classifier"],
        "estimated_head_flops": flops["head"],
        "estimated_input_adapter_flops": flops["input_adapter"],
        "mac_coverage": _mac_coverage(model),
        "flop_coverage": flop_coverage,
        "complexity_protocol": flop_counting_conventions(
            image_size, getattr(model, "effective_input_channels", 1)
        ),
        "full_class_projection": full_class_projection,
        "benchmark_protocol": {
            "warmup_iterations": warmup_iterations,
            "timed_iterations": benchmark_iterations,
            "repetitions": benchmark_repetitions,
            "aggregation": "median_of_repetition_summaries",
            "execution_context": "torch.inference_mode",
            "model_mode": "eval",
            "mode": "eager",
            "scope": "model_forward",
            "cuda_memory_scope": "peak_live_process_allocation",
        },
        "device_metadata": _device_metadata(device),
        "inference_benchmarks": eager_benchmarks,
    }
    write_json(output_dir / "resource_profile.json", profile)
    return profile


def _compare_inference_outputs(
    model: nn.Module,
    optimized_model: nn.Module,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    ramp = torch.linspace(
        0,
        1,
        steps=image_size * image_size,
        device=device,
    ).reshape(1, 1, image_size, image_size)
    rows = torch.arange(image_size, device=device).reshape(-1, 1)
    columns = torch.arange(image_size, device=device).reshape(1, -1)
    checkerboard = ((rows + columns) % 2).reshape(1, 1, image_size, image_size).float()
    samples = torch.cat(
        (
            torch.zeros_like(ramp),
            torch.ones_like(ramp),
            ramp,
            1 - ramp,
            checkerboard,
        )
    )
    model.eval()
    optimized_model.eval()
    with torch.inference_mode():
        expected = model(samples)
        actual = optimized_model(samples)
    absolute_error = torch.abs(expected - actual)
    classifier = getattr(model, "classifier", None)
    logit_scale = max(abs(float(getattr(classifier, "scale", 1.0))), 1.0)
    normalized_expected = expected / logit_scale
    normalized_actual = actual / logit_scale
    normalized_atol = 1e-3
    rtol = 1e-4
    strict_atol = 1e-5
    topk = min(5, expected.shape[1])
    top1_agreement = expected.argmax(dim=1).eq(actual.argmax(dim=1)).float().mean()
    expected_topk = expected.topk(topk, dim=1).indices.sort(dim=1).values
    actual_topk = actual.topk(topk, dim=1).indices.sort(dim=1).values
    topk_agreement = expected_topk.eq(actual_topk).all(dim=1).float().mean()
    normalized_allclose = torch.allclose(
        normalized_expected,
        normalized_actual,
        rtol=rtol,
        atol=normalized_atol,
    )
    return {
        "sample_count": len(samples),
        "maximum_absolute_logit_error": float(absolute_error.max().item()),
        "mean_absolute_logit_error": float(absolute_error.mean().item()),
        "logit_scale": logit_scale,
        "maximum_absolute_normalized_logit_error": float(
            (absolute_error / logit_scale).max().item()
        ),
        "allclose_rtol": rtol,
        "strict_logit_atol": strict_atol,
        "strict_logit_allclose_passed": bool(
            torch.allclose(expected, actual, rtol=rtol, atol=strict_atol)
        ),
        "normalized_logit_atol": normalized_atol,
        "normalized_logit_allclose_passed": bool(normalized_allclose),
        "top1_agreement": float(top1_agreement.item()),
        "top5_agreement": float(topk_agreement.item()),
        "passed": bool(
            normalized_allclose
            and top1_agreement.item() == 1.0
            and topk_agreement.item() == 1.0
        ),
    }


def project_classifier_cost(
    model: nn.Module,
    parameter_counts: dict[str, int],
    parameter_bytes: dict[str, int],
    macs: dict[str, int],
    num_classes: int,
    flops: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Project parameter storage, MACs, and FLOPs for another class count."""
    if num_classes < 2:
        raise ValueError("full-class projection requires at least two classes")
    classifier = getattr(model, "classifier", None)
    if callable(getattr(classifier, "inference_macs", None)):
        return {
            "status": "unavailable",
            "reason": (
                "factorized class allocation cannot be projected to another "
                "vocabulary without a new mask; use measured model costs"
            ),
            "num_classes": num_classes,
        }
    if isinstance(classifier, nn.Sequential):
        linear = next(
            (
                module
                for module in reversed(classifier)
                if isinstance(module, nn.Linear)
            ),
            None,
        )
    else:
        linear = classifier if isinstance(classifier, nn.Linear) else None
    if linear is None:
        return {
            "status": "unavailable",
            "reason": "model does not expose a Linear classifier boundary",
            "num_classes": num_classes,
        }
    original_classifier_parameters = linear.weight.numel()
    projected_classifier_parameters = linear.in_features * num_classes
    if linear.bias is not None:
        original_classifier_parameters += linear.bias.numel()
        projected_classifier_parameters += num_classes
    element_size = linear.weight.element_size()
    original_classifier_bytes = original_classifier_parameters * element_size
    projected_classifier_bytes = projected_classifier_parameters * element_size
    projected_classifier_macs = linear.in_features * num_classes
    original_classifier_macs = linear.in_features * linear.out_features
    fixed_head_parameters = parameter_counts["head"] - original_classifier_parameters
    fixed_head_bytes = parameter_bytes["head"] - original_classifier_bytes
    fixed_head_macs = macs["head"] - original_classifier_macs
    projected_head_parameters = fixed_head_parameters + projected_classifier_parameters
    projected_head_bytes = fixed_head_bytes + projected_classifier_bytes
    projected_head_macs = fixed_head_macs + projected_classifier_macs
    projection = {
        "status": "available",
        "num_classes": num_classes,
        "embedding_dim": linear.in_features,
        "backbone_parameter_count": parameter_counts["backbone"],
        "embedding_projection_parameter_count": fixed_head_parameters,
        "classifier_parameter_count": projected_classifier_parameters,
        "head_parameter_count": projected_head_parameters,
        "head_parameter_size_mib": projected_head_bytes / (1024**2),
        "backbone_macs": macs["backbone"],
        "embedding_projection_macs": fixed_head_macs,
        "classifier_macs": projected_classifier_macs,
        "head_macs": projected_head_macs,
        "total_parameter_count": parameter_counts["backbone"]
        + projected_head_parameters,
        "total_parameter_size_mib": (parameter_bytes["backbone"] + projected_head_bytes)
        / (1024**2),
        "total_macs": macs["backbone"] + projected_head_macs,
    }
    if flops is not None:
        original_classifier_flops = classifier_inference_flops(
            linear, linear.out_features
        )
        projected_classifier_flops = classifier_inference_flops(linear, num_classes)
        fixed_head_flops = flops["head"] - original_classifier_flops
        projected_head_flops = fixed_head_flops + projected_classifier_flops
        projection.update(
            {
                "backbone_flops": flops["backbone"],
                "embedding_projection_flops": fixed_head_flops,
                "classifier_flops": projected_classifier_flops,
                "head_flops": projected_head_flops,
                "total_flops": flops["backbone"] + projected_head_flops,
            }
        )
    return projection


def _mac_coverage(model: nn.Module) -> dict[str, Any]:
    counted_types = set()
    unsupported_types = set()
    for module in model.modules():
        if len(tuple(module.children())) > 0:
            continue
        if (
            isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear))
            or callable(getattr(module, "fixed_filter_macs", None))
            or callable(getattr(module, "inference_macs", None))
        ):
            counted_types.add(type(module).__name__)
        else:
            unsupported_types.add(type(module).__name__)
    return {
        "estimator": "forward_hooks_conv_linear_and_fixed_filters",
        "complete": not unsupported_types,
        "counted_operator_types": sorted(counted_types),
        "unsupported_operator_types": sorted(unsupported_types),
        "note": (
            "Unsupported operators are excluded from estimated MACs; measured "
            "latency remains authoritative."
        ),
    }


def estimate_macs(model: nn.Module, image_size: int, device: str) -> int:
    """Estimate Conv2d/Linear MACs with forward hooks; unsupported ops are omitted."""
    return estimate_macs_by_component(model, image_size, device)["total"]


def estimate_macs_by_component(
    model: nn.Module, image_size: int, device: str
) -> dict[str, int]:
    """Estimate total, backbone and classifier Conv2d/Linear MACs."""
    macs = {
        "total": 0,
        "backbone": 0,
        "embedding_projection": 0,
        "classifier": 0,
        "head": 0,
        "input_adapter": 0,
    }
    component_by_module = {
        id(module): _component_for_name(name) for name, module in model.named_modules()
    }
    input_adapter_modules = {
        id(module)
        for name, module in model.named_modules()
        if name == "input_adapter" or name.startswith("input_adapter.")
    }

    def hook(
        module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        operation_macs = 0
        if isinstance(module, nn.Conv2d):
            output_height, output_width = output.shape[-2:]
            kernel_height, kernel_width = module.kernel_size
            operation_macs = (
                output.shape[0]
                * output.shape[1]
                * output_height
                * output_width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
        elif isinstance(module, nn.Conv1d):
            operation_macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
            )
        elif isinstance(module, nn.Linear):
            operation_macs = output.numel() * module.in_features
        elif callable(getattr(module, "inference_macs", None)):
            operation_macs = output.shape[0] * module.inference_macs()
            operation_macs += getattr(module, "last_forward_extra_macs", 0)
        else:
            fixed_filter_macs = getattr(module, "fixed_filter_macs", None)
            if callable(fixed_filter_macs):
                operation_macs = fixed_filter_macs(_inputs[0])
        component = component_by_module[id(module)]
        macs[component] += operation_macs
        if id(module) in input_adapter_modules:
            macs["input_adapter"] += operation_macs
        macs["total"] += operation_macs

    hooks = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear))
        or callable(getattr(module, "fixed_filter_macs", None))
        or callable(getattr(module, "inference_macs", None))
    ]
    try:
        with torch.inference_mode():
            model.eval()(torch.zeros(1, 1, image_size, image_size, device=device))
    finally:
        for handle in hooks:
            handle.remove()
    macs["head"] = macs["embedding_projection"] + macs["classifier"]
    return macs


def _parameter_counts_by_component(model: nn.Module) -> dict[str, int]:
    counts = {
        "total": 0,
        "backbone": 0,
        "embedding_projection": 0,
        "classifier": 0,
        "head": 0,
    }
    for name, parameter in model.named_parameters():
        component = _component_for_name(name)
        count = parameter.numel()
        counts[component] += count
        counts["total"] += count
    counts["head"] = counts["embedding_projection"] + counts["classifier"]
    return counts


def _parameter_bytes_by_component(model: nn.Module) -> dict[str, int]:
    sizes = {
        "total": 0,
        "backbone": 0,
        "embedding_projection": 0,
        "classifier": 0,
        "head": 0,
    }
    for name, parameter in model.named_parameters():
        component = _component_for_name(name)
        size = parameter.numel() * parameter.element_size()
        sizes[component] += size
        sizes["total"] += size
    sizes["head"] = sizes["embedding_projection"] + sizes["classifier"]
    return sizes


def _component_for_name(name: str) -> str:
    if name == "classifier" or name.startswith("classifier."):
        return "classifier"
    if name == "embedding_projection" or name.startswith("embedding_projection."):
        return "embedding_projection"
    return "backbone"


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
        "samples_per_second": batch_size * 1000 / mean_latency,
    }


def _device_metadata(device: str) -> dict[str, Any]:
    if device == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    metadata = {
        "device": device,
        "dtype": "float32",
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": str(torch.__version__),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
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
