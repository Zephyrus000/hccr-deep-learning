"""Operator-level theoretical FLOP accounting for inference profiles."""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from hccr.models.heads import AngularMarginClassifier

_aten = torch.ops.aten


def _shape_numel(shape: Any) -> int:
    if isinstance(shape, torch.Size):
        return prod(shape)
    if isinstance(shape, (tuple, list)) and all(
        isinstance(dimension, int) for dimension in shape
    ):
        return prod(shape)
    raise TypeError(f"expected a tensor shape, got {shape!r}")


def _primary_output_shape(out_shape: Any) -> Any:
    if (
        isinstance(out_shape, (tuple, list))
        and out_shape
        and isinstance(out_shape[0], (torch.Size, tuple, list))
    ):
        return out_shape[0]
    return out_shape


def _output_elements(out_shape: Any) -> int:
    return _shape_numel(_primary_output_shape(out_shape))


def _matrix_multiply_flops(
    left_shape: Any,
    right_shape: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    return 2 * left_shape[-2] * left_shape[-1] * right_shape[-1]


def _add_matrix_multiply_flops(
    _bias_shape: Any,
    left_shape: Any,
    right_shape: Any,
    *args: Any,
    out_shape: Any = None,
    beta: int | float = 1,
    **kwargs: Any,
) -> int:
    del args, kwargs
    matrix_flops = _matrix_multiply_flops(left_shape, right_shape)
    return matrix_flops + (_output_elements(out_shape) if beta != 0 else 0)


def _convolution_flops(
    input_shape: Any,
    weight_shape: Any,
    bias_shape: Any,
    _stride: Any,
    _padding: Any,
    _dilation: Any,
    transposed: bool,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, kwargs
    output_shape = _primary_output_shape(out_shape)
    spatial_shape = input_shape[2:] if transposed else output_shape[2:]
    macs = input_shape[0] * prod(spatial_shape) * prod(weight_shape)
    bias_additions = _shape_numel(output_shape) if bias_shape is not None else 0
    return 2 * macs + bias_additions


def _batch_norm_inference_flops(
    input_shape: Any,
    weight_shape: Any,
    bias_shape: Any,
    _running_mean_shape: Any,
    _running_variance_shape: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    elements = _shape_numel(input_shape)
    channels = input_shape[1]
    # Direct inference expression:
    # (x - mean) * rsqrt(variance + eps) * weight + bias.
    flops = 2 * elements + 2 * channels
    if weight_shape is not None:
        flops += elements
    if bias_shape is not None:
        flops += elements
    return flops


def _elementwise_formula(flops_per_element: int) -> Callable[..., int]:
    def formula(*args: Any, out_shape: Any = None, **kwargs: Any) -> int:
        del args, kwargs
        return flops_per_element * _output_elements(out_shape)

    return formula


def _clamp_flops(
    _input_shape: Any,
    minimum: Any = None,
    maximum: Any = None,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, kwargs
    comparisons = int(minimum is not None) + int(maximum is not None)
    return comparisons * _output_elements(out_shape)


def _vector_norm_flops(
    input_shape: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    # L2 normalization: square, reduce-add and square root. For V vectors
    # containing N total elements this is N + (N - V) + V = 2N FLOPs.
    return 2 * _shape_numel(input_shape)


def _mean_flops(
    input_shape: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    # Each reduction group performs k - 1 additions and one division.
    return _shape_numel(input_shape)


def _max_pool_flops(
    _input_shape: Any,
    kernel_size: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, kwargs
    kernel = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else tuple(kernel_size)
    )
    return _output_elements(out_shape) * max(prod(kernel) - 1, 0)


def _average_pool_flops(
    _input_shape: Any,
    kernel_size: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, kwargs
    kernel = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else tuple(kernel_size)
    )
    # k - 1 additions plus one division for every output element.
    return _output_elements(out_shape) * prod(kernel)


def _adaptive_average_pool_flops(
    input_shape: Any,
    *args: Any,
    out_shape: Any = None,
    **kwargs: Any,
) -> int:
    del args, out_shape, kwargs
    # Exact for the global-average-pooling case used by all retained models.
    return _shape_numel(input_shape)


_CUSTOM_FLOP_MAPPING: dict[Any, Callable[..., int]] = {
    _aten.mm: _matrix_multiply_flops,
    _aten.addmm: _add_matrix_multiply_flops,
    _aten.convolution: _convolution_flops,
    _aten._convolution: _convolution_flops,
    _aten.cudnn_convolution: _convolution_flops,
    _aten.convolution_overrideable: _convolution_flops,
    _aten._slow_conv2d_forward: _convolution_flops,
    _aten._native_batch_norm_legit_no_training: _batch_norm_inference_flops,
    _aten.add: _elementwise_formula(1),
    _aten.add_: _elementwise_formula(1),
    _aten.sub: _elementwise_formula(1),
    _aten.mul: _elementwise_formula(1),
    _aten.div: _elementwise_formula(1),
    _aten.relu: _elementwise_formula(1),
    _aten.relu_: _elementwise_formula(1),
    _aten.sigmoid: _elementwise_formula(4),
    _aten.silu: _elementwise_formula(5),
    _aten.silu_: _elementwise_formula(5),
    _aten.hardsigmoid: _elementwise_formula(4),
    _aten.hardswish: _elementwise_formula(5),
    _aten.hardswish_: _elementwise_formula(5),
    _aten.clamp: _clamp_flops,
    _aten.clamp_min: _elementwise_formula(1),
    _aten.linalg_vector_norm: _vector_norm_flops,
    _aten.mean: _mean_flops,
    _aten.max_pool2d_with_indices: _max_pool_flops,
    _aten.avg_pool2d: _average_pool_flops,
    _aten.adaptive_avg_pool2d: _adaptive_average_pool_flops,
}


_ZERO_ARITHMETIC_FLOP_OPERATORS = {
    _aten.cat,
    _aten.clone,
    _aten.detach,
    _aten.empty,
    _aten.expand,
    _aten.split,
    _aten.t,
    _aten.transpose,
    _aten.view,
    _aten.zeros,
}


class _CoverageFlopCounterMode(FlopCounterMode):
    """Record every dispatched operator in addition to counting FLOPs."""

    def __init__(self) -> None:
        super().__init__(display=False, custom_mapping=_CUSTOM_FLOP_MAPPING)
        self.seen_operators: set[Any] = set()

    def _count_flops(
        self,
        func_packet: Any,
        out: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self.seen_operators.add(func_packet)
        return super()._count_flops(func_packet, out, args, kwargs)


def estimate_flops_by_component(
    model: nn.Module, image_size: int, device: str
) -> tuple[dict[str, int], dict[str, Any]]:
    """Trace an eval forward pass and count theoretical operator FLOPs.

    A multiplication and an addition are each one FLOP, so one convolution or
    matrix multiplication MAC contributes two FLOPs. Elementwise, reduction,
    normalization, activation, and pooling work is counted separately.
    """
    model.eval()
    sample = torch.zeros(1, 1, image_size, image_size, device=device)
    counter = _CoverageFlopCounterMode()
    with torch.inference_mode(), counter:
        model(sample)

    counts = counter.get_flop_counts()
    root_name = type(model).__name__

    def module_flops(path: str) -> int:
        return sum(counts.get(path, {}).values())

    embedding_projection = module_flops(f"{root_name}.embedding_projection")
    classifier = module_flops(f"{root_name}.classifier")
    head = embedding_projection + classifier
    total = counter.get_total_flops()
    flops = {
        "total": total,
        "backbone": total - head,
        "embedding_projection": embedding_projection,
        "classifier": classifier,
        "head": head,
        "input_adapter": module_flops(f"{root_name}.input_adapter"),
    }

    counted = counter.seen_operators & set(counter.flop_registry)
    zero_flop = counter.seen_operators & _ZERO_ARITHMETIC_FLOP_OPERATORS
    unsupported = counter.seen_operators - counted - zero_flop
    global_counts = counts.get("Global", {})
    coverage = {
        "estimator": "torch_dispatch_operator_formulas_v1",
        "complete": not unsupported,
        "counted_operator_types": sorted(map(str, counted)),
        "zero_arithmetic_flop_operator_types": sorted(map(str, zero_flop)),
        "unsupported_operator_types": sorted(map(str, unsupported)),
        "flops_by_operator": {
            str(operator): operation_flops
            for operator, operation_flops in sorted(
                global_counts.items(), key=lambda item: str(item[0])
            )
        },
        "note": (
            "Complete means every operator observed during the traced eval forward "
            "is either counted or classified as zero arithmetic FLOPs. Data movement "
            "and kernel efficiency are represented by measured latency, not FLOPs."
        ),
    }
    return flops, coverage


def classifier_inference_flops(classifier: nn.Linear, num_classes: int) -> int:
    """Return classifier FLOPs under the repository's inference convention."""
    matrix_flops = 2 * classifier.in_features * num_classes
    if isinstance(classifier, AngularMarginClassifier):
        # Input L2 norm (2D), clamp-min (D), division (D), output clamp (2C),
        # and output scaling (C). Eval-mode weight normalization is cached.
        return matrix_flops + 4 * classifier.in_features + 3 * num_classes
    return matrix_flops + (num_classes if classifier.bias is not None else 0)


def flop_counting_conventions(image_size: int, input_channels: int) -> dict[str, Any]:
    """Describe the reproducible counting protocol persisted with each run."""
    return {
        "estimator": "torch_dispatch_operator_formulas_v1",
        "input_shape": [1, input_channels, image_size, image_size],
        "model_mode": "eval",
        "multiply_and_add_counted_separately": True,
        "flops_per_mac": 2,
        "bias_additions_counted": True,
        "batch_norm_inference_formula": (
            "4 FLOPs per tensor element plus 2 FLOPs per channel"
        ),
        "activation_flops_per_element": {
            "relu": 1,
            "sigmoid": 4,
            "silu": 5,
            "hardsigmoid": 4,
            "hardswish": 5,
        },
        "pooling": "reduction additions/divisions or comparisons are counted",
        "cached_angular_weight_normalization": "excluded from per-sample inference",
        "data_movement": (
            "view, transpose, split, concatenate and copy operations contribute "
            "zero arithmetic FLOPs; their cost is captured by latency"
        ),
    }
