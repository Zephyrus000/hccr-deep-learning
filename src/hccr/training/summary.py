"""Canonical experiment-summary schema and profile migration helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EXPERIMENT_SUMMARY_FIELDS = (
    "run_id",
    "selection_split",
    "test_split",
    "model",
    "precision_requested",
    "precision_resolved",
    "effective_input_channels",
    "width",
    "backbone_output_channels",
    "embedding_dim",
    "dropout",
    "stage_depths",
    "stem_stride",
    "reparameterize_depthwise",
    "classification_head",
    "label_smoothing",
    "logit_scale",
    "angular_margin",
    "margin_warmup_ratio",
    "epochs",
    "resolved_epochs",
    "batch_size",
    "world_size",
    "effective_global_batch_size",
    "optimizer_step_policy",
    "reference_batch_size",
    "total_optimizer_steps",
    "image_size",
    "validation_top1",
    "validation_top5",
    "validation_macro_recall",
    "validation_tail_recall",
    "top1",
    "top5",
    "macro_recall",
    "head_recall",
    "mid_recall",
    "tail_recall",
    "expected_calibration_error",
    "parameter_count",
    "backbone_parameter_count",
    "embedding_projection_parameter_count",
    "classifier_parameter_count",
    "head_parameter_count",
    "estimated_macs",
    "estimated_backbone_macs",
    "estimated_embedding_projection_macs",
    "estimated_classifier_macs",
    "estimated_head_macs",
    "estimated_flops",
    "estimated_backbone_flops",
    "estimated_embedding_projection_flops",
    "estimated_classifier_flops",
    "estimated_head_flops",
    "mac_coverage_complete",
    "unsupported_operator_types",
    "unsupported_mac_operator_types",
    "flop_coverage_complete",
    "unsupported_flop_operator_types",
    "full_class_num_classes",
    "full_class_parameter_count",
    "full_class_estimated_macs",
    "full_class_estimated_flops",
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
    "resolved_learning_rate",
    "learning_rate_scaling",
    "lr_warmup_ratio",
    "weight_decay",
    "scheduler",
    "seed",
    "max_classes",
)


def merge_resource_profile_into_summary(
    source_row: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Update only measurements present in a newly generated profile."""
    row = dict(source_row)
    benchmark = _batch_one(profile.get("inference_benchmarks") or [])
    full_class = profile.get("full_class_projection") or {}
    mac_coverage = profile.get("mac_coverage") or {}
    flop_coverage = profile.get("flop_coverage")
    values = {
        "effective_input_channels": profile.get("effective_input_channels"),
        "backbone_output_channels": profile.get("backbone_output_channels"),
        "embedding_dim": profile.get("embedding_dim"),
        "parameter_count": profile.get("parameter_count"),
        "backbone_parameter_count": profile.get("backbone_parameter_count"),
        "embedding_projection_parameter_count": profile.get(
            "embedding_projection_parameter_count"
        ),
        "classifier_parameter_count": profile.get("classifier_parameter_count"),
        "head_parameter_count": profile.get("head_parameter_count"),
        "estimated_macs": profile.get("estimated_macs"),
        "estimated_backbone_macs": profile.get("estimated_backbone_macs"),
        "estimated_embedding_projection_macs": profile.get(
            "estimated_embedding_projection_macs"
        ),
        "estimated_classifier_macs": profile.get("estimated_classifier_macs"),
        "estimated_head_macs": profile.get("estimated_head_macs"),
        "estimated_flops": profile.get("estimated_flops"),
        "estimated_backbone_flops": profile.get("estimated_backbone_flops"),
        "estimated_embedding_projection_flops": profile.get(
            "estimated_embedding_projection_flops"
        ),
        "estimated_classifier_flops": profile.get(
            "estimated_classifier_flops"
        ),
        "estimated_head_flops": profile.get("estimated_head_flops"),
        "mac_coverage_complete": mac_coverage.get("complete"),
        "unsupported_operator_types": json.dumps(
            mac_coverage.get("unsupported_operator_types", [])
        ),
        "unsupported_mac_operator_types": json.dumps(
            mac_coverage.get("unsupported_operator_types", [])
        ),
        "full_class_num_classes": full_class.get("num_classes"),
        "full_class_parameter_count": full_class.get("total_parameter_count"),
        "full_class_estimated_macs": full_class.get("total_macs"),
        "full_class_estimated_flops": full_class.get("total_flops"),
        "latency_p50_ms": benchmark.get("latency_p50_ms"),
        "latency_p95_ms": benchmark.get("latency_p95_ms"),
        "latency_p99_ms": benchmark.get("latency_p99_ms"),
        "samples_per_second": benchmark.get("samples_per_second"),
        "peak_inference_cuda_memory_mib": benchmark.get(
            "peak_cuda_memory_mib"
        ),
    }
    if flop_coverage is not None:
        values.update(
            {
                "flop_coverage_complete": flop_coverage.get("complete"),
                "unsupported_flop_operator_types": json.dumps(
                    flop_coverage.get("unsupported_operator_types", [])
                ),
            }
        )
    end_to_end = profile.get("end_to_end_batch1_benchmark")
    if end_to_end is not None:
        values.update(
            {
                "end_to_end_latency_p50_ms": end_to_end.get("latency_p50_ms"),
                "end_to_end_latency_p95_ms": end_to_end.get("latency_p95_ms"),
                "end_to_end_latency_p99_ms": end_to_end.get("latency_p99_ms"),
            }
        )
    row.update({key: value for key, value in values.items() if value is not None})
    return row


def write_experiment_summary_rows(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write rows using the canonical CSV column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=EXPERIMENT_SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _batch_one(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        return next(row for row in rows if int(row.get("batch_size", -1)) == 1)
    except StopIteration as error:
        raise ValueError("resource profile has no batch-size-1 benchmark") from error
