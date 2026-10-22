"""Compact paper tables written directly from a run's measured results."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_SUMMARY_FIELDS = (
    "run_id",
    "selection_split",
    "test_split",
    "icdar_split",
    "icdar_top1",
    "icdar_top5",
    "icdar_macro_recall",
    "icdar_head_recall",
    "icdar_mid_recall",
    "icdar_tail_recall",
    "icdar_expected_calibration_error",
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
    "parameter_size_mib",
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
    "unsupported_mac_operator_types",
    "flop_coverage_complete",
    "unsupported_flop_operator_types",
    "full_class_num_classes",
    "full_class_parameter_count",
    "full_class_estimated_macs",
    "full_class_estimated_flops",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
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


@dataclass(frozen=True)
class SummaryTable:
    directory: Path

    def append(self, run: Mapping[str, Any], profile: Mapping[str, Any]) -> Path:
        """Append current measurements, leaving historical CSV schemas untouched."""
        path = self.directory / "experiment_summary.csv"
        if path.exists() and self._fields(path) != EXPERIMENT_SUMMARY_FIELDS:
            path = self.directory / "experiment_summary.v2.csv"
        if path.exists() and self._fields(path) != EXPERIMENT_SUMMARY_FIELDS:
            raise ValueError(f"incompatible summary schema: {path}")
        benchmark = next(
            item for item in profile["inference_benchmarks"] if item["batch_size"] == 1
        )
        full_class = profile.get("full_class_projection", {})
        row = {
            **run,
            **{
                key: value
                for key, value in profile.items()
                if key in EXPERIMENT_SUMMARY_FIELDS
            },
            **{
                key: value
                for key, value in benchmark.items()
                if key in EXPERIMENT_SUMMARY_FIELDS
            },
            "peak_inference_cuda_memory_mib": benchmark.get("peak_cuda_memory_mib"),
            "full_class_num_classes": full_class.get("num_classes"),
            "full_class_parameter_count": full_class.get("total_parameter_count"),
            "full_class_estimated_macs": full_class.get("total_macs"),
            "full_class_estimated_flops": full_class.get("total_flops"),
        }
        for kind in ("mac", "flop"):
            coverage = profile.get(f"{kind}_coverage", {})
            row[f"{kind}_coverage_complete"] = coverage.get("complete")
            row[f"unsupported_{kind}_operator_types"] = json.dumps(
                coverage.get("unsupported_operator_types", [])
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=EXPERIMENT_SUMMARY_FIELDS, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return path

    @staticmethod
    def _fields(path: Path) -> tuple[str, ...]:
        with path.open(newline="", encoding="utf-8") as file:
            return tuple(next(csv.reader(file), ()))
