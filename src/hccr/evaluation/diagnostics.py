"""Validation artifacts used to diagnose model and dataset quality."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class StreamingValidationDiagnostics:
    """Accumulate validation reports batch by batch without retaining logits."""

    def __init__(self, output_dir: Path, source_root: Path | None = None) -> None:
        self.output_dir = output_dir
        self.source_root = source_root
        self.support: Counter[int] = Counter()
        self.hits: Counter[int] = Counter()
        self.bin_counts = [0] * 10
        self.bin_correct = [0] * 10
        self.bin_confidence = [0.0] * 10
        self.errors: list[dict[str, Any]] = []
        self.total_samples = 0
        self.total_confidence = 0.0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        rows: Sequence[dict[str, Any]],
    ) -> None:
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        top_values, top_indices = probabilities.topk(
            min(5, probabilities.shape[1]), dim=1
        )
        for row, target, prediction, score, scores, indices in zip(
            rows,
            targets.tolist(),
            predictions.tolist(),
            confidence.tolist(),
            top_values.tolist(),
            top_indices.tolist(),
            strict=True,
        ):
            self.total_samples += 1
            self.total_confidence += score
            self.support[target] += 1
            is_correct = target == prediction
            if is_correct:
                self.hits[target] += 1
            bin_index = min(9, int(score * 10))
            self.bin_counts[bin_index] += 1
            self.bin_correct[bin_index] += int(is_correct)
            self.bin_confidence[bin_index] += score
            if not is_correct and len(self.errors) < 1000:
                self.errors.append(
                    {
                        "row": row,
                        "target": target,
                        "prediction": prediction,
                        "confidence": score,
                        "margin": scores[0] - scores[1] if len(scores) > 1 else 1.0,
                        "top5": indices,
                    }
                )

    def write(self) -> dict[str, float]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_per_class_metrics(self.output_dir, self.support, self.hits)
        _write_error_records(self.output_dir, self.errors)
        ece = _write_reliability_from_bins(
            self.output_dir, self.bin_counts, self.bin_correct, self.bin_confidence
        )
        if self.source_root is not None:
            _write_error_gallery_from_records(
                self.output_dir, self.errors, self.source_root
            )
        health = {
            "samples": self.total_samples,
            "classes": len(self.support),
            "mean_confidence": self.total_confidence / self.total_samples,
            "expected_calibration_error": ece,
            "recorded_errors": len(self.errors),
        }
        (self.output_dir / "validation_health.json").write_text(
            json.dumps(health, indent=2) + "\n", encoding="utf-8"
        )
        return health


def write_validation_diagnostics(
    output_dir: Path,
    logits: torch.Tensor,
    targets: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    source_root: Path | None = None,
) -> dict[str, float]:
    """Write per-class, calibration, error and data-health validation artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    support = Counter(targets.tolist())
    hits = Counter(targets[correct].tolist())
    _write_per_class_metrics(output_dir, support, hits)
    _write_error_table(output_dir, probabilities, targets, predictions, rows)
    ece = _write_reliability_diagram(output_dir, confidence, correct)
    if source_root is not None:
        _write_error_gallery(output_dir, predictions, targets, rows, source_root)

    health = {
        "samples": len(rows),
        "classes": len(support),
        "mean_confidence": confidence.mean().item(),
        "expected_calibration_error": ece,
    }
    (output_dir / "validation_health.json").write_text(
        json.dumps(health, indent=2) + "\n", encoding="utf-8"
    )
    return health


def _write_per_class_metrics(
    output_dir: Path, support: Counter[int], hits: Counter[int]
) -> None:
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["class_id", "support", "top1_accuracy"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "class_id": class_id,
                "support": count,
                "top1_accuracy": hits[class_id] / count,
            }
            for class_id, count in sorted(support.items())
        )


def _write_error_table(
    output_dir: Path,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    rows: Sequence[dict[str, Any]],
) -> None:
    top_values, top_indices = probabilities.topk(min(5, probabilities.shape[1]), dim=1)
    fields = [
        "sample_id",
        "source_file",
        "target",
        "prediction",
        "confidence",
        "margin",
        "top5",
    ]
    with (output_dir / "validation_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, target, prediction, scores, indices in zip(
            rows, targets, predictions, top_values, top_indices, strict=True
        ):
            if target.item() == prediction.item():
                continue
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_file": row["source_file"],
                    "target": target.item(),
                    "prediction": prediction.item(),
                    "confidence": scores[0].item(),
                    "margin": (
                        (scores[0] - scores[1]).item() if len(scores) > 1 else 1.0
                    ),
                    "top5": json.dumps(indices.tolist()),
                }
            )


def _write_reliability_diagram(
    output_dir: Path, confidence: torch.Tensor, correct: torch.Tensor
) -> float:
    boundaries = torch.linspace(0, 1, 11)
    bin_accuracy, bin_confidence, bin_counts = [], [], []
    expected_calibration_error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        in_bin = (confidence >= lower) & (confidence < upper)
        if upper == 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        count = int(in_bin.sum())
        if count == 0:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            bin_counts.append(0)
            continue
        accuracy = correct[in_bin].float().mean().item()
        average_confidence = confidence[in_bin].mean().item()
        bin_accuracy.append(accuracy)
        bin_confidence.append(average_confidence)
        bin_counts.append(count)
        expected_calibration_error += (
            abs(accuracy - average_confidence) * count / len(confidence)
        )

    figure, axis = plt.subplots(figsize=(6, 5))
    centers = [
        (lower + upper).item() / 2
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    axis.bar(centers, bin_accuracy, width=0.09, alpha=0.75, label="accuracy")
    axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
    axis.plot(centers, bin_confidence, marker="o", label="mean confidence")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="confidence",
        ylabel="accuracy",
        title="Reliability diagram",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=160)
    plt.close(figure)
    (output_dir / "calibration_bins.json").write_text(
        json.dumps(
            {
                "expected_calibration_error": expected_calibration_error,
                "bins": [
                    {
                        "count": count,
                        "accuracy": accuracy,
                        "confidence": mean_confidence,
                    }
                    for count, accuracy, mean_confidence in zip(
                        bin_counts, bin_accuracy, bin_confidence, strict=True
                    )
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return expected_calibration_error


def _write_error_gallery(
    output_dir: Path,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rows: Sequence[dict[str, Any]],
    source_root: Path,
) -> None:
    errors = [
        (row, target.item(), prediction.item())
        for row, target, prediction in zip(rows, targets, predictions, strict=True)
        if target.item() != prediction.item()
    ][:25]
    if not errors:
        return
    tile_size, caption_height, columns = 112, 36, 5
    canvas = Image.new(
        "RGB",
        (
            columns * tile_size,
            ((len(errors) + columns - 1) // columns) * (tile_size + caption_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (row, target, prediction) in enumerate(errors):
        image_path = source_root / row["source_file"]
        with Image.open(image_path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((tile_size, tile_size))
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + caption_height)
        canvas.paste(
            tile,
            (
                x + (tile_size - tile.width) // 2,
                y + (tile_size - tile.height) // 2,
            ),
        )
        draw.text(
            (x + 2, y + tile_size + 2),
            f"true={target} pred={prediction}",
            fill="black",
        )
    canvas.save(output_dir / "validation_error_gallery.png")


def _write_error_records(output_dir: Path, errors: Sequence[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "source_file",
        "target",
        "prediction",
        "confidence",
        "margin",
        "top5",
    ]
    with (output_dir / "validation_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for error in errors:
            writer.writerow(
                {
                    "sample_id": error["row"]["sample_id"],
                    "source_file": error["row"]["source_file"],
                    "target": error["target"],
                    "prediction": error["prediction"],
                    "confidence": error["confidence"],
                    "margin": error["margin"],
                    "top5": json.dumps(error["top5"]),
                }
            )


def _write_reliability_from_bins(
    output_dir: Path,
    bin_counts: Sequence[int],
    bin_correct: Sequence[int],
    bin_confidence: Sequence[float],
) -> float:
    total = sum(bin_counts)
    accuracies = [
        correct / count if count else 0.0
        for count, correct in zip(bin_counts, bin_correct, strict=True)
    ]
    confidences = [
        value / count if count else 0.0
        for count, value in zip(bin_counts, bin_confidence, strict=True)
    ]
    ece = sum(
        abs(accuracy - confidence) * count / total
        for count, accuracy, confidence in zip(
            bin_counts, accuracies, confidences, strict=True
        )
        if count
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    centers = [(index + 0.5) / 10 for index in range(10)]
    axis.bar(centers, accuracies, width=0.09, alpha=0.75, label="accuracy")
    axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
    axis.plot(centers, confidences, marker="o", label="mean confidence")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="confidence",
        ylabel="accuracy",
        title="Reliability diagram",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=160)
    plt.close(figure)
    (output_dir / "calibration_bins.json").write_text(
        json.dumps(
            {
                "expected_calibration_error": ece,
                "bins": [
                    {"count": count, "accuracy": accuracy, "confidence": confidence}
                    for count, accuracy, confidence in zip(
                        bin_counts, accuracies, confidences, strict=True
                    )
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ece


def _write_error_gallery_from_records(
    output_dir: Path, errors: Sequence[dict[str, Any]], source_root: Path
) -> None:
    if not errors:
        return
    rows = [error["row"] for error in errors[:25]]
    predictions = torch.tensor([error["prediction"] for error in errors[:25]])
    targets = torch.tensor([error["target"] for error in errors[:25]])
    _write_error_gallery(output_dir, predictions, targets, rows, source_root)
