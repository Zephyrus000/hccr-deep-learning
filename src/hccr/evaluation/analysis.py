"""Per-sample errors and confusion pairs for model debugging."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch


def write_error_analysis(
    output_dir: Path, logits: torch.Tensor, targets: torch.Tensor, rows
) -> None:
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    errors, pairs = [], Counter()
    for row, target, prediction, confidence in zip(
        rows, targets, predictions, probabilities.max(dim=1).values, strict=True
    ):
        if target.item() != prediction.item():
            errors.append(
                {
                    "sample_id": row["sample_id"],
                    "source_file": row["source_file"],
                    "target": target.item(),
                    "prediction": prediction.item(),
                    "confidence": confidence.item(),
                }
            )
            pairs[(target.item(), prediction.item())] += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fieldnames, records in (
        (
            output_dir / "validation_errors.csv",
            ["sample_id", "source_file", "target", "prediction", "confidence"],
            errors,
        ),
        (
            output_dir / "confusion_pairs.csv",
            ["target", "prediction", "count"],
            [
                {"target": a, "prediction": b, "count": count}
                for (a, b), count in pairs.most_common()
            ],
        ),
    ):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
