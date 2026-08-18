"""Validation loop for frozen HCCR splits."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from hccr.evaluation.diagnostics import write_validation_diagnostics
from hccr.evaluation.metrics import classification_metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    output_dir: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, float]:
    model.eval()
    logits_batches, target_batches, rows = [], [], []
    for images, targets, metadata in loader:
        logits_batches.append(model(images.to(device)).cpu())
        target_batches.append(targets.cpu())
        rows.extend(
            {
                key: (
                    value[index].item()
                    if isinstance(value[index], torch.Tensor)
                    else value[index]
                )
                for key, value in metadata.items()
            }
            for index in range(len(targets))
        )
    logits, targets = torch.cat(logits_batches), torch.cat(target_batches)
    metrics = classification_metrics(logits, targets)
    if output_dir is not None:
        metrics.update(
            write_validation_diagnostics(output_dir, logits, targets, rows, source_root)
        )
    return metrics
