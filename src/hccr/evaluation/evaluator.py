"""Validation loop for frozen HCCR splits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from hccr.evaluation.diagnostics import (
    StreamingValidationDiagnostics,
    summarize_recall_by_support,
)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    output_dir: Path | None = None,
    source_root: Path | None = None,
    labels: Mapping[int, str] | None = None,
    class_support: Mapping[int, int] | None = None,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    top1_hits = 0
    top5_hits = 0
    support: Counter[int] = Counter()
    hits: Counter[int] = Counter()
    diagnostics = (
        StreamingValidationDiagnostics(output_dir, source_root, labels, class_support)
        if output_dir is not None
        else None
    )
    for images, targets, metadata in loader:
        logits = model(images.to(device)).cpu()
        targets = targets.cpu()
        predictions = logits.argmax(dim=1)
        support.update(targets.tolist())
        hits.update(targets[predictions.eq(targets)].tolist())
        total_samples += targets.numel()
        top1_hits += predictions.eq(targets).sum().item()
        top5_hits += (
            logits.topk(min(5, logits.shape[1]), dim=1)
            .indices.eq(targets[:, None])
            .any(dim=1)
            .sum()
            .item()
        )
        if diagnostics is None:
            continue
        rows = [
            {
                key: (
                    value[index].item()
                    if isinstance(value[index], torch.Tensor)
                    else value[index]
                )
                for key, value in metadata.items()
            }
            for index in range(len(targets))
        ]
        diagnostics.update(logits, targets, rows)
    if total_samples == 0:
        raise ValueError("evaluation loader produced no samples")
    metrics = {
        "top1": top1_hits / total_samples,
        "top5": top5_hits / total_samples,
        **summarize_recall_by_support(support, hits, class_support),
    }
    if diagnostics is not None:
        metrics.update(diagnostics.write())
    return metrics
