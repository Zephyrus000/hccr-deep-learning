"""Classification metrics independent of trainer and CLI."""

from __future__ import annotations

import torch


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    top1 = logits.argmax(dim=1).eq(targets).float().mean().item()
    k = min(5, logits.shape[1])
    top5 = (
        logits.topk(k, dim=1)
        .indices.eq(targets[:, None])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    return {"top1": top1, "top5": top5}
