"""Classification heads shared by proposed and reference architectures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AngularMarginClassifier(nn.Linear):
    """Normalized classifier with a train-only CosFace or ArcFace margin."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        kind: str,
        scale: float,
        margin: float,
    ) -> None:
        super().__init__(embedding_dim, num_classes, bias=False)
        if kind not in {"cosface", "arcface"}:
            raise ValueError("classification_head must be cosface or arcface")
        if scale <= 0:
            raise ValueError("logit_scale must be positive")
        if not 0 <= margin < torch.pi / 2:
            raise ValueError("angular_margin must be in [0, pi/2)")
        self.kind, self.scale, self.margin = kind, scale, margin
        self.register_buffer("_normalized_weight", torch.empty(0), persistent=False)

    def train(self, mode: bool = True) -> AngularMarginClassifier:
        super().train(mode)
        if mode:
            self._normalized_weight = torch.empty(
                0, device=self.weight.device, dtype=self.weight.dtype
            )
        else:
            self._normalized_weight = F.normalize(
                self.weight.detach(), dim=1
            ).contiguous()
        return self

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor | None = None,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        # Cosine normalization and ArcFace acos are sensitive to reduced
        # precision. Keep the head in FP32 while AMP accelerates the backbone.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            normalized_weight = (
                F.normalize(self.weight.float(), dim=1)
                if self.training or self._normalized_weight.numel() == 0
                else self._normalized_weight.float()
            )
            cosine = F.linear(
                F.normalize(embeddings.float(), dim=1), normalized_weight
            ).clamp(-1 + 1e-7, 1 - 1e-7)
            if targets is None or margin_multiplier == 0:
                return cosine * self.scale
            if not 0 <= margin_multiplier <= 1:
                raise ValueError("margin_multiplier must be between 0 and 1")
            target_cosine = cosine.gather(1, targets.unsqueeze(1))
            margin = self.margin * margin_multiplier
            target_logit = (
                target_cosine - margin
                if self.kind == "cosface"
                else torch.cos(torch.acos(target_cosine) + margin)
            )
            return cosine.scatter(1, targets.unsqueeze(1), target_logit) * self.scale
