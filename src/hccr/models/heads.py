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
        # This is an inference-only cache, not model state.  Registering it as a
        # buffer makes DDP synchronize it even though its shape changes between
        # train and eval mode.  Rank-zero-only profiling can therefore leave
        # different buffer layouts across ranks before DDP construction.
        self._normalized_weight: torch.Tensor | None = None

    def train(self, mode: bool = True) -> AngularMarginClassifier:
        super().train(mode)
        if mode:
            self._normalized_weight = None
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
            cached_weight = self._normalized_weight
            cache_is_valid = (
                cached_weight is not None
                and cached_weight.device == self.weight.device
                and cached_weight.dtype == self.weight.dtype
            )
            if self.training or not cache_is_valid:
                normalized_weight = F.normalize(self.weight.float(), dim=1)
            else:
                assert cached_weight is not None
                normalized_weight = cached_weight.float()
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
