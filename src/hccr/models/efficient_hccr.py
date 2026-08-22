"""Compact CNN for the retained HCCR experiment family."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and channels == out_channels
            else nn.Conv2d(channels, out_channels, 1, stride, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs) + self.skip(inputs))


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
        normalized_weight = (
            F.normalize(self.weight, dim=1)
            if self.training or self._normalized_weight.numel() == 0
            else self._normalized_weight
        )
        cosine = F.linear(F.normalize(embeddings, dim=1), normalized_weight).clamp(
            -1 + 1e-7, 1 - 1e-7
        )
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


class EfficientHCCRNet(nn.Module):
    """Grayscale CNN limited to architectures that survived ablation."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width: int = 64,
        stage_depths: tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.1,
        classification_head: str = "cosface",
        logit_scale: float = 32.0,
        angular_margin: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("EfficientHCCRNet expects one grayscale input channel")
        if width < 1:
            raise ValueError("width must be positive")
        if len(stage_depths) != 3 or any(depth < 1 for depth in stage_depths):
            raise ValueError("stage_depths must contain three positive values")
        self.stage_depths, self.width = stage_depths, width
        self.classification_head, self.logit_scale, self.angular_margin = (
            classification_head,
            logit_scale,
            angular_margin,
        )
        self.effective_input_channels = 1
        channels = [width, width * 2, width * 4]
        self.stem = ConvNormAct(1, channels[0], stride=2)
        blocks: list[nn.Module] = []
        stage_ranges: list[tuple[int, int]] = []
        previous = channels[0]
        for stage, (output, depth) in enumerate(
            zip(channels, stage_depths, strict=True)
        ):
            start = len(blocks)
            for block_index in range(depth):
                blocks.append(
                    DepthwiseSeparableBlock(
                        previous, output, 2 if stage > 0 and block_index == 0 else 1
                    )
                )
                previous = output
            stage_ranges.append((start, len(blocks)))
        self.features, self.stage_ranges = nn.Sequential(*blocks), tuple(stage_ranges)
        self.pool, self.embedding_dropout = nn.AdaptiveAvgPool2d(1), nn.Dropout(dropout)
        self.classifier = AngularMarginClassifier(
            previous, num_classes, classification_head, logit_scale, angular_margin
        )

    def forward_features(
        self, inputs: torch.Tensor, *, return_stages: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("EfficientHCCRNet requires BCHW grayscale input")
        features = self.stem(inputs)
        if not return_stages:
            return self.features(features)
        stage_outputs = {}
        for stage_index, (start, end) in enumerate(self.stage_ranges, start=1):
            for block_index in range(start, end):
                features = self.features[block_index](features)
            if return_stages:
                stage_outputs[f"stage{stage_index}"] = features
        return (features, stage_outputs) if return_stages else features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        return self.classifier(
            self.embedding_dropout(torch.flatten(self.pool(features), 1))
        )

    def training_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        return self.classifier(
            self.embedding_dropout(torch.flatten(self.pool(features), 1)),
            targets,
            margin_multiplier,
        )


def build_model(name: str, **kwargs) -> EfficientHCCRNet:
    """Build the single retained model family."""
    if name != "efficient_hccr":
        raise ValueError("model name must be efficient_hccr")
    return EfficientHCCRNet(**kwargs)


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Return an eval-only copy with exact Conv-BN folding and no dropout hop."""
    optimized = deepcopy(model).eval()
    if not isinstance(optimized, EfficientHCCRNet):
        return optimized.requires_grad_(False)
    optimized.stem[0] = fuse_conv_bn_eval(optimized.stem[0], optimized.stem[1])
    optimized.stem[1] = nn.Identity()
    for block in optimized.features:
        if not isinstance(block, DepthwiseSeparableBlock):
            continue
        block.block[0] = fuse_conv_bn_eval(block.block[0], block.block[1])
        block.block[1] = nn.Identity()
        block.block[3] = fuse_conv_bn_eval(block.block[3], block.block[4])
        block.block[4] = nn.Identity()
    optimized.embedding_dropout = nn.Identity()
    return optimized.requires_grad_(False)
