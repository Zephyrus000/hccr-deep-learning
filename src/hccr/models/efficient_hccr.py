"""Compact CNN baseline designed for HCCR inference efficiency."""

from __future__ import annotations

import torch
from torch import nn


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


class EfficientHCCRNet(nn.Module):
    """Small grayscale CNN with configurable width and stage depths."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width: int = 32,
        stage_depths: tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 3 or any(depth < 1 for depth in stage_depths):
            raise ValueError("stage_depths must contain three positive values")
        channels = [width, width * 2, width * 4]
        self.stem = ConvNormAct(in_channels, channels[0], stride=2)
        blocks: list[nn.Module] = []
        previous = channels[0]
        for stage, (output, depth) in enumerate(
            zip(channels, stage_depths, strict=True)
        ):
            for block_index in range(depth):
                stride = 2 if stage > 0 and block_index == 0 else 1
                blocks.append(DepthwiseSeparableBlock(previous, output, stride))
                previous = output
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(previous, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(self.stem(inputs))))


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name != "efficient_hccr":
        raise ValueError(f"unsupported model: {name}")
    return EfficientHCCRNet(**kwargs)
