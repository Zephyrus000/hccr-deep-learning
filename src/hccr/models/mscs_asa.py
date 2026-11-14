"""MSCS–ASA from Xu et al. (2022), doi:10.1016/j.compeleceng.2022.107857."""

import torch
from torch import nn


def _conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MultiScaleShuffle(nn.Module):
    """The two half-width convolution branches in Figure 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.spatial = _conv(in_channels // 2, out_channels // 2)
        self.pointwise = _conv(in_channels // 2, out_channels // 2, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        left, right = images.chunk(2, dim=1)
        return torch.channel_shuffle(
            torch.cat((self.spatial(left), self.pointwise(right)), dim=1), 2
        )


class AttentionSpatialAggregation(nn.Module):
    """Equations 7–9: squeeze/excite with a spatial sum at the output."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=(2, 3))
        return features.sum(dim=(2, 3)) * self.excite(pooled)


def build_mscs_asa_backbone() -> nn.Sequential:
    """Build the grayscale feature extractor from Table 1."""
    layers = [_conv(1, 64), _conv(64, 64)]
    in_channels = 64
    for out_channels, bottleneck in ((96, 64), (128, 96), (256, 192), (448, 384)):
        layers.extend(
            (
                nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
                _conv(in_channels, out_channels),
                MultiScaleShuffle(out_channels, bottleneck),
                _conv(bottleneck, out_channels),
            )
        )
        in_channels = out_channels
    return nn.Sequential(*layers, AttentionSpatialAggregation(448))
