"""Compact CNN baseline designed for HCCR inference efficiency."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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


class SqueezeExcitation(nn.Module):
    """Channel reweighting with a small bottleneck MLP implemented as 1x1 convs."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gate(self.pool(inputs))


class EfficientChannelAttention(nn.Module):
    """ECA channel attention without dimensionality reduction."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be a positive odd value")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(inputs).squeeze(-1).transpose(-1, -2)
        weights = self.gate(pooled).transpose(-1, -2).unsqueeze(-1)
        return inputs * weights


class ChannelAttention(nn.Module):
    """CBAM channel gate shared by average- and max-pooled descriptors."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = self.mlp(torch.mean(inputs, dim=(-2, -1), keepdim=True))
        maximum = self.mlp(torch.amax(inputs, dim=(-2, -1), keepdim=True))
        return torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """CBAM spatial gate computed from channel-wise mean and maximum maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("spatial attention kernel_size must be 3 or 7")
        self.convolution = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (
                torch.mean(inputs, dim=1, keepdim=True),
                torch.amax(inputs, dim=1, keepdim=True),
            ),
            dim=1,
        )
        return torch.sigmoid(self.convolution(descriptors))


class ParallelCBAM(nn.Module):
    """Middle-stage C-CBAM variant with parallel channel and spatial gates."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_features = inputs * self.channel(inputs)
        spatial_features = inputs * self.spatial(inputs)
        return 0.5 * (channel_features + spatial_features)


class CrossStageAdd(nn.Module):
    """Project-defined projected additive bridge between adjacent stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        projected = self.projection(source)
        if projected.shape != target.shape:
            raise ValueError(
                "cross-stage projection and target must have identical shapes"
            )
        return target + projected


class CrossStageCBAM(nn.Module):
    """Paper-inspired C-CBAM route adapted to the stage-2→stage-3 boundary.

    The source is refined with the paper's middle-stage parallel CAM/SAM
    arrangement before a learned projection is added to the target stage.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = ParallelCBAM(in_channels)
        self.bridge = CrossStageAdd(in_channels, out_channels)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bridge(self.attention(source), target)


class CrossStagePartialStage(nn.Module):
    """CSP stage with separate bypass/transform paths and an explicit merge."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int,
        split_ratio: float,
    ) -> None:
        super().__init__()
        transformed_channels = round(out_channels * split_ratio)
        bypass_channels = out_channels - transformed_channels
        if min(transformed_channels, bypass_channels) < 1:
            raise ValueError("csp_split_ratio must leave channels in both paths")
        self.bypass = nn.Sequential(
            nn.Conv2d(in_channels, bypass_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(bypass_channels),
            nn.SiLU(inplace=True),
        )
        transformed: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                transformed_channels,
                1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(transformed_channels),
            nn.SiLU(inplace=True),
        ]
        transformed.extend(
            DepthwiseSeparableBlock(
                transformed_channels, transformed_channels, stride=1
            )
            for _ in range(depth)
        )
        self.transformed = nn.Sequential(*transformed)
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.bypass(inputs), self.transformed(inputs)), dim=1)
        return self.merge(features)


class AngularMarginClassifier(nn.Linear):
    """Normalized classifier with train-only CosFace or ArcFace margins."""

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
            raise ValueError("angular classifier kind must be cosface or arcface")
        if scale <= 0:
            raise ValueError("angular classifier scale must be positive")
        if not 0 <= margin < torch.pi / 2:
            raise ValueError("angular classifier margin must be in [0, pi/2)")
        self.kind = kind
        self.scale = scale
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor | None = None,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1)
        ).clamp(-1 + 1e-7, 1 - 1e-7)
        if targets is None or margin_multiplier == 0:
            return cosine * self.scale
        if not 0 <= margin_multiplier <= 1:
            raise ValueError("margin_multiplier must be between 0 and 1")
        target_cosine = cosine.gather(1, targets.unsqueeze(1))
        margin = self.margin * margin_multiplier
        if self.kind == "cosface":
            target_logit = target_cosine - margin
        else:
            target_logit = torch.cos(torch.acos(target_cosine) + margin)
        return cosine.scatter(1, targets.unsqueeze(1), target_logit) * self.scale


class DirectionalInputAdapter(nn.Module):
    """Append deterministic Sobel or four-orientation Gabor response channels."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"grayscale", "grayscale_sobel", "grayscale_gabor"}:
            raise ValueError(
                "input_mode must be one of: grayscale, grayscale_sobel, grayscale_gabor"
            )
        self.mode = mode
        if mode == "grayscale_sobel":
            kernels = torch.tensor(
                [
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                ]
            ).unsqueeze(1)
        elif mode == "grayscale_gabor":
            kernels = torch.stack(
                [_gabor_kernel(angle) for angle in (0.0, 45.0, 90.0, 135.0)]
            ).unsqueeze(1)
        else:
            kernels = torch.empty(0, 1, 1, 1)
        self.register_buffer("kernels", kernels, persistent=False)

    @property
    def output_channels(self) -> int:
        if self.mode == "grayscale_sobel":
            return 2
        if self.mode == "grayscale_gabor":
            return 5
        return 1

    def fixed_filter_macs(self, inputs: torch.Tensor) -> int:
        if self.mode == "grayscale":
            return 0
        kernel_size = self.kernels.shape[-1]
        return (
            inputs.shape[0]
            * inputs.shape[-2]
            * inputs.shape[-1]
            * self.kernels.shape[0]
            * kernel_size
            * kernel_size
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("directional input adapter requires BCHW grayscale input")
        if self.mode == "grayscale":
            return inputs
        padding = self.kernels.shape[-1] // 2
        padded = F.pad(inputs, (padding, padding, padding, padding), mode="replicate")
        responses = F.conv2d(padded, self.kernels)
        if self.mode == "grayscale_sobel":
            directional = responses.square().sum(dim=1, keepdim=True).sqrt()
            directional = (directional / 4.0).clamp(0.0, 1.0)
        else:
            directional = responses.abs().clamp(0.0, 1.0)
        return torch.cat((inputs, directional), dim=1)


def _gabor_kernel(angle_degrees: float) -> torch.Tensor:
    coordinates = torch.arange(-3, 4, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(coordinates, coordinates, indexing="ij")
    angle = math.radians(angle_degrees)
    rotated_x = x_grid * math.cos(angle) + y_grid * math.sin(angle)
    rotated_y = -x_grid * math.sin(angle) + y_grid * math.cos(angle)
    gaussian = torch.exp(-(rotated_x.square() + 0.25 * rotated_y.square()) / 8.0)
    kernel = gaussian * torch.cos(2 * math.pi * rotated_x / 4.0)
    kernel -= kernel.mean()
    return kernel / kernel.abs().sum().clamp_min(1e-12)


class EfficientHCCRNet(nn.Module):
    """Small grayscale CNN with configurable width and stage depths."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width: int = 32,
        stage_depths: tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.1,
        attention: str = "none",
        attention_stages: tuple[int, ...] = (3,),
        cross_stage: str = "none",
        csp_stages: tuple[int, ...] = (),
        csp_split_ratio: float = 0.5,
        classification_head: str = "linear",
        logit_scale: float = 32.0,
        angular_margin: float = 0.2,
        input_mode: str = "grayscale",
    ) -> None:
        super().__init__()
        if len(stage_depths) != 3 or any(depth < 1 for depth in stage_depths):
            raise ValueError("stage_depths must contain three positive values")
        if attention not in {"none", "se", "eca"}:
            raise ValueError("attention must be one of: none, se, eca")
        if len(set(attention_stages)) != len(attention_stages) or any(
            stage not in {1, 2, 3} for stage in attention_stages
        ):
            raise ValueError("attention_stages must contain unique values from 1 to 3")
        if cross_stage not in {"none", "projected_residual", "c_cbam"}:
            raise ValueError(
                "cross_stage must be one of: none, projected_residual, c_cbam"
            )
        if len(set(csp_stages)) != len(csp_stages) or any(
            stage not in {2, 3} for stage in csp_stages
        ):
            raise ValueError("csp_stages must contain unique values from 2 to 3")
        if not 0.0 < csp_split_ratio < 1.0:
            raise ValueError("csp_split_ratio must be between 0 and 1")
        if classification_head not in {"linear", "cosface", "arcface"}:
            raise ValueError(
                "classification_head must be one of: linear, cosface, arcface"
            )
        channels = [width, width * 2, width * 4]
        self.stage_depths = stage_depths
        self.attention = attention
        self.attention_stages = attention_stages
        self.cross_stage = cross_stage
        self.csp_stages = csp_stages
        self.csp_split_ratio = csp_split_ratio
        self.classification_head = classification_head
        self.logit_scale = logit_scale
        self.angular_margin = angular_margin
        if in_channels != 1:
            raise ValueError("EfficientHCCRNet expects one raw grayscale input channel")
        self.input_mode = input_mode
        self.input_adapter = DirectionalInputAdapter(input_mode)
        self.effective_input_channels = self.input_adapter.output_channels
        self.stem = ConvNormAct(self.effective_input_channels, channels[0], stride=2)
        blocks: list[nn.Module] = []
        stage_ranges: list[tuple[int, int]] = []
        previous = channels[0]
        for stage, (output, depth) in enumerate(
            zip(channels, stage_depths, strict=True)
        ):
            stage_start = len(blocks)
            stage_number = stage + 1
            if stage_number in csp_stages:
                blocks.append(
                    CrossStagePartialStage(
                        previous,
                        output,
                        depth,
                        stride=2 if stage > 0 else 1,
                        split_ratio=csp_split_ratio,
                    )
                )
                previous = output
            else:
                for block_index in range(depth):
                    stride = 2 if stage > 0 and block_index == 0 else 1
                    blocks.append(DepthwiseSeparableBlock(previous, output, stride))
                    previous = output
            stage_ranges.append((stage_start, len(blocks)))
        self.features = nn.Sequential(*blocks)
        self.stage_ranges = tuple(stage_ranges)
        attention_modules: dict[str, nn.Module] = {}
        for stage_index, stage_channels in enumerate(channels, start=1):
            if attention == "none" or stage_index not in attention_stages:
                continue
            attention_modules[str(stage_index)] = (
                SqueezeExcitation(stage_channels)
                if attention == "se"
                else EfficientChannelAttention()
            )
        self.stage_attention = nn.ModuleDict(attention_modules)
        self.cross_stage_bridge: CrossStageAdd | CrossStageCBAM | None
        if cross_stage == "projected_residual":
            self.cross_stage_bridge = CrossStageAdd(channels[1], channels[2])
        elif cross_stage == "c_cbam":
            self.cross_stage_bridge = CrossStageCBAM(channels[1], channels[2])
        else:
            self.cross_stage_bridge = None
        self.pool = nn.AdaptiveAvgPool2d(1)
        if classification_head == "linear":
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout), nn.Linear(previous, num_classes)
            )
            self.embedding_dropout = None
        else:
            self.classifier = AngularMarginClassifier(
                previous,
                num_classes,
                classification_head,
                logit_scale,
                angular_margin,
            )
            self.embedding_dropout = nn.Dropout(dropout)

    def forward_features(
        self, inputs: torch.Tensor, *, return_stages: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the backbone while exposing stable logical stage boundaries."""
        features = self.stem(self.input_adapter(inputs))
        stage_outputs: dict[str, torch.Tensor] = {}
        previous_stage_features: torch.Tensor | None = None
        for stage_index, (start, end) in enumerate(self.stage_ranges, start=1):
            for block_index in range(start, end):
                features = self.features[block_index](features)
            if stage_index == 3 and self.cross_stage_bridge is not None:
                if previous_stage_features is None:
                    raise RuntimeError("cross-stage bridge requires a stage-2 source")
                features = self.cross_stage_bridge(previous_stage_features, features)
            stage_key = str(stage_index)
            if stage_key in self.stage_attention:
                features = self.stage_attention[stage_key](features)
            if return_stages:
                stage_outputs[f"stage{stage_index}"] = features
            previous_stage_features = features
        if return_stages:
            return features, stage_outputs
        return features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        pooled = self.pool(features)
        if isinstance(self.classifier, AngularMarginClassifier):
            embeddings = torch.flatten(pooled, 1)
            assert self.embedding_dropout is not None
            return self.classifier(self.embedding_dropout(embeddings))
        return self.classifier(pooled)

    def training_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Return logits with a target margin only for angular training heads."""
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        pooled = self.pool(features)
        if isinstance(self.classifier, AngularMarginClassifier):
            embeddings = torch.flatten(pooled, 1)
            assert self.embedding_dropout is not None
            return self.classifier(
                self.embedding_dropout(embeddings), targets, margin_multiplier
            )
        return self.classifier(pooled)


def build_model(name: str, **kwargs: object) -> nn.Module:
    if name != "efficient_hccr":
        raise ValueError(f"unsupported model: {name}")
    return EfficientHCCRNet(**kwargs)
