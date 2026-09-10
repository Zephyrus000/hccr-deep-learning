"""Compact CNN for the retained HCCR experiment family."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from hccr.models.baselines import build_baseline_model

MODEL_NAMES = ("efficient_hccr", "resnet18", "mobilenet_v3_small")


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        stride: int,
        reparameterize_depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.reparameterize_depthwise = reparameterize_depthwise
        if reparameterize_depthwise:
            self.depthwise_branches = nn.ModuleList(
                [
                    self._depthwise_branch(channels, (3, 3), stride, (1, 1)),
                    self._depthwise_branch(channels, (1, 3), stride, (0, 1)),
                    self._depthwise_branch(channels, (3, 1), stride, (1, 0)),
                ]
            )
            self.depthwise_activation = nn.SiLU(inplace=True)
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.block = nn.Identity()
        else:
            self.depthwise_branches = nn.ModuleList()
            self.depthwise_activation = nn.Identity()
            self.pointwise = nn.Identity()
            self.block = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, stride, 1, groups=channels, bias=False
                ),
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
        if self.reparameterize_depthwise:
            branch_outputs = [branch(inputs) for branch in self.depthwise_branches]
            spatial = branch_outputs[0] + branch_outputs[1] + branch_outputs[2]
            projected = self.pointwise(self.depthwise_activation(spatial))
        else:
            projected = self.block(inputs)
        return self.activation(projected + self.skip(inputs))

    def fuse_for_inference(self) -> None:
        """Collapse training branches and Conv-BN pairs into the deploy block."""
        if self.training:
            raise RuntimeError("depthwise block must be in eval mode before fusion")
        if self.reparameterize_depthwise:
            fused_branches = [
                fuse_conv_bn_eval(branch[0], branch[1])
                for branch in self.depthwise_branches
            ]
            reference = fused_branches[0]
            kernel = reference.weight.detach().clone()
            bias = reference.bias.detach().clone()
            kernel += F.pad(fused_branches[1].weight.detach(), (0, 0, 1, 1))
            kernel += F.pad(fused_branches[2].weight.detach(), (1, 1, 0, 0))
            bias += fused_branches[1].bias.detach() + fused_branches[2].bias.detach()
            fused_depthwise = nn.Conv2d(
                reference.in_channels,
                reference.out_channels,
                3,
                reference.stride,
                1,
                groups=reference.groups,
                bias=True,
            ).to(device=kernel.device, dtype=kernel.dtype)
            with torch.no_grad():
                fused_depthwise.weight.copy_(kernel)
                fused_depthwise.bias.copy_(bias)
            fused_pointwise = fuse_conv_bn_eval(self.pointwise[0], self.pointwise[1])
            self.block = nn.Sequential(
                fused_depthwise,
                nn.Identity(),
                nn.SiLU(inplace=True),
                fused_pointwise,
                nn.Identity(),
            )
            self.depthwise_branches = nn.ModuleList()
            self.depthwise_activation = nn.Identity()
            self.pointwise = nn.Identity()
            self.reparameterize_depthwise = False
            return
        self.block[0] = fuse_conv_bn_eval(self.block[0], self.block[1])
        self.block[1] = nn.Identity()
        self.block[3] = fuse_conv_bn_eval(self.block[3], self.block[4])
        self.block[4] = nn.Identity()

    @staticmethod
    def _depthwise_branch(
        channels: int,
        kernel_size: tuple[int, int],
        stride: int,
        padding: tuple[int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                stride,
                padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )


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
        stem_stride: int = 2,
        backbone_output_channels: int | None = None,
        embedding_dim: int | None = None,
        reparameterize_depthwise: bool = False,
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
        if stem_stride not in {1, 2}:
            raise ValueError("stem_stride must be 1 or 2")
        if backbone_output_channels is not None and backbone_output_channels < 1:
            raise ValueError("backbone_output_channels must be positive when set")
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError("embedding_dim must be positive when set")
        if classification_head not in {"cosface", "arcface", "softmax"}:
            raise ValueError("classification_head must be cosface, arcface, or softmax")
        self.name = "efficient_hccr"
        self.stage_depths, self.width = stage_depths, width
        self.stem_stride = stem_stride
        self.reparameterize_depthwise = reparameterize_depthwise
        self.classification_head, self.logit_scale, self.angular_margin = (
            classification_head,
            logit_scale,
            angular_margin,
        )
        self.effective_input_channels = 1
        channels = [width, width * 2, width * 4]
        self.stem = ConvNormAct(1, channels[0], stride=stem_stride)
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
                        previous,
                        output,
                        2 if stage > 0 and block_index == 0 else 1,
                        reparameterize_depthwise,
                    )
                )
                previous = output
            stage_ranges.append((start, len(blocks)))
        self.features, self.stage_ranges = nn.Sequential(*blocks), tuple(stage_ranges)
        feature_channels = previous
        self.backbone_output_channels = backbone_output_channels or feature_channels
        self.late_stage_projection = (
            nn.Identity()
            if self.backbone_output_channels == feature_channels
            else nn.Sequential(
                nn.Conv2d(
                    feature_channels,
                    self.backbone_output_channels,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.backbone_output_channels),
                nn.SiLU(inplace=True),
            )
        )
        self.embedding_dim = embedding_dim or feature_channels
        self.pool, self.embedding_dropout = nn.AdaptiveAvgPool2d(1), nn.Dropout(dropout)
        self.embedding_projection = (
            nn.Identity()
            if self.embedding_dim == self.backbone_output_channels
            else nn.Linear(
                self.backbone_output_channels,
                self.embedding_dim,
                bias=False,
            )
        )
        self.classifier: nn.Module
        if classification_head == "softmax":
            self.classifier = nn.Linear(self.embedding_dim, num_classes)
        else:
            self.classifier = AngularMarginClassifier(
                self.embedding_dim,
                num_classes,
                classification_head,
                logit_scale,
                angular_margin,
            )

    def forward_features(
        self, inputs: torch.Tensor, *, return_stages: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("EfficientHCCRNet requires BCHW grayscale input")
        features = self.stem(inputs)
        if not return_stages:
            return self.late_stage_projection(self.features(features))
        stage_outputs = {}
        for stage_index, (start, end) in enumerate(self.stage_ranges, start=1):
            for block_index in range(start, end):
                features = self.features[block_index](features)
            if return_stages:
                stage_outputs[f"stage{stage_index}"] = features
        features = self.late_stage_projection(features)
        return (features, stage_outputs) if return_stages else features

    def _forward_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        assert isinstance(features, torch.Tensor)
        pooled = self.embedding_dropout(torch.flatten(self.pool(features), 1))
        return self.embedding_projection(pooled)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Return inference logits or target-margin training logits.

        Keeping the margin path inside ``forward`` lets a DDP wrapper observe
        the complete forward/backward graph instead of bypassing its reducer.
        """
        embeddings = self._forward_embedding(inputs)
        if isinstance(self.classifier, AngularMarginClassifier):
            return self.classifier(embeddings, targets, margin_multiplier)
        return self.classifier(embeddings)

    def training_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compatibility wrapper for callers outside DDP training."""
        return self(inputs, targets, margin_multiplier)


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a proposed architecture or a fixed reference baseline."""
    if name == "efficient_hccr":
        return EfficientHCCRNet(**kwargs)
    return build_baseline_model(name, **kwargs)


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
        block.fuse_for_inference()
    late_projection = optimized.late_stage_projection
    if isinstance(late_projection, nn.Sequential):
        late_projection[0] = fuse_conv_bn_eval(late_projection[0], late_projection[1])
        late_projection[1] = nn.Identity()
    optimized.embedding_dropout = nn.Identity()
    return optimized.requires_grad_(False)
