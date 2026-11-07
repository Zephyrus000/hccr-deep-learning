"""Frozen EfficientHCCR embeddings fused with a compact HOG projection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from hccr.models.artifacts import build_model_from_metadata
from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.models.heads import AngularMarginClassifier
from hccr.training.precision import resolve_precision


@dataclass(frozen=True)
class HOGConfig:
    image_size: int = 96
    cell_size: int = 8
    bins: int = 9
    block_cells: int = 2
    clip: float = 0.2
    eps: float = 1e-5

    def __post_init__(self):
        if (
            self.cell_size < 1
            or self.image_size % self.cell_size
            or self.image_size // self.cell_size < self.block_cells
            or self.block_cells < 1
            or self.bins < 2
            or self.clip <= 0
            or self.eps <= 0
        ):
            raise ValueError("invalid HOG geometry or normalization")

    @property
    def dimension(self) -> int:
        blocks = self.image_size // self.cell_size - self.block_cells + 1
        return blocks**2 * self.block_cells**2 * self.bins


class HistogramOfOrientedGradients(nn.Module):
    """Unsigned hard-bin HOG with central differences and L2-Hys blocks.

    Cells are non-overlapping, blocks overlap with a one-cell stride, and
    descriptors are ordered by block row/column, cell row/column, then bin.
    Boundary gradients are zero; gamma correction and spatial interpolation
    are disabled. All descriptor arithmetic runs in FP32.
    """

    def __init__(self, config: HOGConfig):
        super().__init__()
        self.config = config
        axis = torch.arange(config.image_size) // config.cell_size
        cells = config.image_size // config.cell_size
        self.register_buffer(
            "cell_indices",
            (axis[:, None] * cells + axis[None, :]).flatten(),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if images.ndim != 4 or images.shape[1:] != (1, cfg.image_size, cfg.image_size):
            raise ValueError("HOG requires BCHW grayscale images of its declared size")
        with torch.autocast(device_type=images.device.type, enabled=False):
            pixels = images[:, 0].float()
            gx, gy = torch.zeros_like(pixels), torch.zeros_like(pixels)
            gx[:, :, 1:-1] = pixels[:, :, 2:] - pixels[:, :, :-2]
            gy[:, 1:-1, :] = pixels[:, 2:, :] - pixels[:, :-2, :]
            magnitude = torch.sqrt(gx.square() + gy.square())
            angle = torch.remainder(torch.atan2(gy, gx), math.pi)
            bins = (angle * (cfg.bins / math.pi)).long().clamp_max(cfg.bins - 1)
            indices = self.cell_indices[None, :] * cfg.bins + bins.flatten(1)
            cells = cfg.image_size // cfg.cell_size
            histogram = pixels.new_zeros((len(images), cells * cells * cfg.bins))
            histogram.scatter_add_(1, indices, magnitude.flatten(1))
            histogram = histogram.reshape(-1, cells, cells, cfg.bins)
            histogram = histogram.permute(0, 3, 1, 2) / cfg.cell_size**2
            blocks = F.unfold(histogram, kernel_size=cfg.block_cells, stride=1)
            blocks = (
                blocks.reshape(len(images), cfg.bins, cfg.block_cells**2, -1)
                .permute(0, 3, 2, 1)
                .flatten(2)
            )
            blocks = blocks / torch.sqrt(
                blocks.square().sum(-1, keepdim=True) + cfg.eps**2
            )
            blocks = blocks.clamp_max(cfg.clip)
            blocks = blocks / torch.sqrt(
                blocks.square().sum(-1, keepdim=True) + cfg.eps**2
            )
            return blocks.flatten(1)


class HOGCNNFusion(nn.Module):
    """Train only HOG -> 64 and concat(320,64) -> 320 before frozen CosFace."""

    def __init__(
        self,
        cnn: EfficientHCCRNet,
        hog_config: HOGConfig,
        hog_dim: int = 64,
        backbone_precision: str = "float32",
    ):
        super().__init__()
        if (
            not isinstance(cnn, EfficientHCCRNet)
            or cnn.embedding_dim != 320
            or not isinstance(cnn.classifier, AngularMarginClassifier)
            or cnn.classifier.kind != "cosface"
            or hog_dim < 1
        ):
            raise ValueError(
                "fusion requires a 320D EfficientHCCRNet with dense CosFace"
            )
        self.cnn = cnn.requires_grad_(False).eval()
        self.hog = HistogramOfOrientedGradients(hog_config)
        self.hog_dim = hog_dim
        self.backbone_precision = backbone_precision
        self.hog_projection = nn.Linear(hog_config.dimension, hog_dim)
        self.hog_normalization = nn.LayerNorm(hog_dim)
        self.fusion = nn.Linear(320 + hog_dim, 320)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :320].copy_(torch.eye(320))
            self.fusion.bias.zero_()

    def train(self, mode: bool = True):
        super().train(mode)
        self.cnn.eval()
        return self

    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        precision = resolve_precision(self.backbone_precision, str(images.device))
        with torch.no_grad(), precision.autocast():
            cnn = self.cnn._forward_embedding(images)
        with torch.no_grad():
            hog = self.hog(images)
        return cnn.float(), hog

    def forward_cached(self, cnn, hog, targets=None, margin_multiplier=1.0):
        with torch.autocast(device_type=cnn.device.type, enabled=False):
            projected = self.hog_normalization(self.hog_projection(hog.float()))
            embedding = self.fusion(torch.cat((cnn.float(), projected), dim=1))
            return self.cnn.classifier(embedding, targets, margin_multiplier)

    def forward(self, images, targets=None, margin_multiplier=1.0):
        return self.forward_cached(
            *self.extract(images), targets=targets, margin_multiplier=margin_multiplier
        )

    def specification(self) -> dict:
        return {
            "name": "hog_cnn_fusion",
            "hog": asdict(self.hog.config),
            "hog_dim": self.hog_dim,
            "hog_normalization": "layer_norm",
            "backbone_precision": self.backbone_precision,
            "fusion_initialization": "identity_cnn_zero_hog",
            "freeze_backbone_and_classifier": True,
        }


def load_hog_fusion_run(run_dir: Path, device: str = "cpu") -> HOGCNNFusion:
    """Load a self-contained fusion checkpoint without its parent's files."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text("utf-8"))
    spec = metadata["model"]
    if spec["name"] != "hog_cnn_fusion":
        raise ValueError("not a HOG-CNN fusion artifact")
    cnn = build_model_from_metadata({"model": spec["cnn"]})
    model = HOGCNNFusion(
        cnn, HOGConfig(**spec["hog"]), spec["hog_dim"], spec["backbone_precision"]
    )
    model.load_state_dict(
        torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    )
    return model.to(device).eval()
