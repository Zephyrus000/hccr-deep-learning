"""Train-only HOG/TDA heads; deployment remains an ordinary EfficientHCCRNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.models.heads import AngularMarginClassifier
from hccr.models.hog_fusion import HistogramOfOrientedGradients, HOGConfig


@dataclass(frozen=True)
class SupervisionConfig:
    method: str = "none"
    hog_weight: float = 0.1
    tda_weight: float = 0.1
    warmup_epochs: int = 2

    def __post_init__(self):
        if self.method not in {"none", "hog", "tda", "both"}:
            raise ValueError("method must be none, hog, tda or both")
        if (
            not 0 <= self.hog_weight < float("inf")
            or not 0 <= self.tda_weight < float("inf")
            or self.warmup_epochs < 0
        ):
            raise ValueError("loss weights must be finite/nonnegative; warmup >= 0")
        if self.method in {"hog", "both"} and self.hog_weight == 0:
            raise ValueError("HOG supervision requires a positive hog_weight")
        if self.method in {"tda", "both"} and self.tda_weight == 0:
            raise ValueError("TDA supervision requires a positive tda_weight")

    def scale(self, epoch: int) -> float:
        if epoch < 1:
            raise ValueError("training epochs start at one")
        return min(1.0, epoch / self.warmup_epochs) if self.warmup_epochs else 1.0


class DescriptorSupervision(nn.Module):
    """Shared trainable CNN with optional spatial HOG and embedding TDA heads.

    ``forward`` is always the plain classification path, even in train mode.
    ``training_losses`` computes classification and any same-view auxiliary losses.
    Export ``cnn.state_dict()`` with its original architecture metadata.
    """

    def __init__(
        self, cnn: EfficientHCCRNet, config: SupervisionConfig, image_size: int = 96
    ):
        super().__init__()
        if (
            not isinstance(cnn, EfficientHCCRNet)
            or cnn.embedding_dim != 320
            or not isinstance(cnn.classifier, AngularMarginClassifier)
            or cnn.classifier.kind != "cosface"
        ):
            raise ValueError("supervision requires a 320-D EfficientHCCRNet/CosFace")
        self.cnn = cnn.requires_grad_(True)
        self.config = config
        self.hog = None
        self.hog_head = None
        self.tda_head = None
        # Head initialization must not change the CNN-only dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            if config.method in {"hog", "both"}:
                self.hog = HistogramOfOrientedGradients(
                    HOGConfig(image_size=image_size)
                )
                channels = self.hog.config.block_cells**2 * self.hog.config.bins
                blocks = image_size // self.hog.config.cell_size - 1
                self.hog_head = nn.Sequential(
                    nn.Conv2d(cnn.width * 2, channels, kernel_size=1),
                    nn.AdaptiveAvgPool2d((blocks, blocks)),
                )
            if config.method in {"tda", "both"}:
                self.tda_head = nn.Linear(320, 12)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.cnn(images)

    def training_losses(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        *,
        tda_targets: torch.Tensor | None = None,
        tda_active: torch.Tensor | None = None,
        separate_tda_view: bool = False,
        epoch: int = 1,
    ) -> dict[str, torch.Tensor]:
        if self.hog_head is not None:
            features, stages = self.cnn.forward_features(images, return_stages=True)
            embedding = self.cnn.embedding_projection(
                self.cnn.embedding_dropout(torch.flatten(self.cnn.pool(features), 1))
            )
        else:
            embedding = self.cnn._forward_embedding(images)
            stages = {}
        logits = self.cnn.classifier(embedding, targets)
        with torch.autocast(device_type=images.device.type, enabled=False):
            classification = F.cross_entropy(logits.float(), targets)
            hog_loss = classification.new_zeros(())
            tda_loss = classification.new_zeros(())
            if self.hog_head is not None:
                prediction = self.hog_head(stages["stage2"].float())
                # HOG flatten order is block_y, block_x, cell_y, cell_x, bin.
                prediction = prediction.permute(0, 2, 3, 1).flatten(1)
                with torch.no_grad():
                    hog_target = self.hog(images.detach())
                hog_loss = F.mse_loss(prediction, hog_target)
            if self.tda_head is not None and not separate_tda_view:
                tda_loss = self.topology_loss(embedding, tda_targets, tda_active)
            scale = self.config.scale(epoch)
            loss = classification + scale * (
                self.config.hog_weight * hog_loss + self.config.tda_weight * tda_loss
            )
        return {
            "loss": loss,
            "classification_loss": classification,
            "hog_loss": hog_loss,
            "tda_loss": tda_loss,
            "logits": logits,
        }

    def topology_loss(self, embedding, targets, active) -> torch.Tensor:
        if targets is None or active is None:
            raise ValueError("TDA requires train-normalized targets and active mask")
        if targets.shape != (len(embedding), 12) or active.shape != (12,):
            raise ValueError("TDA targets must be Bx12 and mask must have 12 values")
        if active.dtype != torch.bool or not active.any():
            raise ValueError("TDA requires at least one nonconstant target")
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            prediction = self.tda_head(embedding.float())
            return F.smooth_l1_loss(prediction[:, active], targets.detach()[:, active])

    def clean_topology_loss(self, images, targets, active) -> torch.Tensor:
        features = self.cnn.forward_features(images)
        embedding = self.cnn.embedding_projection(
            torch.flatten(self.cnn.pool(features), 1)
        )
        return self.topology_loss(embedding, targets, active)
