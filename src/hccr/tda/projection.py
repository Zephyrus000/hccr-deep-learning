"""Frozen one-pass stage capture and supervised signed scalar projections."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.training.precision import resolve_precision


class FrozenStage2Adapter(nn.Module):
    """Capture stage 2 and the unchanged final embedding in one traversal."""

    def __init__(self, baseline: EfficientHCCRNet) -> None:
        super().__init__()
        if not isinstance(baseline, EfficientHCCRNet):
            raise ValueError("stage 2 PH requires an EfficientHCCRNet checkpoint")
        self.baseline = baseline.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenStage2Adapter:
        super().train(mode)
        self.baseline.eval()
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final, stages = self.baseline.forward_features(images, return_stages=True)
        pooled = self.baseline.embedding_dropout(self.baseline.pool(final).flatten(1))
        embedding = self.baseline.embedding_projection(pooled)
        return embedding, stages["stage2"]

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, stage2 = self.encode(images)
        return embedding, stage2, self.baseline.classifier(embedding)


class ScalarProjection(nn.Conv2d):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, 1, 1, stride=1, padding=0, groups=1, bias=False)
        nn.init.constant_(self.weight, 1 / channels)


def projection_digest(projection: ScalarProjection) -> str:
    return hashlib.sha256(
        projection.weight.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class ProjectionBudget:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 7
    precision: str = "float32"


def fit_projection(
    adapter: FrozenStage2Adapter,
    train_loader,
    validation_loader,
    num_classes: int,
    device: str,
    budget: ProjectionBudget,
) -> tuple[ScalarProjection, dict]:
    torch.manual_seed(budget.seed)
    precision = resolve_precision(budget.precision, device)
    images = next(iter(train_loader))[0].to(device)
    _, stage = adapter.encode(images)
    channels, height, width = stage.shape[1:]
    projection = ScalarProjection(channels).to(device)
    temporary = nn.Linear(height * width, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        [*projection.parameters(), *temporary.parameters()],
        lr=budget.learning_rate,
        weight_decay=budget.weight_decay,
    )
    initial = projection.weight.detach().clone()
    best_loss, best_state = float("inf"), None
    history = []
    started = perf_counter()
    for epoch in range(budget.epochs):
        projection.train()
        temporary.train()
        train_total = train_samples = 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                _, stage = adapter.encode(images)
                logits = temporary(projection(stage).flatten(1))
                loss = F.cross_entropy(logits.float(), targets)
            precision.backward(loss)
            precision.step(optimizer)
            train_total += loss.item() * len(targets)
            train_samples += len(targets)
        projection.eval()
        temporary.eval()
        total = samples = 0
        with torch.no_grad():
            for images, targets, _ in validation_loader:
                with precision.autocast():
                    _, stage = adapter.encode(images.to(device))
                    logits = temporary(projection(stage).flatten(1))
                total += F.cross_entropy(
                    logits.float(), targets.to(device), reduction="sum"
                ).item()
                samples += len(targets)
        validation_loss = total / samples
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_total / train_samples,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(projection.state_dict())
    if best_state is None:
        raise RuntimeError("projection fitting produced no finite validation loss")
    projection.load_state_dict(best_state)
    projection.eval().requires_grad_(False)
    updated = not torch.equal(initial, projection.weight)
    if not updated:
        raise RuntimeError(
            "projection weights did not change; cannot label this projection learned"
        )
    return projection, {
        "training_status": "learned",
        "supervision": "classification_cross_entropy_not_PH",
        "selection_rule": "minimum_validation_loss_first_tie",
        "best_validation_loss": best_loss,
        "selected_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "history": history,
        "budget": vars(budget),
        "precision": precision.metadata(),
        "stage2_shape": [channels, height, width],
        "projection_parameters": channels,
        "temporary_classifier_parameters": sum(
            p.numel() for p in temporary.parameters()
        ),
        "temporary_classifier_discarded": True,
        "weights_updated": updated,
        "wall_seconds": perf_counter() - started,
    }


def save_projection(path: Path, projection: ScalarProjection, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "channels": projection.in_channels,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in projection.state_dict().items()
            },
            "metadata": metadata,
            "weights_digest": projection_digest(projection),
        },
        path,
    )


def load_projection(path: Path, device: str = "cpu") -> tuple[ScalarProjection, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1:
        raise ValueError("incompatible projection checkpoint schema; refit projection")
    projection = ScalarProjection(payload["channels"])
    projection.load_state_dict(payload["state_dict"])
    if projection_digest(projection) != payload["weights_digest"]:
        raise ValueError("projection weight digest mismatch")
    return projection.to(device).eval().requires_grad_(False), payload["metadata"]


@torch.no_grad()
def calibrate_maps(
    adapter: FrozenStage2Adapter,
    projection: ScalarProjection,
    loader,
    sample_ids: list[str],
    device: str,
    max_samples: int,
) -> dict:
    squared, pixels, observed = 0.0, 0, []
    for images, _, indices in loader:
        remaining = max_samples - len(observed)
        if remaining <= 0:
            break
        _, stage = adapter.encode(images[:remaining].to(device))
        maps = projection(stage).double()
        squared += maps.square().sum().item()
        pixels += maps.numel()
        observed.extend(sample_ids[int(index)] for index in indices[:remaining])
    if not pixels:
        raise ValueError("map calibration requires nonempty training samples")
    rms = (squared / pixels) ** 0.5
    if not isfinite(rms):
        raise ValueError(
            "nonfinite training map RMS; check projection/backbone weights"
        )
    return {
        "scale": rms if rms > 0 else 1.0,
        "degenerate_zero_map": rms == 0,
        "sample_ids": observed,
        "split": "train",
        "method": "global_rms_no_clipping",
    }
