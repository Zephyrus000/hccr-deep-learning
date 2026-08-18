"""Small, explicit configuration contracts used across workflow layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    manifest_path: Path
    image_size: int = 64
    seed: int = 20260817


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 7186
    in_channels: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path = Path("experiments")
    device: str = "auto"
