"""PyTorch dataset backed by the frozen CSV manifest."""

from __future__ import annotations

from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch.utils.data import Dataset

from hccr.data.manifest import read_manifest


class HCCRDataset(Dataset[tuple[torch.Tensor, int, dict[str, str]]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform=None,
        class_id_map: dict[int, int] | None = None,
    ) -> None:
        self.root = manifest_path.parents[2]
        self.rows = [
            row
            for row in read_manifest(manifest_path)
            if row["split"] == split
            and (class_id_map is None or int(row["class_id"]) in class_id_map)
        ]
        self.transform = transform
        self.class_id_map = class_id_map

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        row = self.rows[index]
        image = Image.open(self.root / row["source_file"]).convert("L")
        image = self.transform(image) if self.transform else image
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = pixels.reshape(1, image.height, image.width).float().div(255)
        class_id = int(row["class_id"])
        target = self.class_id_map[class_id] if self.class_id_map else class_id
        return tensor, target, row


def select_class_subset(
    manifest_path: Path, max_classes: int, seed: int
) -> dict[int, int]:
    """Choose a deterministic class subset and remap labels to a compact range."""
    class_ids = sorted({int(row["class_id"]) for row in read_manifest(manifest_path)})
    if max_classes < 2 or max_classes > len(class_ids):
        raise ValueError(f"max_classes must be in [2, {len(class_ids)}]")
    selected = sorted(Random(seed).sample(class_ids, max_classes))
    return {class_id: index for index, class_id in enumerate(selected)}
