"""PyTorch dataset backed by the frozen CSV manifest."""

from __future__ import annotations

from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as vision_functional

from hccr.data.lmdb_store import LMDBImageStore, manifest_digest
from hccr.data.manifest import read_manifest


class HCCRDataset(Dataset[tuple[torch.Tensor, int, object]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform=None,
        class_id_map: dict[int, int] | None = None,
        storage_backend: str = "auto",
        lmdb_path: Path | None = None,
        metadata_mode: str = "full",
    ) -> None:
        self.rows = [
            row
            for row in read_manifest(manifest_path)
            if row["split"] == split
            and (class_id_map is None or int(row["class_id"]) in class_id_map)
        ]
        if storage_backend not in {"auto", "filesystem", "lmdb"}:
            raise ValueError("storage_backend must be auto, filesystem or lmdb")
        if metadata_mode not in {"full", "augmentations", "index"}:
            raise ValueError("metadata_mode must be full, augmentations or index")
        default_lmdb_path = manifest_path.with_name("images.lmdb")
        self.lmdb_path = Path(lmdb_path) if lmdb_path is not None else default_lmdb_path
        self.storage_backend = (
            "lmdb"
            if storage_backend == "lmdb"
            or (storage_backend == "auto" and self.lmdb_path.is_file())
            else "filesystem"
        )
        if self.storage_backend == "filesystem":
            self.root = _resolve_data_root(manifest_path, self.rows)
        else:
            try:
                self.root = _resolve_data_root(manifest_path, self.rows)
            except FileNotFoundError:
                self.root = _data_directory(manifest_path)
        self.image_store = (
            LMDBImageStore(self.lmdb_path, manifest_digest(manifest_path))
            if self.storage_backend == "lmdb"
            else None
        )
        if self.image_store is not None:
            _ = self.image_store.metadata
        self.transform = transform
        self.class_id_map = class_id_map
        self.metadata_mode = metadata_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, object]:
        row = self.rows[index]
        image = self.load_image(row)
        image = self.transform(image) if self.transform else image
        augmentations = ",".join(image.info.get("applied_augmentations", ()))
        tensor = vision_functional.to_dtype(
            vision_functional.to_image(image), torch.float32, scale=True
        )
        class_id = int(row["class_id"])
        target = self.class_id_map[class_id] if self.class_id_map else class_id
        if self.metadata_mode == "augmentations":
            metadata: object = augmentations
        elif self.metadata_mode == "index":
            metadata = index
        else:
            full_metadata = dict(row)
            full_metadata["applied_augmentations"] = augmentations
            metadata = full_metadata
        return tensor, target, metadata

    def load_image(self, row_or_index: dict[str, str] | int) -> Image.Image:
        row = self.rows[row_or_index] if isinstance(row_or_index, int) else row_or_index
        if self.image_store is not None:
            return self.image_store.read_image(row["sample_id"])
        with Image.open(self.root / row["source_file"]) as image:
            return image.convert("L")

    def metadata_for_index(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "source_file": row["source_file"],
        }

    def close(self) -> None:
        if self.image_store is not None:
            self.image_store.close()


def select_class_subset(
    manifest_path: Path, max_classes: int, seed: int
) -> dict[int, int]:
    """Choose a deterministic class subset and remap labels to a compact range."""
    class_ids = sorted({int(row["class_id"]) for row in read_manifest(manifest_path)})
    if max_classes < 2 or max_classes > len(class_ids):
        raise ValueError(f"max_classes must be in [2, {len(class_ids)}]")
    selected = sorted(Random(seed).sample(class_ids, max_classes))
    return {class_id: index for index, class_id in enumerate(selected)}


def _resolve_data_root(manifest_path: Path, rows: list[dict[str, str]]) -> Path:
    """Support manifests relative to either ``data`` or ``data/raw``."""
    data_directory = _data_directory(manifest_path)
    if not rows:
        return data_directory
    source_file = Path(rows[0]["source_file"])
    fallback_directory = manifest_path.parents[2]
    candidates = (
        manifest_path.parent,
        data_directory,
        data_directory / "raw",
        fallback_directory,
        fallback_directory / "raw",
    )
    for candidate in candidates:
        if (candidate / source_file).is_file():
            return candidate
    raise FileNotFoundError(
        "manifest source_file is not found under either "
        f"{data_directory} or {data_directory / 'raw'}: {source_file}"
    )


def resolve_data_root(manifest_path: Path, rows: list[dict[str, str]]) -> Path:
    """Resolve the root used by manifest-relative image paths."""
    return _resolve_data_root(manifest_path, rows)


def _data_directory(manifest_path: Path) -> Path:
    return next(
        (parent for parent in manifest_path.parents if parent.name == "data"),
        manifest_path.parents[2],
    )
