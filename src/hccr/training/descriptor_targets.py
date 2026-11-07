"""Image cubical-PH targets, with train-only scaling and checked disk caches."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hccr.artifacts.checkpoints import file_digest
from hccr.tda.cubical import FILTRATION, PH_FEATURE_NAMES, cubical_features
from hccr.training.hog_fusion import FusionImages, write_json

TDA_INDICES = tuple(index for index in range(14) if index not in (6, 13))
TDA_TARGET_NAMES = tuple(PH_FEATURE_NAMES[index] for index in TDA_INDICES)


class IndexedImages(FusionImages):
    """Keep compact LMDB metadata and return the stable row index with each image."""

    def __init__(self, manifest, split, labels, preprocess, manifest_sha):
        super().__init__(manifest, split, labels, preprocess, manifest_sha)
        self.manifest = Path(manifest)
        self.split = split

    @property
    def rows(self):
        with self.manifest.open(newline="", encoding="utf-8") as stream:
            yield from (
                row for row in csv.DictReader(stream) if row["split"] == self.split
            )

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


def image_tda_targets(images: torch.Tensor) -> np.ndarray:
    """Sublevel PH of grayscale intensity: dark ink enters before white canvas."""
    if images.ndim != 4 or images.shape[1] != 1 or len(images) == 0:
        raise ValueError("TDA requires a nonempty BCHW grayscale image batch")
    values = images.detach().cpu().double().numpy()[:, 0]
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("TDA image targets require finite pixels in [0,1]")
    return np.stack(
        [cubical_features(image).features[list(TDA_INDICES)] for image in values]
    ).astype(np.float32)


@dataclass(frozen=True)
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> TargetScaler:
        if train.ndim != 2 or train.shape[1] != 12 or not len(train):
            raise ValueError("fit requires nonempty training targets with 12 columns")
        # Bounded two-pass reduction; never materialize the full target cache.
        total, squares, count = np.zeros(12), np.zeros(12), 0
        for start in range(0, len(train), 8192):
            block = np.asarray(train[start : start + 8192], dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite training topology target")
            total += block.sum(axis=0)
            count += len(block)
        mean = total / count
        for start in range(0, len(train), 8192):
            block = np.asarray(train[start : start + 8192], dtype=np.float64)
            squares += ((block - mean) ** 2).sum(axis=0)
        std = np.sqrt(squares / count)
        active = std > 1e-8
        if not active.any():
            raise ValueError("all training topology targets are constant")
        return cls(
            mean.astype(np.float32), np.maximum(std, 1e-8).astype(np.float32), active
        )

    def transform(self, values: np.ndarray) -> torch.Tensor:
        result = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        result[:, ~self.active] = 0
        return torch.from_numpy(np.array(result, copy=True))

    def metadata(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "active": self.active.tolist(),
        }


def target_identity(
    dataset: IndexedImages, manifest_sha: str, preprocess: dict
) -> dict:
    return {
        "schema": 1,
        "manifest_sha256": manifest_sha,
        "ordered_ids_sha256": dataset.ordered_ids_sha256,
        "samples": len(dataset),
        "preprocess": preprocess,
        "filtration": FILTRATION,
        "target_source": "preprocessed_grayscale_image_not_cnn_activation",
        "versions": {name: version(name) for name in ("gudhi", "numpy", "Pillow")},
        "target_names": list(TDA_TARGET_NAMES),
        "cubical_implementation_sha256": file_digest(
            Path(__file__).parents[1] / "tda/cubical.py"
        ),
        "target_implementation_sha256": file_digest(Path(__file__)),
        "image_pipeline_sha256": {
            relative: file_digest(Path(__file__).parents[1] / relative)
            for relative in (
                "training/hog_fusion.py",
                "preprocessing/pipeline.py",
                "data/lmdb_store.py",
            )
        },
    }


def prepare_tda_cache(
    dataset: Dataset,
    folder: Path,
    identity: dict,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> np.ndarray:
    """Prepare deterministic image targets; completed caches are immutable."""
    folder = Path(folder)
    if folder.exists():
        return load_tda_cache(folder, identity)
    folder.mkdir(parents=True)
    path = folder / "targets.npy"
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(dataset), 12)
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    completed = 0
    for images, _, indices in loader:
        values[indices.numpy()] = image_tda_targets(images)
        completed += len(images)
    if completed != len(dataset):
        raise RuntimeError("incomplete topology target extraction")
    values.flush()
    del values
    write_json(
        folder / "metadata.json", {"identity": identity, "sha256": file_digest(path)}
    )
    return load_tda_cache(folder, identity)


def load_tda_cache(folder: Path, identity: dict) -> np.ndarray:
    receipt = Path(folder) / "metadata.json"
    if not receipt.is_file():
        raise ValueError("TDA cache is incomplete; prepare into a new cache directory")
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    path = Path(folder) / "targets.npy"
    if metadata["identity"] != identity or metadata["sha256"] != file_digest(path):
        raise ValueError("TDA cache provenance or content checksum changed")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (identity["samples"], 12) or values.dtype != np.float32:
        raise ValueError("invalid topology target cache shape or dtype")
    return values
