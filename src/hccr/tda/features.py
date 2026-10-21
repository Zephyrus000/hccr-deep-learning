"""Train-fitted feature blocks and immutable, content-checked array caches."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np

from hccr.tda.cubical import FILTRATION, PH_FEATURE_NAMES, STATISTIC_NAMES

VARIANTS = ("original", "cnn", "gap", "flat", "statistics", "h0", "h0_h1", "mean_h0_h1")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def implementation_identity() -> dict:
    package_root = Path(__file__).parents[1]
    sources = [*Path(__file__).parent.glob("*.py")]
    sources.extend(
        package_root / relative
        for relative in (
            "models/efficient_hccr.py",
            "models/heads.py",
            "models/artifacts.py",
            "data/dataset.py",
            "data/gnt.py",
            "data/lmdb_store.py",
            "preprocessing/pipeline.py",
            "training/precision.py",
        )
    )
    return {
        "source": {
            path.relative_to(package_root).as_posix(): digest_file(path)
            for path in sorted(sources)
        },
        "versions": {
            package: version(package)
            for package in ("torch", "torchvision", "numpy", "gudhi", "Pillow", "lmdb")
        },
        "filtration": FILTRATION,
        "ph_features": list(PH_FEATURE_NAMES),
        "statistics": list(STATISTIC_NAMES),
        "quantile_interpolation": "linear",
        "feature_schema": 2,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".json-", dir=path.parent) as temporary:
        staged = Path(temporary) / "value.json"
        staged.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, path)


def publish_cache(staged: Path, path: Path) -> None:
    """Retry Windows sharing denials without publishing incomplete data."""
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0)
    for attempt in range(len(delays) + 1):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite feature cache: {path}")
        try:
            staged.rename(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == len(
                delays
            ):
                raise
            logging.getLogger(__name__).warning(
                "Cache publication denied by Windows; retry %d/%d: %s",
                attempt + 1,
                len(delays),
                path,
            )
            time.sleep(delays[attempt])


def save_cache(path: Path, arrays: dict[str, np.ndarray], identity: dict) -> None:
    """Publish once. An interrupted staging directory is never a valid cache."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
        staged = Path(temporary) / "cache"
        staged.mkdir()
        inventory = {}
        for name, values in arrays.items():
            np.save(staged / f"{name}.npy", values, allow_pickle=False)
            inventory[name] = {
                "sha256": digest_file(staged / f"{name}.npy"),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        write_json(
            staged / "metadata.json",
            {
                "identity": identity,
                "signature": signature(identity),
                "arrays": inventory,
            },
        )
        publish_cache(staged, path)


class FeatureCache(dict):
    """Own memory-map handles for the duration of a comparison or evaluation."""

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self):
        for array in self.values():
            array._mmap.close()


def load_cache(path: Path, expected: dict) -> FeatureCache:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"incomplete cache {path}; re-extract into a new cache directory"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata["identity"]
    if actual != expected or metadata["signature"] != signature(expected):
        changed = sorted(
            key
            for key in actual.keys() | expected.keys()
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            f"incompatible cache {path}: {changed}; "
            "re-extract with the selected projection/config"
        )
    arrays = FeatureCache()
    try:
        for name, specification in metadata["arrays"].items():
            target = path / f"{name}.npy"
            if not target.is_file() or digest_file(target) != specification["sha256"]:
                raise ValueError(f"corrupt cache array {target}; re-extract")
            arrays[name] = np.load(target, mmap_mode="r", allow_pickle=False)
    except BaseException:
        arrays.close()
        raise
    return arrays


class CacheWriter:
    """Write sample batches directly to staged .npy memory maps, then publish."""

    def __init__(self, path: Path, identity: dict, samples: int) -> None:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite feature cache: {path}")
        if samples < 1:
            raise ValueError("feature cache requires at least one sample")
        self.path, self.identity, self.samples = path, identity, samples
        self.offset = 0
        self.arrays = {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(
            mkdtemp(prefix=f".{self.path.name}-", dir=self.path.parent)
        )
        self.staged = self.temporary / "cache"
        self.staged.mkdir()
        return self

    def append(self, batch: dict[str, np.ndarray]) -> None:
        size = len(next(iter(batch.values())))
        if size < 1 or self.offset + size > self.samples:
            raise ValueError("invalid feature cache batch length")
        if self.arrays and batch.keys() != self.arrays.keys():
            raise ValueError("feature cache batch keys changed")
        for name, values in batch.items():
            if len(values) != size or values.dtype.hasobject:
                raise ValueError("incompatible feature cache batch")
            if name not in self.arrays:
                self.arrays[name] = np.lib.format.open_memmap(
                    self.staged / f"{name}.npy",
                    mode="w+",
                    dtype=values.dtype,
                    shape=(self.samples, *values.shape[1:]),
                )
            target = self.arrays[name]
            if target.shape[1:] != values.shape[1:] or target.dtype != values.dtype:
                raise ValueError("feature cache batch shape or dtype changed")
            target[self.offset : self.offset + size] = values
        self.offset += size

    def add_inspection(self, arrays: dict[str, np.ndarray]) -> None:
        for name, values in arrays.items():
            if name in self.arrays or not name.startswith("inspection_"):
                raise ValueError("inspection arrays must not replace sample arrays")
            np.save(self.staged / f"{name}.npy", values, allow_pickle=False)

    def __exit__(self, exception_type, exception, traceback):
        keep_complete_staging = False
        try:
            for array in self.arrays.values():
                array.flush()
                array._mmap.close()
            self.arrays.clear()
            if exception_type is None:
                if self.offset != self.samples:
                    raise ValueError("incomplete feature cache sample count")
                inventory = {}
                for source in sorted(self.staged.glob("*.npy")):
                    array = np.load(source, mmap_mode="r", allow_pickle=False)
                    inventory[source.stem] = {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": digest_file(source),
                    }
                    array._mmap.close()
                write_json(
                    self.staged / "metadata.json",
                    {
                        "identity": self.identity,
                        "signature": signature(self.identity),
                        "arrays": inventory,
                    },
                )
                keep_complete_staging = True
                try:
                    publish_cache(self.staged, self.path)
                except OSError as error:
                    error.add_note(
                        f"Complete cache retained for recovery at {self.staged}"
                    )
                    raise
                keep_complete_staging = False
        finally:
            if not keep_complete_staging:
                shutil.rmtree(self.temporary)


def training_moments(rows, chunk_size: int, *, covariance: bool = False):
    """Merge float64 batch moments without copying the whole training matrix."""
    if len(rows) < 1 or chunk_size < 1:
        raise ValueError("nonempty training rows and positive chunk_size required")
    width = rows.shape[1]
    mean = np.zeros(width, dtype=np.float64)
    m2 = np.zeros((width, width) if covariance else width, dtype=np.float64)
    count = 0
    for start in range(0, len(rows), chunk_size):
        batch = np.asarray(rows[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("nonfinite training feature values")
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = centered.T @ centered if covariance else np.sum(centered**2, axis=0)
        delta, total = batch_mean - mean, count + len(batch)
        correction = np.outer(delta, delta) if covariance else delta**2
        m2 += batch_m2 + correction * (count * len(batch) / total)
        mean += delta * (len(batch) / total)
        count = total
    return mean, m2 / count


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray, *, chunk_size: int = 4096) -> Standardizer:
        mean, variance = training_moments(training, chunk_size)
        scale = np.sqrt(np.maximum(variance, 0))
        return cls(mean, np.where(scale < 1e-8, 1.0, scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "variance": "population",
            "scale_floor": 1e-8,
        }


@dataclass(frozen=True)
class FittedBlocks:
    cnn: Standardizer
    auxiliary: Standardizer
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None

    @classmethod
    def fit(
        cls,
        cnn: np.ndarray,
        auxiliary: np.ndarray,
        *,
        pca: bool = False,
        chunk_size: int = 4096,
    ) -> FittedBlocks:
        mean = components = None
        if pca:
            mean, covariance = training_moments(auxiliary, chunk_size, covariance=True)
            eigenvalues, vectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
            tolerance = (
                max(covariance.shape)
                * np.finfo(np.float64).eps
                * max(float(eigenvalues[0]), 0.0)
            )
            rank = min(14, len(auxiliary) - 1, int(np.sum(eigenvalues > tolerance)))
            components = vectors[:, :rank].T.copy()
            # Fix each eigenvector's sign so saved coordinates have a stable convention.
            for component in components:
                if component[np.argmax(np.abs(component))] < 0:
                    component *= -1
            auxiliary = ProjectedRows(auxiliary, mean, components)
        return cls(
            Standardizer.fit(cnn, chunk_size=chunk_size),
            Standardizer.fit(auxiliary, chunk_size=chunk_size),
            mean,
            components,
        )

    def auxiliary_values(self, values: np.ndarray) -> np.ndarray:
        if self.pca_mean is not None:
            values = (values - self.pca_mean) @ self.pca_components.T
            values = np.pad(values, ((0, 0), (0, 14 - values.shape[1])))
        return values

    def fuse(self, cnn: np.ndarray, auxiliary: np.ndarray, alpha: float) -> np.ndarray:
        return np.concatenate(
            [
                self.cnn.transform(cnn),
                alpha * self.auxiliary.transform(self.auxiliary_values(auxiliary)),
            ],
            axis=1,
        )

    def state(self) -> dict:
        return {
            "cnn": self.cnn.state(),
            "auxiliary": self.auxiliary.state(),
            "pca_mean": self.pca_mean.tolist() if self.pca_mean is not None else None,
            "pca_components": (
                self.pca_components.tolist()
                if self.pca_components is not None
                else None
            ),
            "pca_rank": (
                len(self.pca_components) if self.pca_components is not None else None
            ),
            "pca_solver": (
                "train_float64_streaming_covariance_eigh"
                if self.pca_mean is not None
                else None
            ),
        }

    @classmethod
    def from_state(cls, state: dict) -> FittedBlocks:
        return cls(
            Standardizer(
                np.array(state["cnn"]["mean"]), np.array(state["cnn"]["scale"])
            ),
            Standardizer(
                np.array(state["auxiliary"]["mean"]),
                np.array(state["auxiliary"]["scale"]),
            ),
            np.array(state["pca_mean"]) if state["pca_mean"] is not None else None,
            (
                np.array(state["pca_components"]).reshape(state["pca_rank"], -1)
                if state["pca_rank"]
                else (
                    np.empty((0, len(state["pca_mean"])))
                    if state["pca_mean"] is not None
                    else None
                )
            ),
        )


class ProjectedRows:
    def __init__(self, rows, mean, components):
        self.rows, self.mean, self.components = rows, mean, components
        self.shape = (len(rows), 14)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        values = (self.rows[index] - self.mean) @ self.components.T
        return np.pad(values, ((0, 0), (0, 14 - values.shape[1])))


class AuxiliaryRows:
    def __init__(self, cache, variant):
        self.rows = cache["h0_h1"] if variant == "h0" else cache["cnn"]
        self.variant = variant
        self.shape = (len(self.rows), 14)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if self.variant == "cnn":
            size = (
                len(range(*index.indices(len(self))))
                if isinstance(index, slice)
                else len(index)
            )
            return np.zeros((size, 14), dtype=np.float32)
        return np.pad(self.rows[index][:, :7], ((0, 0), (0, 7)))


class FusedRows:
    def __init__(self, cnn, auxiliary, blocks: FittedBlocks, alpha: float):
        self.cnn, self.auxiliary, self.blocks, self.alpha = (
            cnn,
            auxiliary,
            blocks,
            alpha,
        )
        self.shape = (len(cnn), cnn.shape[1] + 14)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return self.blocks.fuse(self.cnn[index], self.auxiliary[index], self.alpha)


def auxiliary_block(cache: dict[str, np.ndarray], variant: str):
    if variant == "cnn":
        return AuxiliaryRows(cache, variant)
    if variant == "h0":
        return AuxiliaryRows(cache, variant)
    return cache[variant]
