"""Validated, read-only 320D embedding-cache inputs and bounded row views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from hccr.tda.features import digest_file, signature

from .neighborhood import EMBEDDING_DIM, normalize_rows


def id_digest(sample_ids) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class IndexedRows:
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, indices
        self.shape = (len(indices), *rows.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.rows[self.indices[index]]


class EmbeddingView:
    def __init__(self, source, original_classes: tuple[int, ...]):
        lookup = np.full(len(source.class_mapping), -1, dtype=np.int64)
        lookup[list(original_classes)] = np.arange(len(original_classes))
        indices = []
        for start in range(0, len(source.targets), 4096):
            targets = source.targets[start : start + 4096]
            indices.append(np.flatnonzero(lookup[targets] >= 0) + start)
        self.indices = np.concatenate(indices)
        if not len(self.indices):
            raise ValueError(f"empty candidate sample plan for {source.split}")
        self.features = IndexedRows(source.features, self.indices)
        self.targets = lookup[source.targets[self.indices]]
        self.sample_ids = np.array([source.sample_ids[int(i)] for i in self.indices])
        self.split = source.split
        for start in range(0, len(self.indices), 4096):
            normalize_rows(self.features[start : start + 4096])
        self.identity = {
            "source": source.pin,
            "split": self.split,
            "ordered_sample_ids": id_digest(self.sample_ids),
            "samples": len(self.indices),
            "original_classes": list(original_classes),
        }


class CachedEmbeddings:
    """Open only the requested split and verify every consumed array checksum."""

    def __init__(self, root: Path, split: str, expected_shared: dict | None = None):
        self.split, self.path = split, root / split
        self.features = self.targets = None
        try:
            raw = (self.path / "metadata.json").read_bytes()
            metadata = json.loads(raw)
            identity = metadata["identity"]
            required = {
                "split",
                "sample_ids",
                "class_mapping",
                "checkpoint",
                "manifest",
                "preprocessing",
                "extraction_precision",
            }
            if not required.issubset(identity):
                raise ValueError("source cache lacks required provenance fields")
            if metadata.get("signature") != signature(identity):
                raise ValueError("source cache identity signature mismatch")
            if identity["split"] != split:
                raise ValueError("source cache split does not match requested split")
            self.shared_identity = {
                key: value
                for key, value in identity.items()
                if key not in {"split", "sample_ids"}
            }
            if expected_shared is not None and self.shared_identity != expected_shared:
                raise ValueError(
                    "source caches disagree on checkpoint, mapping or provenance"
                )
            self.class_mapping = identity["class_mapping"]
            if (
                not isinstance(self.class_mapping, dict)
                or not self.class_mapping
                or any(
                    not isinstance(label, str)
                    or not isinstance(target, int)
                    or isinstance(target, bool)
                    for label, target in self.class_mapping.items()
                )
                or set(self.class_mapping.values())
                != set(range(len(self.class_mapping)))
            ):
                raise ValueError(
                    "source class_mapping must have unique contiguous integer targets"
                )
            self.sample_ids = identity["sample_ids"]
            if (
                not isinstance(self.sample_ids, list)
                or not self.sample_ids
                or any(
                    not isinstance(value, str) or not value.strip() or "\0" in value
                    for value in self.sample_ids
                )
                or len(set(self.sample_ids)) != len(self.sample_ids)
            ):
                raise ValueError("source sample IDs must be nonempty and unique")
            self.pin = {
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                "source_signature": metadata["signature"],
                "arrays": {},
            }
            for name, attribute in (("cnn", "features"), ("targets", "targets")):
                specification = metadata["arrays"][name]
                target = self.path / f"{name}.npy"
                actual = digest_file(target)
                if actual != specification["sha256"]:
                    raise ValueError(f"source array checksum mismatch: {name}")
                array = np.load(target, mmap_mode="r", allow_pickle=False)
                setattr(self, attribute, array)
                if (
                    list(array.shape) != specification["shape"]
                    or str(array.dtype) != specification["dtype"]
                ):
                    raise ValueError(f"source array schema mismatch: {name}")
                self.pin["arrays"][name] = specification
            if (
                self.features.shape != (len(self.sample_ids), EMBEDDING_DIM)
                or self.features.dtype != np.float32
            ):
                raise ValueError(
                    "source cnn must be float32 [samples, 320] aligned with IDs"
                )
            if self.targets.shape != (len(self.sample_ids),) or not np.issubdtype(
                self.targets.dtype, np.integer
            ):
                raise ValueError("source targets must be an aligned integer vector")
            for start in range(0, len(self.targets), 4096):
                values = self.targets[start : start + 4096]
                if np.any(values < 0) or np.any(values >= len(self.class_mapping)):
                    raise ValueError("source target is outside class_mapping")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self):
        for array in (self.features, self.targets):
            if array is not None:
                array._mmap.close()
