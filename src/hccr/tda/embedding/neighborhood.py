"""TRAIN reference selection, exact tiled retrieval and full Euclidean Rips PH."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass

import numpy as np

from hccr.tda.cubical import distribution_statistics, summarize_intervals

EMBEDDING_DIM = 320
MAX_NEIGHBORS = 64
RIPS_PROTOCOL = "euclidean_rips_complete_2_skeleton_h0_h1_f2_positive_lifetimes"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
        raise ValueError("embeddings must have shape [samples, 320]")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must be finite")
    scale = np.max(np.abs(array), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("L2 geometry requires nonzero embeddings")
    scaled = array / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def select_reference_indices(
    targets: np.ndarray,
    sample_ids: np.ndarray,
    *,
    per_class: int,
    seed: int,
    split: str = "train",
) -> np.ndarray:
    """Keep up to the quota's smallest seeded ID hashes per TRAIN class."""
    if split != "train":
        raise ValueError("reference bank must be fitted on TRAIN only")
    if per_class < 1 or len(targets) != len(sample_ids) or not len(targets):
        raise ValueError("invalid TRAIN bank quota or sample plan")
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise ValueError("TRAIN bank sample IDs must be unique")
    ranks = np.empty(len(sample_ids), dtype=np.int64)
    ranks[np.argsort(sample_ids, kind="stable")] = np.arange(len(sample_ids))
    heaps = {}
    for index, (target, sample_id) in enumerate(zip(targets, sample_ids, strict=True)):
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest(), "big"
        )
        entry = (-priority, -int(ranks[index]), index)
        heap = heaps.setdefault(int(target), [])
        if len(heap) < per_class:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    chosen = [entry[2] for heap in heaps.values() for entry in heap]
    return np.array(sorted(chosen, key=lambda i: sample_ids[i]), dtype=np.int64)


@dataclass(frozen=True)
class NeighborBatch:
    indices: np.ndarray
    distances: np.ndarray


class ReferenceBank:
    """Read-only normalized TRAIN references; labels never determine geometry."""

    def __init__(self, vectors, targets, sample_ids):
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError("reference vectors must have width 320")
        if len(vectors) != len(targets) or len(vectors) != len(sample_ids):
            raise ValueError("reference vector, label and ID counts differ")
        if len(set(map(str, sample_ids))) != len(sample_ids):
            raise ValueError("reference sample IDs must be unique")
        for start in range(0, len(vectors), 4096):
            batch = np.asarray(vectors[start : start + 4096], dtype=np.float64)
            if not np.isfinite(batch).all() or not np.allclose(
                np.linalg.norm(batch, axis=1), 1, rtol=0, atol=1e-12
            ):
                raise ValueError("reference vectors must be finite and L2-normalized")
        self.vectors, self.targets, self.sample_ids = vectors, targets, sample_ids
        self.ranks = np.empty(len(vectors), dtype=np.int64)
        self.ranks[np.argsort(sample_ids, kind="stable")] = np.arange(len(vectors))

    def neighbors(
        self,
        queries,
        query_ids,
        k: int,
        *,
        reference_batch_size: int = 2048,
        query_batch_size: int = 32,
    ) -> NeighborBatch:
        if not 1 <= k < len(self.vectors):
            raise ValueError("neighbors require k+1 or more reference vectors")
        if (
            len(queries) != len(query_ids)
            or min(reference_batch_size, query_batch_size) < 1
        ):
            raise ValueError("invalid query IDs or search batch sizes")
        result = np.empty((len(queries), k), dtype=np.int64)
        distances = np.empty((len(queries), k), dtype=np.float64)
        for start in range(0, len(queries), query_batch_size):
            end = min(start + query_batch_size, len(queries))
            query = normalize_rows(queries[start:end])
            ids = np.asarray(query_ids[start:end])
            best_distances = np.empty((len(query), 0), dtype=np.float64)
            best_indices = np.empty((len(query), 0), dtype=np.int64)
            for reference_start in range(0, len(self.vectors), reference_batch_size):
                reference_end = reference_start + reference_batch_size
                reference = np.asarray(
                    self.vectors[reference_start:reference_end], dtype=np.float64
                )
                # Fixed-order float64 dot products do not depend on tile sizes.
                dot = np.einsum("id,jd->ij", query, reference, optimize=False)
                squared = np.maximum(
                    np.sum(query**2, axis=1)[:, None]
                    + np.sum(reference**2, axis=1)[None, :]
                    - 2 * dot,
                    0,
                )
                same_id = (
                    ids[:, None]
                    == self.sample_ids[reference_start:reference_end][None, :]
                )
                squared[same_id] = np.inf
                candidate_indices = np.broadcast_to(
                    np.arange(reference_start, reference_start + len(reference)),
                    squared.shape,
                )
                candidates = np.concatenate([best_indices, candidate_indices], axis=1)
                candidate_distances = np.concatenate([best_distances, squared], axis=1)
                order = np.lexsort(
                    (self.ranks[candidates], candidate_distances), axis=1
                )[:, :k]
                best_indices = np.take_along_axis(candidates, order, axis=1)
                best_distances = np.take_along_axis(candidate_distances, order, axis=1)
            if not np.isfinite(best_distances).all():
                raise ValueError("not enough neighbors after sample-ID exclusion")
            result[start:end] = best_indices
            for row, indices in enumerate(best_indices):
                distances[start + row] = np.linalg.norm(
                    self.vectors[indices] - query[row], axis=1
                )
        return NeighborBatch(result, distances)


@dataclass(frozen=True)
class RipsResult:
    features: np.ndarray
    diagrams: tuple[np.ndarray, np.ndarray]


def rips_features(points: np.ndarray) -> RipsResult:
    cloud = np.asarray(points, dtype=np.float64)
    if (
        cloud.ndim != 2
        or cloud.shape[1] != EMBEDDING_DIM
        or not 3 <= len(cloud) <= MAX_NEIGHBORS + 1
    ):
        raise ValueError("Rips PH needs 3 to 65 points in 320 dimensions")
    if not np.isfinite(cloud).all():
        raise ValueError("Rips point cloud must be finite")
    try:
        import gudhi
    except ImportError as error:
        raise RuntimeError(
            'Install the PH extra with pip install -e ".[tda]"'
        ) from error
    complex_ = gudhi.RipsComplex(points=cloud).create_simplex_tree(max_dimension=2)
    complex_.compute_persistence(
        homology_coeff_field=2, min_persistence=0, persistence_dim_max=False
    )
    diagrams = tuple(complex_.persistence_intervals_in_dimension(d) for d in (0, 1))
    features = np.concatenate([summarize_intervals(diagram) for diagram in diagrams])
    if features[6] != 1 or features[13] != 0 or not np.isfinite(features).all():
        raise RuntimeError(
            "invalid persistence for complete finite Euclidean Rips cloud"
        )
    return RipsResult(features, diagrams)


def distance_features(distances: np.ndarray) -> np.ndarray:
    return np.stack(
        [distribution_statistics(row[None, :]) for row in distances]
    ).astype(np.float32)


def mix_knn_scores(
    logits: np.ndarray, neighbor_targets: np.ndarray, mix: float
) -> np.ndarray:
    if not 0 <= mix <= 1 or logits.ndim != 2 or len(logits) != len(neighbor_targets):
        raise ValueError("invalid kNN mixture")
    classes = logits.shape[1]
    if (
        neighbor_targets.ndim != 2
        or not neighbor_targets.shape[1]
        or np.any(neighbor_targets < 0)
        or np.any(neighbor_targets >= classes)
    ):
        raise ValueError("invalid kNN reference targets")
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    votes = np.zeros_like(probabilities)
    np.add.at(
        votes,
        (np.arange(len(votes))[:, None], neighbor_targets),
        1 / neighbor_targets.shape[1],
    )
    return (1 - mix) * probabilities + mix * votes
