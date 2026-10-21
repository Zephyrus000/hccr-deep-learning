"""Bounded spawn-safe Rips work and ordered extraction with one cache writer."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

from .accelerated import ExactMatrixSearch
from .neighborhood import distance_features, normalize_rows, rips_features

_WORKER_VECTORS = None


def _initialize_worker(path):
    global _WORKER_VECTORS
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_VECTORS = np.load(path, mmap_mode="r", allow_pickle=False)


def _compute_chunk(vectors, queries, indices, inspection):
    started = perf_counter()
    features, diagrams = [], ()
    for row, (query, neighbors) in enumerate(zip(queries, indices, strict=True)):
        result = rips_features(np.vstack([query, vectors[neighbors]]))
        features.append(result.features)
        if inspection and row == 0:
            diagrams = result.diagrams
    return np.stack(features), diagrams, perf_counter() - started


def _worker_chunk(task):
    return _compute_chunk(_WORKER_VECTORS, *task)


@dataclass(frozen=True)
class PHBatch:
    features: np.ndarray
    first_diagrams: tuple
    worker_seconds: float


class PHComputer:
    def __init__(self, vectors_path: Path, *, workers=0, chunk_size=16):
        if not isinstance(workers, int) or workers < 0 or chunk_size < 1:
            raise ValueError("invalid PH worker count or chunk size")
        self.path, self.workers, self.chunk_size = vectors_path, workers, chunk_size
        self.executor = self.vectors = None

    def __enter__(self):
        self.vectors = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self.workers:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(str(self.path),),
            )
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=exception is not None)
        finally:
            if self.vectors is not None:
                self.vectors._mmap.close()

    def submit(self, queries, indices):
        if len(queries) != len(indices) or not len(queries):
            raise ValueError("empty or misaligned PH query batch")
        if self.executor is None:
            return [_compute_chunk(self.vectors, queries, indices, True)]
        # At most two queued chunks per worker, independent of dataset size.
        chunk = max(self.chunk_size, ceil(len(queries) / (2 * self.workers)))
        return [
            self.executor.submit(
                _worker_chunk,
                (
                    queries[start : start + chunk],
                    indices[start : start + chunk],
                    start == 0,
                ),
            )
            for start in range(0, len(queries), chunk)
        ]

    def collect(self, jobs):
        results = [job.result() for job in jobs] if self.executor is not None else jobs
        return PHBatch(
            np.concatenate([item[0] for item in results]),
            results[0][1],
            sum(item[2] for item in results),
        )

    def compute(self, queries, indices):
        return self.collect(self.submit(queries, indices))


@dataclass(frozen=True)
class ExtractedBatch:
    arrays: dict[str, np.ndarray]
    inspection: dict[str, np.ndarray]


class NeighborhoodExtractor:
    def __init__(self, bank, vectors_path, config):
        self.bank, self.vectors_path, self.config = bank, vectors_path, config
        self.resources = ExitStack()
        self.previous_threads = None
        self.timings = {
            "search_seconds": 0.0,
            "ph_wait_seconds": 0.0,
            "ph_worker_wall_seconds": 0.0,
        }

    def __enter__(self):
        try:
            if self.config.search_backend == "cpu_exact":
                import torch

                self.previous_threads = torch.get_num_threads()
                torch.set_num_threads(self.config.search_threads)
            self.search = (
                self.bank
                if self.config.search_backend == "reference"
                else self.resources.enter_context(
                    ExactMatrixSearch(
                        self.bank,
                        device=(
                            "cuda"
                            if self.config.search_backend == "cuda_exact"
                            else "cpu"
                        ),
                    )
                )
            )
            self.ph = self.resources.enter_context(
                PHComputer(
                    self.vectors_path,
                    workers=self.config.ph_workers,
                    chunk_size=self.config.ph_chunk_size,
                )
            )
            return self
        except BaseException:
            self.resources.close()
            self._restore_threads()
            raise

    def __exit__(self, exception_type, exception, traceback):
        try:
            return self.resources.__exit__(exception_type, exception, traceback)
        finally:
            self._restore_threads()

    def _restore_threads(self):
        if self.previous_threads is not None:
            import torch

            torch.set_num_threads(self.previous_threads)
            self.previous_threads = None

    def _finish(self, view, pending):
        start, raw, ids, queries, nearest, jobs = pending
        started = perf_counter()
        result = self.ph.collect(jobs)
        self.timings["ph_wait_seconds"] += perf_counter() - started
        self.timings["ph_worker_wall_seconds"] += result.worker_seconds
        end = start + len(raw)
        inspection = (
            {
                "inspection_cloud": np.vstack(
                    [queries[0], self.bank.vectors[nearest.indices[0]]]
                ),
                "inspection_h0": result.first_diagrams[0],
                "inspection_h1": result.first_diagrams[1],
            }
            if start == 0
            else {}
        )
        return ExtractedBatch(
            {
                "cnn": raw.astype(np.float32),
                "targets": view.targets[start:end],
                "sample_ids": ids,
                "source_indices": view.indices[start:end],
                "neighbors": nearest.indices.astype(np.int32),
                "distance": distance_features(nearest.distances),
                "ph": result.features.astype(np.float32),
            },
            inspection,
        )

    def batches(self, view):
        pending = None
        for start in range(0, len(view.targets), self.config.query_batch_size):
            end = start + self.config.query_batch_size
            raw, ids = view.features[start:end], view.sample_ids[start:end]
            started = perf_counter()
            nearest = self.search.neighbors(
                raw,
                ids,
                self.config.neighbors,
                reference_batch_size=self.config.reference_batch_size,
                query_batch_size=self.config.query_batch_size,
            )
            self.timings["search_seconds"] += perf_counter() - started
            queries = normalize_rows(raw)
            completed = self._finish(view, pending) if pending is not None else None
            pending = (
                start,
                raw,
                ids,
                queries,
                nearest,
                self.ph.submit(queries, nearest.indices),
            )
            if completed is not None:
                yield completed
        if pending is not None:
            yield self._finish(view, pending)
