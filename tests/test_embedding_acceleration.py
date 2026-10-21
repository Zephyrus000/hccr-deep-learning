from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hccr.tda.embedding.neighborhood import (
    ReferenceBank,
    normalize_rows,
    rips_features,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA exact-search parity")
class CudaSearchTests(unittest.TestCase):
    def test_small_bank_does_not_reintroduce_masked_self_in_rerank(self):
        from hccr.tda.embedding.accelerated import ExactMatrixSearch

        vectors = np.eye(3, 320)
        bank = ReferenceBank(vectors, np.array([0, 1, 2]), np.array(["a", "b", "c"]))
        actual = ExactMatrixSearch(bank, device="cuda").neighbors(vectors[:1], ["a"], 2)
        np.testing.assert_array_equal(actual.indices, [[1, 2]])

    def test_gpu_exhaustive_neighbors_match_fixed_order_float64_oracle(self):
        from hccr.tda.embedding.accelerated import ExactMatrixSearch

        rng = np.random.default_rng(17)
        vectors = normalize_rows(rng.normal(size=(137, 320)))
        ids = np.array([f"r-{i:04}" for i in range(len(vectors))])
        bank = ReferenceBank(vectors, np.arange(len(vectors)) % 3, ids)
        queries = np.vstack([vectors[:4], rng.normal(size=(7, 320))])
        query_ids = list(ids[:4]) + [f"q-{i}" for i in range(7)]
        expected = bank.neighbors(queries, query_ids, 8, reference_batch_size=13)
        search = ExactMatrixSearch(bank, device="cuda")
        for batch in (1, 4, 11):
            actual = search.neighbors(queries, query_ids, 8, query_batch_size=batch)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            np.testing.assert_array_equal(actual.distances, expected.distances)
        forced_fallback = ExactMatrixSearch(bank, device="cuda", candidate_cap=1)
        actual = forced_fallback.neighbors(queries, query_ids, 8)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.distances, expected.distances)

    def test_gpu_keeps_all_boundary_ties_and_excludes_only_self_id(self):
        from hccr.tda.embedding.accelerated import ExactMatrixSearch

        vectors = np.zeros((128, 320))
        vectors[:, 0] = 1
        ids = np.array([f"r-{i:04}" for i in range(127, -1, -1)])
        bank = ReferenceBank(vectors, np.arange(128) % 2, ids)
        expected = bank.neighbors(vectors[:2], list(ids[:2]), 4)
        actual = ExactMatrixSearch(bank, device="cuda").neighbors(
            vectors[:2], list(ids[:2]), 4
        )
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.distances, [[0] * 4] * 2)

    def test_gpu_near_identical_and_near_boundary_points_match(self):
        from hccr.tda.embedding.accelerated import ExactMatrixSearch

        rng = np.random.default_rng(29)
        vectors = normalize_rows(np.ones((101, 320)) + rng.normal(0, 1e-8, (101, 320)))
        bank = ReferenceBank(
            vectors, np.arange(101) % 2, np.array([f"r-{i:04}" for i in range(101)])
        )
        queries = vectors[:3].copy()
        ids = ["q-a", "q-b", "r-0002"]
        expected = bank.neighbors(queries, ids, 8)
        actual = ExactMatrixSearch(bank, device="cuda").neighbors(queries, ids, 8)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.distances, expected.distances)


class ParallelPHTests(unittest.TestCase):
    def test_cpu_blas_search_matches_oracle(self):
        from hccr.tda.embedding.accelerated import ExactMatrixSearch

        rng = np.random.default_rng(17)
        references = normalize_rows(rng.normal(size=(101, 320)))
        bank = ReferenceBank(
            references, np.arange(101) % 3, np.array([f"r-{i:04}" for i in range(101)])
        )
        queries = rng.normal(size=(5, 320))
        ids = [f"q-{i}" for i in range(5)]
        expected = bank.neighbors(queries, ids, 8)
        with ExactMatrixSearch(bank, device="cpu") as search:
            actual = search.neighbors(queries, ids, 8, query_batch_size=3)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.distances, expected.distances)

    def test_spawned_workers_preserve_feature_order_and_first_diagrams(self):
        from hccr.tda.embedding.parallel import PHComputer

        rng = np.random.default_rng(7)
        references = normalize_rows(rng.normal(size=(15, 320)))
        queries = normalize_rows(rng.normal(size=(5, 320)))
        indices = np.array([[i, i + 1, i + 2] for i in range(5)])
        expected = [
            rips_features(np.vstack([query, references[neighbors]]))
            for query, neighbors in zip(queries, indices, strict=True)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npy"
            np.save(path, references)
            for workers in (0, 2):
                with PHComputer(path, workers=workers, chunk_size=2) as computer:
                    actual = computer.compute(queries, indices)
                np.testing.assert_array_equal(
                    actual.features, np.stack([item.features for item in expected])
                )
                for actual_diagram, expected_diagram in zip(
                    actual.first_diagrams, expected[0].diagrams, strict=True
                ):
                    np.testing.assert_array_equal(actual_diagram, expected_diagram)

    def test_worker_errors_propagate(self):
        from hccr.tda.embedding.parallel import PHComputer

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npy"
            np.save(path, np.eye(4, 320))
            with (
                PHComputer(path, workers=2) as computer,
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                computer.compute(np.full((1, 320), np.nan), np.array([[0, 1, 2]]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA pipeline parity")
    def test_pipelined_cuda_workers_match_serial_arrays_across_batch_sizes(self):
        from hccr.tda.embedding.parallel import NeighborhoodExtractor
        from hccr.tda.embedding.study import EmbeddingConfig

        rng = np.random.default_rng(7)
        vectors = normalize_rows(rng.normal(size=(21, 320)))
        bank = ReferenceBank(
            vectors, np.arange(21) % 3, np.array([f"r-{i}" for i in range(21)])
        )
        view = SimpleNamespace(
            features=rng.normal(size=(7, 320)).astype(np.float32),
            sample_ids=np.array([f"q-{i}" for i in range(7)]),
            targets=np.arange(7) % 3,
            indices=np.arange(7),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npy"
            np.save(path, vectors)
            outputs = []
            for backend, workers, size in (("reference", 0, 2), ("cuda_exact", 2, 3)):
                config = EmbeddingConfig(
                    source_root="unused",
                    max_classes=3,
                    neighbors=3,
                    search_backend=backend,
                    ph_workers=workers,
                    query_batch_size=size,
                )
                with NeighborhoodExtractor(bank, path, config) as extractor:
                    batches = list(extractor.batches(view))
                outputs.append(
                    {
                        name: np.concatenate([batch.arrays[name] for batch in batches])
                        for name in batches[0].arrays
                    }
                )
            for name in outputs[0]:
                np.testing.assert_array_equal(outputs[0][name], outputs[1][name])


if __name__ == "__main__":
    unittest.main()
