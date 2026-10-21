from __future__ import annotations

import unittest

import numpy as np


class NeighborhoodTests(unittest.TestCase):
    def test_rips_square_has_literal_h0_and_h1_in_320_dimensions(self):
        from hccr.tda.embedding.neighborhood import rips_features

        points = np.zeros((4, 320))
        points[0, 0], points[1, 1] = 1, 1
        points[2, 0], points[3, 1] = -1, -1
        result = rips_features(points)
        np.testing.assert_allclose(result.diagrams[1], [[2**0.5, 2]])
        np.testing.assert_allclose(
            result.features[:7],
            [3, 3 * 2**0.5, 2**0.5, 0, 2**0.5, np.log(3), 1],
            atol=1e-14,
        )
        self.assertEqual(result.features[7], 1)
        self.assertEqual(result.features[13], 0)
        with self.assertRaisesRegex(ValueError, "320"):
            rips_features(np.zeros((320, 1)))

    def test_rips_identical_points_and_invalid_input(self):
        from hccr.tda.embedding.neighborhood import rips_features

        result = rips_features(np.zeros((4, 320)))
        np.testing.assert_array_equal(result.features, [0, 0, 0, 0, 0, 0, 1] + [0] * 7)
        with self.assertRaisesRegex(ValueError, "finite"):
            rips_features(np.full((4, 320), np.nan))

    def test_neighbors_exclude_id_not_identical_vector_and_keep_stable_ties(self):
        from hccr.tda.embedding.neighborhood import ReferenceBank

        vectors = np.zeros((4, 320))
        vectors[0, 0] = vectors[1, 0] = 1
        vectors[2, 0], vectors[3, 1] = -1, 1
        bank = ReferenceBank(
            vectors, np.array([0, 1, 1, 0]), np.array(["b", "a", "c", "d"])
        )
        result = bank.neighbors(vectors[:1], ["query"], k=2, reference_batch_size=1)
        np.testing.assert_array_equal(result.indices, [[1, 0]])
        np.testing.assert_array_equal(result.distances, [[0, 0]])
        excluded = bank.neighbors(vectors[:1], ["b"], k=2, reference_batch_size=4)
        np.testing.assert_array_equal(excluded.indices, [[1, 3]])
        np.testing.assert_allclose(excluded.distances, [[0, 2**0.5]])

    def test_tiled_search_matches_exhaustive_and_reference_labels_do_not_affect_it(
        self,
    ):
        from hccr.tda.embedding.neighborhood import ReferenceBank, normalize_rows

        rng = np.random.default_rng(7)
        points, queries = normalize_rows(rng.normal(size=(11, 320))), rng.normal(
            size=(5, 320)
        )
        ids = np.array([f"r-{i:02}" for i in range(11)])
        bank = ReferenceBank(points, np.arange(11) % 2, ids)
        expected = np.argsort(
            np.linalg.norm(normalize_rows(queries)[:, None] - points[None], axis=2),
            axis=1,
        )[:, :3]
        for chunk in (1, 4, 11):
            actual = bank.neighbors(
                queries,
                [f"q-{i}" for i in range(5)],
                k=3,
                reference_batch_size=chunk,
                query_batch_size=2,
            )
            np.testing.assert_array_equal(actual.indices, expected)
        altered = ReferenceBank(points, 1 - np.arange(11) % 2, ids)
        np.testing.assert_array_equal(
            altered.neighbors(queries, [f"q-{i}" for i in range(5)], 3).indices,
            expected,
        )

    def test_bank_selection_is_seeded_train_only_and_order_independent(self):
        from hccr.tda.embedding.neighborhood import select_reference_indices

        ids = np.array(["a", "b", "c", "d", "e", "f"])
        labels = np.array([0, 0, 0, 1, 1, 1])
        chosen = select_reference_indices(labels, ids, per_class=2, seed=7)
        np.testing.assert_array_equal(np.bincount(labels[chosen]), [2, 2])
        order = np.array([4, 2, 0, 5, 3, 1])
        permuted = select_reference_indices(
            labels[order], ids[order], per_class=2, seed=7
        )
        self.assertEqual(ids[chosen].tolist(), ids[order][permuted].tolist())
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            select_reference_indices(
                labels, ids, per_class=2, seed=7, split="validation"
            )

    def test_zero_embedding_and_insufficient_bank_fail(self):
        from hccr.tda.embedding.neighborhood import ReferenceBank, normalize_rows

        with self.assertRaisesRegex(ValueError, "nonzero"):
            normalize_rows(np.zeros((2, 320)))
        points = np.zeros((2, 320))
        points[:, 0] = 1
        bank = ReferenceBank(points, np.array([0, 1]), np.array(["a", "b"]))
        with self.assertRaisesRegex(ValueError, "neighbors"):
            bank.neighbors(points[:1], ["a"], k=2)

    def test_uniform_votes_and_probability_mixture_have_literal_values(self):
        from hccr.tda.embedding.neighborhood import mix_knn_scores

        logits = np.array([[0.0, 0.0]])
        scores = mix_knn_scores(logits, np.array([[0, 0, 1, 0]]), mix=0.5)
        np.testing.assert_allclose(scores, [[0.625, 0.375]])


if __name__ == "__main__":
    unittest.main()
