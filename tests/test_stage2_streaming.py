from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hccr.tda.features import FittedBlocks, Standardizer


class BoundedRows:
    """Reject eager materialization and reads larger than the agreed batch."""

    def __init__(self, values, limit=2):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.limit = limit

    def __len__(self):
        return self.shape[0]

    def __array__(self, *args, **kwargs):
        raise AssertionError("entire feature matrix materialized")

    def __getitem__(self, index):
        result = self.values[index]
        if result.ndim == 2 and len(result) > self.limit:
            raise AssertionError("read exceeded the configured batch")
        return result


class StreamingTests(unittest.TestCase):
    def test_standardizer_reads_only_chunks_and_keeps_population_variance(self):
        rows = BoundedRows([[1, 4], [3, 4], [5, 4], [7, 4]])
        fitted = Standardizer.fit(rows, chunk_size=2)
        np.testing.assert_array_equal(fitted.mean, [4, 4])
        np.testing.assert_allclose(fitted.scale, [5**0.5, 1])

    def test_pca_and_lazy_fusion_do_not_materialize_training_matrix(self):
        from hccr.tda.features import FusedRows

        rows = BoundedRows([[0, 0], [1, 2], [2, 4], [3, 6]])
        fitted = FittedBlocks.fit(rows, rows, pca=True, chunk_size=2)
        self.assertEqual(fitted.state()["pca_rank"], 1)
        lazy = FusedRows(rows, rows, fitted, 0.5)
        self.assertEqual(lazy.shape, (4, 16))
        np.testing.assert_array_equal(
            lazy[[3, 0]], fitted.fuse(rows.values[[3, 0]], rows.values[[3, 0]], 0.5)
        )

    def test_streaming_pca_matches_dense_subspace_and_handles_zero_rank(self):
        training = np.random.default_rng(7).normal(size=(19, 5))
        blocks = FittedBlocks.fit(training, training, pca=True, chunk_size=3)
        centered = training - training.mean(axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        # Sign-invariant comparison against the dense reference solver.
        np.testing.assert_allclose(
            np.abs(blocks.pca_components @ components.T), np.eye(5), atol=1e-12
        )
        constant = np.ones((5, 3))
        zero_rank = FittedBlocks.fit(constant, constant, pca=True, chunk_size=2)
        self.assertEqual(zero_rank.state()["pca_rank"], 0)
        restored = FittedBlocks.from_state(zero_rank.state())
        np.testing.assert_array_equal(
            restored.auxiliary_values(np.full((2, 3), 8)), np.zeros((2, 14))
        )

    def test_streamed_cache_publishes_only_complete_rows(self):
        from hccr.tda.features import CacheWriter, load_cache

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete"
            with CacheWriter(path, {"fixture": 1}, 3) as cache:
                cache.append({"cnn": np.array([[1, 2], [3, 4]], dtype=np.float32)})
                self.assertFalse(path.exists())
                cache.append({"cnn": np.array([[5, 6]], dtype=np.float32)})
            arrays = load_cache(path, {"fixture": 1})
            self.assertIsInstance(arrays["cnn"], np.memmap)
            np.testing.assert_array_equal(arrays["cnn"], [[1, 2], [3, 4], [5, 6]])
            del arrays
            interrupted = Path(directory) / "interrupted"
            with (
                self.assertRaisesRegex(ValueError, "incomplete"),
                CacheWriter(interrupted, {}, 3) as cache,
            ):
                cache.append({"cnn": np.ones((1, 2), dtype=np.float32)})
            self.assertFalse(interrupted.exists())

    def test_cache_context_releases_maps_even_with_retained_views(self):
        from hccr.tda.features import load_cache, save_cache

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache"
            save_cache(path, {"cnn": np.ones((2, 2), dtype=np.float32)}, {})
            with (
                self.assertRaisesRegex(RuntimeError, "consumer failed"),
                load_cache(path, {}) as cache,
            ):
                retained = cache["cnn"]
                raise RuntimeError("consumer failed")
            self.assertTrue(retained._mmap.closed)

    def test_streaming_metrics_equal_eager_metrics_with_uneven_batches(self):
        from torch import nn

        from hccr.tda.experiment import classification_metrics, head_logits

        rows = BoundedRows([[3, 1], [0, 2], [0, 4], [5, 1], [0, 3]])
        targets = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        result = classification_metrics(
            head_logits(nn.Identity(), rows, "cpu", 2), targets
        )
        self.assertEqual(
            result,
            {
                "top1": 0.6,
                "samples": 5,
                "class_support": {"0": 2, "1": 3},
                "class_errors": {"0": 1, "1": 1},
            },
        )

    def test_lmdb_plan_packs_ids_without_retaining_manifest_row_dicts(self):
        import pickle

        from hccr.tda.experiment import CompactPlan

        original = [
            {"sample_id": "train-中", "unicode_label": "中", "unused": "discard"},
            {"sample_id": "train-b", "unicode_label": "一", "unused": "discard"},
        ]
        plan = CompactPlan(iter(original), {"一": 0, "中": 1})
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0], {"sample_id": "train-中", "unicode_label": "中"})
        restored = pickle.loads(pickle.dumps(plan))
        self.assertEqual(
            list(restored),
            [
                {"sample_id": "train-中", "unicode_label": "中"},
                {"sample_id": "train-b", "unicode_label": "一"},
            ],
        )
        self.assertIsInstance(plan.sample_ids, np.ndarray)
        self.assertIsInstance(plan.targets, np.ndarray)

    def test_head_training_reads_only_the_configured_feature_batch(self):
        import torch

        from hccr.tda.experiment import PilotConfig, train_common_head

        torch.set_num_threads(1)
        rows = BoundedRows([[1, 0], [0, 1], [2, 0], [0, 2]])
        targets = np.array([0, 1, 0, 1], dtype=np.int64)
        config = PilotConfig("unused", "unused", head_epochs=1, head_batch_size=2)
        head, evidence = train_common_head(
            rows, targets, rows, targets, 2, config, 32, 0.1, "cpu"
        )
        self.assertEqual(evidence["selected_epoch"], 1)
        self.assertEqual(evidence["parameters"], 4)
        self.assertFalse(head.training)

    def test_compact_lmdb_plan_reads_images_with_four_spawn_workers(self):
        import csv

        import torch

        from hccr.data import build_lmdb_image_store
        from hccr.tda.experiment import CompactPlan, PilotConfig, Stage2Study
        from hccr.tda.smoke import create_synthetic_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = create_synthetic_inputs(root)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            build_lmdb_image_store(
                manifest, rows, root / "data", manifest.with_name("images.lmdb")
            )
            study = Stage2Study(
                PilotConfig(
                    str(run),
                    str(manifest),
                    str(root / "study"),
                    max_classes=2,
                    dataset_backend="lmdb",
                    num_workers=4,
                    batch_size=3,
                    device="cpu",
                )
            )
            dataset = study.dataset("train")
            self.assertIsInstance(dataset.rows, CompactPlan)
            expected = [
                (row["sample_id"], row["unicode_label"])
                for row in rows
                if row["split"] == "train"
            ]
            self.assertEqual(
                [(row["sample_id"], row["unicode_label"]) for row in dataset.rows],
                expected,
            )
            batches = list(study.loader(dataset))
            self.assertEqual(
                torch.cat([batch[2] for batch in batches]).tolist(), list(range(16))
            )
            self.assertEqual(
                torch.cat([batch[1] for batch in batches]).tolist(), [0] * 8 + [1] * 8
            )
            dataset.store.close()

    def test_extraction_stores_restricted_predictions_instead_of_dense_logits(self):
        import torch

        from hccr.tda.experiment import PilotConfig, Stage2Study
        from hccr.tda.smoke import create_synthetic_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = create_synthetic_inputs(root)
            study = Stage2Study(
                PilotConfig(
                    str(run),
                    str(manifest),
                    str(root / "study"),
                    max_classes=2,
                    projection_epochs=1,
                    batch_size=4,
                    device="cpu",
                )
            )
            study.fit_projection()
            with study.extract("train") as cache:
                self.assertNotIn("original_logits", cache)
                self.assertEqual(cache["original_predictions"].shape, (16,))
                dataset = study.dataset("train")
                with torch.no_grad():
                    logits = study.adapter(
                        torch.stack([dataset[i][0] for i in range(16)])
                    )[2]
                expected = logits[:, study.original_ids].argmax(dim=1).numpy()
                np.testing.assert_array_equal(cache["original_predictions"], expected)


if __name__ == "__main__":
    unittest.main()
