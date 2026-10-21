from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from hccr.tda.embedding.smoke import create_synthetic_caches
from hccr.tda.embedding.source import CachedEmbeddings, EmbeddingView
from hccr.tda.embedding.study import (
    CONTROLS,
    EmbeddingConfig,
    EmbeddingStudy,
    load_config,
)
from hccr.tda.features import digest_file, signature, write_json


class EmbeddingStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        create_synthetic_caches(self.source)
        self.config = EmbeddingConfig(
            source_root=str(self.source),
            output_dir=str(self.root / "study"),
            max_classes=3,
            bank_per_class=4,
            neighbors=3,
            query_batch_size=4,
            reference_batch_size=5,
            head_epochs=1,
            head_batch_size=8,
            alpha_grid=(0.5, 1.0),
            knn_mix_grid=(0.0, 0.5, 1.0),
            device="cpu",
        )

    def test_development_never_opens_or_stats_holdouts_and_evaluation_is_gated(self):
        original_open, original_stat = Path.open, Path.stat

        def forbidden(path):
            return any(path.is_relative_to(self.source / s) for s in ("test", "icdar"))

        def checked_open(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError("holdout opened before explicit evaluation")
            return original_open(path, *args, **kwargs)

        def checked_stat(path, *args, **kwargs):
            if forbidden(path):
                raise AssertionError("holdout stat before explicit evaluation")
            return original_stat(path, *args, **kwargs)

        with (
            patch.object(Path, "open", checked_open),
            patch.object(Path, "stat", checked_stat),
            EmbeddingStudy(self.config) as study,
        ):
            with self.assertRaisesRegex(ValueError, "four comparisons"):
                study.evaluate()
            with self.assertRaisesRegex(ValueError, "development splits"):
                study.extract("test")
            selection = study.compare()
            self.assertEqual(selection["bank"]["class_support"], [4, 4, 4])
            self.assertEqual(set(selection["variants"]), set(CONTROLS))
            self.assertEqual(selection["search_budget"]["head_trials"], 5)
        with EmbeddingStudy(self.config) as study:
            report = study.evaluate(("test", "icdar"))
        self.assertTrue(report["selection_unchanged"])
        self.assertIsNone(report["writer_disjoint_verified"])
        for split in ("test", "icdar"):
            for control in CONTROLS:
                self.assertEqual(report["held_out"][split][control]["samples"], 9)
        cnn, knn = (selection["variants"][name] for name in ("cnn", "cnn_knn"))
        self.assertEqual(cnn["head"], knn["head"])
        self.assertEqual(cnn["head_sha256"], knn["head_sha256"])
        self.assertEqual(cnn["parameters"], 334 * 3)

    def test_cached_cnn_is_raw_and_fitted_statistics_use_train_only(self):
        source_hash = digest_file(self.source / "train" / "cnn.npy")
        with EmbeddingStudy(self.config) as study:
            selection = study.compare()
            with study.extract("train") as cache:
                np.testing.assert_array_equal(cache["cnn"], study.train.features[:])
                expected_mean = study.train.features[:].mean(axis=0, dtype=np.float64)
                for variant in ("cnn", "distance", "ph"):
                    actual = selection["variants"][variant]["transforms"]["cnn"]["mean"]
                    np.testing.assert_allclose(actual, expected_mean, atol=1e-15)
                self.assertEqual(cache["ph"].shape, (24, 14))
                self.assertEqual(cache["distance"].shape, (24, 14))
                for sample, neighbors in zip(
                    cache["sample_ids"], cache["neighbors"], strict=True
                ):
                    self.assertNotIn(sample, study.prepare().sample_ids[neighbors])
        self.assertEqual(source_hash, digest_file(self.source / "train" / "cnn.npy"))

    def test_selection_and_array_changes_fail_before_holdout_access(self):
        with EmbeddingStudy(self.config) as study:
            study.compare()
            path = study.output / "selection.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            changed = json.loads(json.dumps(original))
            changed["variants"]["ph"]["alpha"] = 42
            write_json(path, changed)
            with (
                patch.object(study, "_holdout", side_effect=AssertionError("leak")),
                self.assertRaisesRegex(ValueError, "selection"),
            ):
                study.evaluate()
            write_json(path, original)
            array_path = study.output / "features" / "validation" / "ph.npy"
            values = np.load(array_path)
            values[0, 0] += 1
            np.save(array_path, values)
            with (
                patch.object(study, "_holdout", side_effect=AssertionError("leak")),
                self.assertRaisesRegex(ValueError, "corrupt cache"),
            ):
                study.evaluate()

    def test_selected_head_and_bank_corruption_fail_before_holdout_access(self):
        with EmbeddingStudy(self.config) as study:
            selected = study.compare()
            head_path = study.output / selected["variants"]["cnn"]["head"]
            original = head_path.read_bytes()
            head_path.write_bytes(b"corrupt")
            with (
                patch.object(study, "_holdout", side_effect=AssertionError("leak")),
                self.assertRaisesRegex(ValueError, "head changed"),
            ):
                study.evaluate()
            head_path.write_bytes(original)
        bank_path = self.root / "study" / "seed-7" / "bank" / "vectors.npy"
        values = np.load(bank_path)
        values[0, 0] += 1
        np.save(bank_path, values)
        with (
            EmbeddingStudy(self.config) as study,
            patch.object(study, "_holdout", side_effect=AssertionError("leak")),
            self.assertRaisesRegex(ValueError, "corrupt cache"),
        ):
            study.evaluate()

    def test_repeated_study_reuses_identical_bank_and_refuses_config_change(self):
        with EmbeddingStudy(self.config) as study:
            first = study.prepare().sample_ids.copy()
        with EmbeddingStudy(self.config) as study:
            np.testing.assert_array_equal(study.prepare().sample_ids, first)
        with self.assertRaisesRegex(ValueError, "changed"):
            EmbeddingStudy(replace(self.config, neighbors=4))
        other = replace(self.config, output_dir=str(self.root / "other"))
        with EmbeddingStudy(other) as study:
            np.testing.assert_array_equal(study.prepare().sample_ids, first)

    def rewrite_identity(self, split, change):
        path = self.source / split / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        change(metadata["identity"])
        metadata["signature"] = signature(metadata["identity"])
        write_json(path, metadata)

    def test_checkpoint_mismatch_and_split_id_leakage_are_rejected(self):
        self.rewrite_identity(
            "validation", lambda value: value.update(checkpoint="other")
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            EmbeddingStudy(self.config)
        self.rewrite_identity(
            "validation",
            lambda value: value.update(
                checkpoint={"sha256": "synthetic-not-a-trained-checkpoint"},
                sample_ids=["train-0000"] + value["sample_ids"][1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            EmbeddingStudy(self.config)

    def test_source_checksums_and_invalid_embeddings_are_rejected(self):
        path = self.source / "train" / "cnn.npy"
        values = np.load(path)
        values[0] = 0
        np.save(path, values)
        with self.assertRaisesRegex(ValueError, "checksum"):
            CachedEmbeddings(self.source, "train")
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["arrays"]["cnn"]["sha256"] = digest_file(path)
        write_json(metadata_path, metadata)
        with (
            CachedEmbeddings(self.source, "train") as source,
            self.assertRaisesRegex(ValueError, "nonzero"),
        ):
            EmbeddingView(source, (0, 1, 2))

    def test_bounded_view_reads_only_requested_numeric_rows(self):
        with CachedEmbeddings(self.source, "train") as source:
            view = EmbeddingView(source, (0, 2))
            original = view.features.rows

            class GuardedRows:
                def __getitem__(self, indices):
                    if len(indices) > 4:
                        raise AssertionError("whole embedding matrix materialized")
                    return original[indices]

            view.features.rows = GuardedRows()
            self.assertEqual(view.features[:4].shape, (4, 320))
            self.assertEqual(view.features[[2, 5]].shape, (2, 320))

    def test_invalid_config_and_yaml_keys_are_rejected(self):
        for changes in (
            {"neighbors": 65},
            {"head_epochs": 0},
            {"bank_per_class": True},
            {"alpha_grid": (float("nan"),)},
            {"knn_mix_grid": (0.5,)},
            {"seed": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)
        path = self.root / "bad.yaml"
        path.write_text("source_root: foo\nunknown: 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
