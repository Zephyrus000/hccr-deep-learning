from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from hccr.models import EfficientHCCRNet
from hccr.tda.cubical import PH_FEATURE_NAMES, cubical_features, distribution_statistics
from hccr.tda.experiment import PilotConfig, Stage2Study, classification_metrics
from hccr.tda.features import FittedBlocks, Standardizer, load_cache, save_cache
from hccr.tda.projection import (
    FrozenStage2Adapter,
    ScalarProjection,
    load_projection,
    save_projection,
)
from hccr.tda.smoke import create_synthetic_inputs, synthetic_smoke


class CubicalTests(unittest.TestCase):
    def test_literal_constant_vertex_map(self) -> None:
        result = cubical_features(np.zeros((3, 3)))
        np.testing.assert_array_equal(
            result.features, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(result.diagrams[0], [[0, np.inf]])
        self.assertEqual(result.diagrams[1].shape, (0, 2))
        self.assertEqual(len(PH_FEATURE_NAMES), 14)

    def test_literal_ring_and_merging_components(self) -> None:
        ring = np.zeros((3, 3))
        ring[1, 1] = 2
        result = cubical_features(ring)
        np.testing.assert_array_equal(result.diagrams[1], [[0, 2]])
        np.testing.assert_array_equal(result.features[7:], [1, 2, 2, 0, 2, 0, 0])
        merged = cubical_features([[0, 2, 0], [2, 2, 2], [2, 2, 2]])
        np.testing.assert_array_equal(merged.features[:7], [1, 2, 2, 0, 2, 0, 1])
        self.assertEqual(sorted(merged.diagrams[0][:, 1]), [2, np.inf])

    def test_signed_maps_ties_statistics_and_boundary_errors(self) -> None:
        np.testing.assert_array_equal(
            cubical_features(np.full((3, 3), -2)).diagrams[0], [[-2, np.inf]]
        )
        np.testing.assert_array_equal(
            distribution_statistics(np.full((3, 3), -2)),
            [-2, 0, 2, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        )
        for invalid in ([], [[[0]]], [[np.nan]], [[np.inf]]):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "2D vertex map"),
            ):
                cubical_features(invalid)


class FeatureTests(unittest.TestCase):
    def test_literal_two_block_zscore_and_weight(self) -> None:
        cnn = np.array([[1, 2], [3, 6]])
        auxiliary = np.array([[10], [14]])
        blocks = FittedBlocks.fit(cnn, auxiliary)
        np.testing.assert_array_equal(
            blocks.fuse(cnn, auxiliary, 0.5), [[-1, -1, -0.5], [1, 1, 0.5]]
        )
        restored = FittedBlocks.from_state(blocks.state())
        np.testing.assert_array_equal(
            restored.fuse(cnn, auxiliary, 0.5), [[-1, -1, -0.5], [1, 1, 0.5]]
        )

    def test_train_constant_columns_do_not_erase_changed_heldout_values(self) -> None:
        fitted = Standardizer.fit(np.array([[2, 4], [2, 8]]))
        np.testing.assert_array_equal(fitted.transform(np.array([[9, 10]])), [[7, 2]])
        np.testing.assert_array_equal(fitted.scale, [1, 2])

    def test_pca_rank_padding_and_heldout_do_not_refit(self) -> None:
        train = np.array([[0, 0], [1, 2], [2, 4]], dtype=float)
        blocks = FittedBlocks.fit(train, train, pca=True)
        state_before = json.dumps(blocks.state(), sort_keys=True)
        self.assertEqual(blocks.state()["pca_rank"], 1)
        transformed = blocks.auxiliary_values(np.array([[100, -100]]))
        self.assertEqual(transformed.shape, (1, 14))
        np.testing.assert_array_equal(transformed[:, 1:], np.zeros((1, 13)))
        restored = FittedBlocks.from_state(blocks.state())
        np.testing.assert_array_equal(
            restored.fuse(train, train, 0.5), blocks.fuse(train, train, 0.5)
        )
        self.assertEqual(json.dumps(blocks.state(), sort_keys=True), state_before)

    def test_cache_rejects_projection_schema_changes_and_incomplete_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "train"
            identity = {
                "projection": "weights-a",
                "feature_schema": 1,
                "sample_ids": ["train-1"],
            }
            save_cache(cache, {"cnn": np.array([[1, 2]], dtype=np.float32)}, identity)
            np.testing.assert_array_equal(load_cache(cache, identity)["cnn"], [[1, 2]])
            for key, changed in (("projection", "weights-b"), ("feature_schema", 2)):
                with self.assertRaisesRegex(ValueError, key):
                    load_cache(cache, {**identity, key: changed})
            with self.assertRaisesRegex(ValueError, "incomplete cache"):
                load_cache(Path(directory) / "interrupted", identity)

    def test_candidate_restriction_and_remapping_are_consistent(self) -> None:
        logits = np.array([[0, 5, 100, 1], [0, 1, 100, 5]])
        candidates = [1, 3]
        original_to_pilot = {1: 0, 3: 1}
        targets = np.array([original_to_pilot[label] for label in [1, 3]])
        self.assertEqual(
            classification_metrics(logits[:, candidates], targets)["top1"], 1
        )
        self.assertEqual(
            classification_metrics(np.array([[5, 1], [1, 5]]), targets)["top1"], 1
        )


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_mean_initialization_known_signed_weights_and_roundtrip(self) -> None:
        features = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        projection = ScalarProjection(2)
        torch.testing.assert_close(
            projection(features),
            torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]]),
            rtol=0,
            atol=0,
        )
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[[[2.0]], [[-1.0]]]]))
        expected = torch.tensor([[[[-3.0, -2.0], [-1.0, 0.0]]]])
        torch.testing.assert_close(projection(features), expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "projection.pt"
            save_projection(target, projection, {"mode": "fixture"})
            restored, metadata = load_projection(target)
            torch.testing.assert_close(restored(features), expected, rtol=0, atol=0)
            self.assertEqual(metadata, {"mode": "fixture"})

    def test_one_pass_baseline_logit_parity_and_frozen_eval(self) -> None:
        torch.manual_seed(7)
        baseline = EfficientHCCRNet(3, width=4, stage_depths=(1, 1, 1)).eval()
        images = torch.randn(2, 1, 16, 16)
        original = baseline(images)
        adapter = FrozenStage2Adapter(baseline).train()
        with patch.object(
            baseline, "forward_features", wraps=baseline.forward_features
        ) as forward:
            embedding, stage, logits = adapter(images)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(list(stage.shape), [2, 8, 4, 4])
        self.assertEqual(list(embedding.shape), [2, 16])
        torch.testing.assert_close(logits, original, rtol=0, atol=0)
        self.assertFalse(baseline.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in baseline.parameters())
        )

    def test_spatial_supervision_updates_only_projection_and_caches_are_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = create_synthetic_inputs(root)
            config = PilotConfig(
                str(run),
                str(manifest),
                str(root / "study"),
                max_classes=2,
                batch_size=4,
                projection_epochs=2,
                projection_learning_rate=0.01,
                calibration_samples=3,
                head_epochs=1,
                device="cpu",
            )
            study = Stage2Study(config)
            before = {
                key: value.clone()
                for key, value in study.adapter.baseline.state_dict().items()
            }
            study.fit_projection()
            projection, metadata = study.projection()
            self.assertTrue(metadata["training"]["weights_updated"])
            self.assertEqual(
                metadata["training"]["supervision"],
                "classification_cross_entropy_not_PH",
            )
            self.assertEqual(len(metadata["calibration"]["sample_ids"]), 3)
            self.assertTrue(
                all(
                    sample.startswith("train-")
                    for sample in metadata["calibration"]["sample_ids"]
                )
            )
            for key, value in study.adapter.baseline.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            first, second = study.extract("train"), study.extract("train")
            np.testing.assert_array_equal(first["h0_h1"], second["h0_h1"])
            repeated = Stage2Study(replace(config, output_dir=str(root / "repeat")))
            save_projection(repeated.output / "projection.pt", projection, metadata)
            recomputed = repeated.extract("train")
            for key in first:
                np.testing.assert_array_equal(first[key], recomputed[key])
            # Release open memory maps before Windows removes the test directory.
            del first, second, recomputed
            with torch.no_grad():
                projection.weight.add_(0.25)
            save_projection(study.output / "projection.pt", projection, metadata)
            with self.assertRaisesRegex(ValueError, "projection"):
                study.extract("train")


class PipelineTests(unittest.TestCase):
    def test_bounded_synthetic_path_completes_all_controls_and_holdouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = synthetic_smoke(Path(directory))
            self.assertEqual(report["train_samples"], 16)
            self.assertEqual(report["heldout_splits"], ["test", "icdar"])
            self.assertEqual(
                report["variants"],
                [
                    "original",
                    "cnn",
                    "gap",
                    "flat",
                    "statistics",
                    "h0",
                    "h0_h1",
                    "mean_h0_h1",
                ],
            )
            self.assertFalse(report["research_accuracy_gain_established"])

    def test_holdouts_cannot_open_before_complete_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = create_synthetic_inputs(root)
            study = Stage2Study(
                PilotConfig(
                    str(run),
                    str(manifest),
                    str(root / "study"),
                    max_classes=2,
                    device="cpu",
                )
            )
            with patch.object(
                study, "dataset", side_effect=AssertionError("holdout was opened")
            ):
                with self.assertRaisesRegex(ValueError, "before opening test"):
                    study.evaluate(("test", "icdar"))
                with self.assertRaisesRegex(ValueError, "extract accepts only"):
                    study.extract("test")
            self.assertEqual(study.identity["candidate_policy"], "pilot_classes_only")


if __name__ == "__main__":
    unittest.main()
