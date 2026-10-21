from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from hccr.models import EfficientHCCRNet
from hccr.models.heads import AngularMarginClassifier
from hccr.tda.experiment import PilotConfig, Stage2Study, train_common_head
from hccr.tda.features import (
    digest_file,
    implementation_identity,
    load_cache,
    save_cache,
)
from hccr.tda.projection import (
    FrozenStage2Adapter,
    ProjectionBudget,
    ScalarProjection,
    calibrate_maps,
    fit_projection,
)
from hccr.tda.smoke import create_synthetic_inputs


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_train_rms_literal_zero_and_nonfinite_boundaries(self) -> None:
        images = torch.stack([torch.ones(1, 2, 2), torch.full((1, 2, 2), 3.0)])
        loader = DataLoader(
            TensorDataset(images, torch.tensor([0, 1]), torch.arange(2)), batch_size=2
        )
        adapter = SimpleNamespace(encode=lambda values: (values, values))
        projection = ScalarProjection(1)
        scale = calibrate_maps(
            adapter, projection, loader, ["train-a", "train-b"], "cpu", 2
        )
        self.assertAlmostEqual(scale["scale"], 5**0.5)
        self.assertEqual(scale["sample_ids"], ["train-a", "train-b"])
        self.assertEqual(
            calibrate_maps(
                adapter, projection, loader, ["train-a", "train-b"], "cpu", 1
            )["scale"],
            1,
        )
        with torch.no_grad():
            projection.weight.zero_()
        self.assertEqual(
            calibrate_maps(
                adapter, projection, loader, ["train-a", "train-b"], "cpu", 2
            )["scale"],
            1,
        )
        self.assertTrue(
            calibrate_maps(
                adapter, projection, loader, ["train-a", "train-b"], "cpu", 2
            )["degenerate_zero_map"]
        )
        with torch.no_grad():
            projection.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite training map RMS"):
            calibrate_maps(
                adapter, projection, loader, ["train-a", "train-b"], "cpu", 2
            )

    def test_projection_checkpoint_uses_validation_loss_not_last_epoch(self) -> None:
        torch.manual_seed(7)
        adapter = FrozenStage2Adapter(
            EfficientHCCRNet(2, width=4, stage_depths=(1, 1, 1))
        )
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 1, 16, 16), torch.tensor([0, 1, 0, 1]), torch.arange(4)
            ),
            batch_size=4,
        )
        original_loss = functional.cross_entropy
        holder, states = {}, []

        def create_projection(channels):
            holder["projection"] = ScalarProjection(channels)
            return holder["projection"]

        def controlled_validation(logits, targets, **options):
            if options.get("reduction") == "sum":
                states.append(holder["projection"].weight.detach().clone())
                return torch.tensor(0.0 if len(states) == 1 else 40.0)
            return original_loss(logits, targets, **options)

        with (
            patch(
                "hccr.tda.projection.ScalarProjection", side_effect=create_projection
            ),
            patch(
                "hccr.tda.projection.F.cross_entropy", side_effect=controlled_validation
            ),
        ):
            projection, evidence = fit_projection(
                adapter,
                loader,
                loader,
                2,
                "cpu",
                ProjectionBudget(epochs=2, learning_rate=0.01),
            )
        self.assertEqual(evidence["selected_epoch"], 1)
        self.assertEqual(evidence["best_validation_loss"], 0)
        torch.testing.assert_close(projection.weight, states[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(states[0], states[1]))

    def test_head_checkpoint_uses_validation_top1_not_last_epoch(self) -> None:
        features = np.eye(2, dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        config = PilotConfig(
            "unused", "unused", head_epochs=2, head_batch_size=2, device="cpu"
        )
        holder, states = {}, []

        def create_head(*args):
            holder["head"] = AngularMarginClassifier(*args)
            return holder["head"]

        def controlled_validation(*args):
            states.append(holder["head"].weight.detach().clone())
            return {"top1": 1.0 if len(states) == 1 else 0.0}

        with (
            patch(
                "hccr.tda.experiment.AngularMarginClassifier", side_effect=create_head
            ),
            patch(
                "hccr.tda.experiment.classification_metrics",
                side_effect=controlled_validation,
            ),
        ):
            head, evidence = train_common_head(
                features, labels, features, labels, 2, config, 32, 0.1, "cpu"
            )
        self.assertEqual(evidence["selected_epoch"], 1)
        torch.testing.assert_close(head.weight, states[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(states[0], states[1]))

    def test_alpha_ties_take_first_and_no_holdout_opens_during_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, manifest = create_synthetic_inputs(root)
            config = PilotConfig(
                str(run),
                str(manifest),
                str(root / "study"),
                max_classes=2,
                batch_size=4,
                projection_epochs=1,
                projection_learning_rate=0.01,
                calibration_samples=2,
                head_epochs=1,
                variants=("cnn", "statistics"),
                device="cpu",
            )
            study = Stage2Study(config)
            study.fit_projection()
            requests = []
            original_dataset = study.dataset

            def development_only(split):
                requests.append(split)
                self.assertIn(split, ("train", "validation"))
                return original_dataset(split)

            def tied_head(
                training,
                targets,
                validation,
                validation_targets,
                classes,
                settings,
                scale,
                margin,
                device,
            ):
                head = AngularMarginClassifier(
                    training.shape[1], classes, "cosface", scale, margin
                ).to(device)
                with torch.no_grad():
                    head.weight.zero_()
                return head, {"parameters": head.weight.numel(), "selected_epoch": 1}

            with (
                patch.object(study, "dataset", side_effect=development_only),
                patch("hccr.tda.experiment.train_common_head", side_effect=tied_head),
            ):
                selection = study.compare()
            self.assertEqual(selection["variants"]["statistics"]["alpha"], 0.25)
            self.assertEqual(
                selection["variants"]["statistics"]["validation"]["top1"], 0.5
            )
            self.assertEqual(len(selection["variants"]["statistics"]["trials"]), 4)
            self.assertEqual(len(selection["variants"]["cnn"]["trials"]), 1)
            self.assertEqual(set(requests), {"train", "validation"})
            heldout = study.evaluate(("test", "icdar"))
            self.assertEqual(set(heldout["held_out"]["test"]), {"cnn", "statistics"})

    def test_cache_invalidates_when_upstream_feature_implementation_changes(
        self,
    ) -> None:
        original = implementation_identity()
        self.assertIn("models/heads.py", original["source"])
        self.assertIn("preprocessing/pipeline.py", original["source"])

        def changed_source(path):
            return (
                "changed-head-implementation"
                if path.name == "heads.py"
                else digest_file(path)
            )

        with patch("hccr.tda.features.digest_file", side_effect=changed_source):
            changed = implementation_identity()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "train"
            save_cache(cache, {"cnn": np.ones((1, 2))}, {"implementation": original})
            with self.assertRaisesRegex(ValueError, "implementation"):
                load_cache(cache, {"implementation": changed})


if __name__ == "__main__":
    unittest.main()
