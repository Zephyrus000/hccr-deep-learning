from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hccr.data.lmdb_store import build_lmdb_image_store
from hccr.evaluation.hog_fusion import (
    benchmark_forward,
    pin_evaluation,
    prediction_indices,
    run_hog_evaluation,
    summarize_predictions,
)
from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.models.hog_fusion import HOGCNNFusion, HOGConfig


class HOGFusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        run = root / "run"
        run.mkdir()
        labels = {"labels": {"0": "a", "1": "b", "2": "c"}}
        for directory in (root, run):
            (directory / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        torch.manual_seed(7)
        cnn = EfficientHCCRNet(
            num_classes=3, width=2, stage_depths=(1, 1, 1), embedding_dim=320
        )
        model = HOGCNNFusion(cnn, HOGConfig(image_size=16)).eval()
        with torch.no_grad():
            model.fusion.weight[:, :320].neg_()
        torch.save(model.state_dict(), run / "checkpoint.pt")
        metadata = {
            "model": {
                **model.specification(),
                "cnn": {
                    "name": "efficient_hccr",
                    "num_classes": 3,
                    "width": 2,
                    "stage_depths": [1, 1, 1],
                    "embedding_dim": 320,
                },
            },
            "preprocess": {"image_size": 16, "margin": 2},
            "selection": {"epoch": 2, "split": "validation"},
        }
        (run / "checkpoint_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        rows = []
        for split in ("train", "validation", "test", "icdar"):
            for index in range(3):
                name = f"{split}-{index}"
                pixels = np.random.default_rng(index).integers(
                    0, 256, size=(16, 16), dtype=np.uint8
                )
                Image.fromarray(pixels).save(root / f"{name}.png")
                rows.append(
                    {
                        "sample_id": name,
                        "source_file": f"{name}.png",
                        "writer_id": f"{split}-writer",
                        "class_id": str(index),
                        "unicode_label": chr(ord("a") + index),
                        "split": split,
                    }
                )
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        build_lmdb_image_store(manifest, rows, root, root / "images.lmdb")
        return model, Namespace(
            run_dir=str(run),
            manifest=str(manifest),
            output_dir=str(root / "evaluation"),
            device="cpu",
            batch_size=2,
            num_workers=0,
            cpu_threads=1,
            skip_latency=True,
        )

    def test_metrics_match_known_predictions_and_paired_changes(self):
        targets = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = {
            "baseline": np.asarray([[0], [1], [0], [1], [2], [1]]),
            "fusion": np.asarray([[1], [0], [1], [1], [2], [1]]),
            "hog_off": np.asarray([[0], [0], [1], [1], [2], [2]]),
        }
        losses = {name: np.full(6, 2.0) for name in predictions}
        report = summarize_predictions(targets, predictions, losses, 3)
        self.assertEqual(report["samples"], 6)
        self.assertEqual(report["metrics"]["baseline"]["top1"], 0.5)
        self.assertEqual(report["metrics"]["fusion"]["repairs"], 2)
        self.assertEqual(report["metrics"]["fusion"]["damage"], 1)
        self.assertAlmostEqual(report["metrics"]["fusion"]["delta_pp"], 100 / 6)
        self.assertEqual(report["metrics"]["hog_off"]["macro_recall"], 1.0)

    def test_tied_logits_keep_argmax_separate_from_top_five(self):
        logits = torch.zeros(2, 32)
        first, five = prediction_indices(logits)
        np.testing.assert_array_equal(first, logits.argmax(1).numpy())
        np.testing.assert_array_equal(five, logits.topk(5, dim=1).indices.numpy())
        self.assertEqual(first.tolist(), [0, 0])
        names = ("baseline", "fusion", "hog_off")
        report = summarize_predictions(
            np.asarray([0]),
            {name: np.asarray([[1, 2, 3, 4, 5]]) for name in names},
            {name: np.asarray([1.0]) for name in names},
            32,
            top1={name: np.asarray([0]) for name in names},
        )
        self.assertEqual(report["metrics"]["baseline"]["top1"], 1.0)
        self.assertEqual(report["metrics"]["baseline"]["top5"], 0.0)

    def test_full_stream_pins_artifact_and_recomputable_archive_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args = self.fixture(root)
            report = run_hog_evaluation(args, expected_counts={"test": 3, "icdar": 3})
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["selection_frozen_before_images"])
            plan = json.loads(
                (root / "evaluation/evaluation_plan.json").read_text("utf-8")
            )
            self.assertEqual(plan["selection"]["epoch"], 2)
            self.assertEqual(plan["selection_after_holdout"], "forbidden")
            for split in ("test", "icdar"):
                result = report["results"][split]
                self.assertTrue(result["full_endpoint"])
                self.assertEqual(result["samples"], 3)
                with np.load(root / f"evaluation/{split}/predictions.npz") as archive:
                    self.assertEqual(
                        archive["sample_ids"].tolist(),
                        [f"{split}-{i}" for i in range(3)],
                    )
                    metric = summarize_predictions(
                        archive["targets"],
                        {
                            name: archive[f"{name}_top5"]
                            for name in ("baseline", "fusion", "hog_off")
                        },
                        {
                            name: archive[f"{name}_loss"]
                            for name in ("baseline", "fusion", "hog_off")
                        },
                        3,
                        top1={
                            name: archive[f"{name}_top1"]
                            for name in ("baseline", "fusion", "hog_off")
                        },
                    )
                    self.assertEqual(metric["metrics"], result["metrics"])
            resumed = run_hog_evaluation(args, expected_counts={"test": 3, "icdar": 3})
            self.assertEqual(resumed, report)
            archive = root / "evaluation/test/predictions.npz"
            archive.write_bytes(archive.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                run_hog_evaluation(args, expected_counts={"test": 3, "icdar": 3})

    def test_identity_rejects_changed_checkpoint_and_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args = self.fixture(root)
            settings = {"expected_counts": {"test": 3, "icdar": 3}}
            pin_evaluation(args.run_dir, args.manifest, args.output_dir, settings)
            checkpoint = root / "run/checkpoint.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                pin_evaluation(args.run_dir, args.manifest, args.output_dir, settings)
            meta_file = root / "run/checkpoint_metadata.json"
            meta = json.loads(meta_file.read_text("utf-8"))
            meta["selection"]["epoch"] = 0
            meta_file.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trained fusion"):
                pin_evaluation(args.run_dir, args.manifest, root / "another", settings)

    def test_full_endpoint_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "requires 4 rows"):
                run_hog_evaluation(args, expected_counts={"test": 4, "icdar": 3})
            self.assertTrue((root / "evaluation/evaluation_plan.json").exists())
            self.assertFalse((root / "evaluation/test/metrics.json").exists())

    def test_benchmark_executes_both_real_forward_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _ = self.fixture(Path(directory))
            report = benchmark_forward(
                model, torch.rand(1, 1, 16, 16), warmup=1, iterations=2, repetitions=2
            )
            self.assertEqual(report["failures"], 0)
            for name in ("baseline", "fusion"):
                blocks = report["measurements"][name]["raw_blocks_ms"]
                self.assertEqual([len(block) for block in blocks], [2, 2])
                self.assertGreater(
                    report["measurements"][name]["median_block_p50_ms"], 0
                )


if __name__ == "__main__":
    unittest.main()
