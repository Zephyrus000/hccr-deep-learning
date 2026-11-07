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
from torch.nn import functional as F

from hccr.data.lmdb_store import build_lmdb_image_store
from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.models.hog_fusion import (
    HistogramOfOrientedGradients,
    HOGCNNFusion,
    HOGConfig,
    load_hog_fusion_run,
)
from hccr.training.hog_fusion import cached_batches, evaluate_pair, run_hog_fusion


def numpy_hog(pixels, config):
    gx, gy = np.zeros_like(pixels), np.zeros_like(pixels)
    gx[:, 1:-1] = pixels[:, 2:] - pixels[:, :-2]
    gy[1:-1] = pixels[2:] - pixels[:-2]
    magnitude = np.sqrt(gx**2 + gy**2)
    bins = np.floor(np.mod(np.arctan2(gy, gx), np.pi) * config.bins / np.pi)
    cells = config.image_size // config.cell_size
    hist = np.zeros((cells, cells, config.bins))
    for y in range(config.image_size):
        for x in range(config.image_size):
            index = min(int(bins[y, x]), config.bins - 1)
            hist[y // config.cell_size, x // config.cell_size, index] += magnitude[y, x]
    hist /= config.cell_size**2
    blocks = []
    for y in range(cells - config.block_cells + 1):
        for x in range(cells - config.block_cells + 1):
            block = hist[y : y + config.block_cells, x : x + config.block_cells].ravel()
            block = block / np.sqrt(np.sum(block**2) + config.eps**2)
            block = np.minimum(block, config.clip)
            blocks.append(block / np.sqrt(np.sum(block**2) + config.eps**2))
    return np.concatenate(blocks)


class HOGFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(7)
        cnn = EfficientHCCRNet(
            num_classes=3, width=2, stage_depths=(1, 1, 1), embedding_dim=320
        )
        return HOGCNNFusion(cnn, HOGConfig(image_size=16))

    def test_hog_matches_independent_reference_and_known_directions(self):
        cfg = HOGConfig(image_size=24)
        pixels = np.random.default_rng(5).random((24, 24)).astype(np.float32)
        hog = HistogramOfOrientedGradients(cfg)
        actual = hog(torch.from_numpy(pixels)[None, None])[0].numpy()
        np.testing.assert_allclose(actual, numpy_hog(pixels, cfg), atol=2e-6)
        cfg = HOGConfig(image_size=16)
        hog = HistogramOfOrientedGradients(cfg)
        horizontal = torch.arange(16).float().expand(16, 16) / 16
        for image, direction in ((horizontal, 0), (horizontal.T, 4)):
            expected = torch.zeros(4, 9)
            expected[:, direction] = 0.5
            torch.testing.assert_close(hog(image[None, None])[0], expected.flatten())
        torch.testing.assert_close(
            hog(horizontal[None, None]), hog((1 - horizontal)[None, None])
        )
        self.assertTrue(torch.equal(hog(torch.zeros(2, 1, 16, 16)), torch.zeros(2, 36)))

    def test_identity_then_training_preserves_frozen_cnn_and_updates_predictions(self):
        model = self.model()
        images = torch.rand(8, 1, 16, 16)
        baseline = model.cnn(images).detach()
        model.train()
        self.assertFalse(model.cnn.training)
        self.assertTrue(torch.equal(model(images), baseline))
        before = {k: v.clone() for k, v in model.cnn.state_dict().items()}
        cnn, hog = model.extract(images)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        optimizer.zero_grad()
        F.cross_entropy(model.forward_cached(cnn, hog), torch.arange(8) % 3).backward()
        self.assertIsNotNone(model.fusion.weight.grad)
        self.assertGreater(model.fusion.weight.grad[:, 320:].abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(model(images), baseline))
        for name, value in model.cnn.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(p.grad is None for p in model.cnn.parameters()))

    def test_fusion_checkpoint_round_trip_without_parent(self):
        model = self.model().eval()
        with torch.no_grad():
            model.fusion.weight[:, 320:].normal_(std=0.01)
        images = torch.rand(3, 1, 16, 16)
        expected = model(images)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                }
            }
            (root / "checkpoint_metadata.json").write_text(json.dumps(metadata))
            torch.save(model.state_dict(), root / "checkpoint.pt")
            restored = load_hog_fusion_run(root)
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_cached_epoch_visits_every_sample_and_preserves_pairs(self):
        targets = np.arange(8201, dtype=np.int64)
        arrays = {"cnn": targets[:, None].astype(np.float32)}
        arrays["hog"] = arrays["cnn"] * 3
        visited = []
        for cnn, hog, labels in cached_batches(
            arrays, targets, "cpu", 127, rng=np.random.default_rng(7)
        ):
            self.assertTrue(torch.equal(cnn[:, 0].long(), labels))
            self.assertTrue(torch.equal(hog[:, 0].long(), labels * 3))
            visited.extend(labels.tolist())
        self.assertEqual(sorted(visited), targets.tolist())
        self.assertNotEqual(visited, targets.tolist())

    def test_pair_evaluation_counts_repairs_and_damage(self):
        model = self.model().eval()
        arrays = {
            "cnn": np.random.default_rng(9).normal(size=(9, 320)).astype(np.float32),
            "hog": np.zeros((9, 36), dtype=np.float32),
        }
        targets = np.arange(9, dtype=np.int64) % 3
        initial, _, _ = evaluate_pair(model, arrays, targets, "cpu", 4)
        self.assertEqual(initial["repairs"], 0)
        self.assertEqual(initial["damage"], 0)
        with torch.no_grad():
            model.fusion.weight[:, :320].neg_()
        scores, baseline, fusion = evaluate_pair(model, arrays, targets, "cpu", 4)
        expected = 100 * ((fusion == targets).mean() - (baseline == targets).mean())
        self.assertAlmostEqual(scores["delta_pp"], expected)
        self.assertGreater(scores["prediction_changes"], 0)

    def test_complete_runner_with_real_images_cache_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            labels = {str(index): chr(0x4E00 + index) for index in range(3755)}
            payload = {"labels": labels}
            for folder in (root, source):
                (folder / "labels.json").write_text(json.dumps(payload))
            rows = []
            for index in range(8):
                filename = f"image-{index}.png"
                pixels = np.random.default_rng(index).integers(
                    0, 256, (16, 16), dtype=np.uint8
                )
                Image.fromarray(pixels).save(root / filename)
                target = index % 2
                rows.append(
                    {
                        "sample_id": str(index),
                        "source_file": filename,
                        "writer_id": "train-writer" if index < 6 else "val-writer",
                        "unicode_label": chr(0x4E00 + target),
                        "class_id": str(target),
                        "split": "train" if index < 6 else "validation",
                    }
                )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            build_lmdb_image_store(manifest, rows, root, root / "images.lmdb")
            cnn = EfficientHCCRNet(
                num_classes=3755, width=2, stage_depths=(1, 1, 1), embedding_dim=320
            )
            meta = {
                "model": {
                    "name": "efficient_hccr",
                    "num_classes": 3755,
                    "width": 2,
                    "stage_depths": [1, 1, 1],
                    "embedding_dim": 320,
                },
                "preprocess": {"image_size": 16, "margin": 2},
                "training": {"precision": {"resolved": "float32"}},
            }
            (source / "checkpoint_metadata.json").write_text(json.dumps(meta))
            torch.save(cnn.state_dict(), source / "checkpoint.pt")
            args = Namespace(
                source_run=str(source),
                manifest=str(manifest),
                output_dir=str(root / "run"),
                device="cpu",
                seed=7,
                epochs=1,
                learning_rate=1e-3,
                batch_size=2,
                extract_batch_size=2,
                num_workers=0,
                cpu_threads=1,
                resume=False,
                min_baseline_top1=0.0,
            )
            scores = run_hog_fusion(args)
            self.assertEqual(scores["samples"], 2)
            self.assertTrue(scores["frozen_state_unchanged"])
            self.assertEqual(scores["test_status"], "not_run")
            report = json.loads((root / "run/hog_ablation.json").read_text())
            self.assertEqual(report["without_hog"]["samples"], 2)
            args.resume = True
            resumed = run_hog_fusion(args)
            self.assertEqual(resumed, scores)


if __name__ == "__main__":
    unittest.main()
