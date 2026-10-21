from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from scripts.run_embedding_ph_study import accuracy_comparison, main

from hccr.tda.embedding.smoke import create_synthetic_caches
from hccr.tda.embedding.study import EmbeddingConfig
from hccr.tda.features import write_json


class EmbeddingRunnerTests(unittest.TestCase):
    def test_complete_synthetic_job_then_idempotent_resume(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_synthetic_caches(root / "source")
            config = EmbeddingConfig(
                source_root=str(root / "source"),
                output_dir=str(root / "output"),
                max_classes=3,
                bank_per_class=4,
                neighbors=3,
                head_epochs=1,
                head_batch_size=8,
                alpha_grid=(1.0,),
                knn_mix_grid=(0.0, 1.0),
                device="cpu",
            )
            path = root / "config.yaml"
            write_json(path, asdict(config))
            arguments = ["--config", str(path), "--threads", "1"]
            self.assertEqual(main(arguments), 0)
            output = root / "output" / "seed-7"
            state = json.loads((output / "status.json").read_text(encoding="utf-8"))
            report = json.loads(
                (output / "accuracy_comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(report["test_samples"], 9)
            self.assertEqual(set(report["top1"]), {"cnn", "cnn_knn", "distance", "ph"})
            self.assertFalse(report["statistical_significance_established"])
            with self.assertRaises(FileExistsError):
                main(arguments)
            self.assertEqual(main([*arguments, "--resume"]), 0)

    def test_delta_is_percentage_points_and_all_controls_must_be_beaten(self):
        selection = {"selection_signature": "synthetic"}
        heldout = {
            "fingerprint": "synthetic",
            "seed": 7,
            "classes": 3,
            "writer_disjoint_verified": None,
            "held_out": {
                "test": {
                    name: {"top1": top1, "samples": 100}
                    for name, top1 in (
                        ("cnn", 0.94),
                        ("cnn_knn", 0.97),
                        ("distance", 0.95),
                        ("ph", 0.96),
                    )
                }
            },
        }
        report = accuracy_comparison(selection, heldout)
        self.assertAlmostEqual(report["ph_delta_percentage_points"]["cnn"], 2.0)
        self.assertAlmostEqual(report["ph_delta_percentage_points"]["cnn_knn"], -1.0)
        self.assertFalse(report["ph_beats_all_matched_controls"])


if __name__ == "__main__":
    unittest.main()
