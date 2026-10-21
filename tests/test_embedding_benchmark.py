from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.benchmark_embedding_ph import benchmark
from scripts.run_optimized_embedding_ph import main

from hccr.tda.embedding.smoke import create_synthetic_caches
from hccr.tda.embedding.study import EmbeddingConfig, EmbeddingStudy
from hccr.tda.features import write_json


class BenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA benchmark fixture")
    def test_all_timed_paths_publish_identical_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_synthetic_caches(root / "source")
            config = EmbeddingConfig(
                source_root=str(root / "source"),
                output_dir=str(root / "study"),
                max_classes=3,
                bank_per_class=4,
                neighbors=3,
                query_batch_size=8,
                ph_workers=2,
                ph_chunk_size=2,
            )
            with EmbeddingStudy(config) as study:
                study.prepare()
            output = root / "benchmark"
            try:
                benchmark(
                    config,
                    root / "study" / "seed-7" / "bank",
                    output,
                    queries=12,
                    trials=5,
                    notify=lambda message: None,
                )
            except RuntimeError as error:
                # A synthetic fixture is a parity check, never a speed claim.
                self.assertIn("no clear measured throughput gain", str(error))
            report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
            self.assertTrue(report["exact_array_hash_parity"])
            self.assertFalse(report["test_or_icdar_opened"])
            for name in ("reference", "cpu_exact", "cuda_exact"):
                self.assertEqual(len(report["summary"][name]["seconds"]), 5)

    def test_failed_benchmark_cannot_launch_full_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = EmbeddingConfig(
                source_root="unused", output_dir=str(root / "output")
            )
            config_path = root / "config.yaml"
            write_json(config_path, asdict(config))
            with patch("scripts.run_optimized_embedding_ph.run_study") as runner:
                code = main(
                    ["--config", str(config_path), "--bank", str(root / "absent")]
                )
                runner.assert_not_called()
            self.assertEqual(code, 1)
            state = json.loads(
                (root / "output" / "dispatch_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failed_stage"], "benchmark_initializing")


if __name__ == "__main__":
    unittest.main()
