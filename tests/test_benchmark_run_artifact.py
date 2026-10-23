import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hccr.cli import main
from hccr.commands import benchmark
from hccr.models import EfficientHCCRNet


class BenchmarkRunArtifactTests(unittest.TestCase):
    def test_cli_benchmarks_exact_factorized_checkpoint_and_stored_preprocessing(self):
        options = dict(
            num_classes=7,
            width=4,
            stage_depths=(1, 1, 1),
            embedding_dim=8,
            dropout=0.0,
            classification_head="normalized_softmax",
            classifier_structure="dynamic_residual",
            classifier_rank=3,
            residual_class_indices=[5, 1],
            gate_use_gap=False,
            logit_scale=16.0,
            angular_margin=0.0,
        )
        original = EfficientHCCRNet(**options).eval()
        with torch.no_grad():
            original.classifier.residuals.normal_(std=0.2)
            original.classifier.gate_weight.normal_(std=0.2)
            sample = torch.ones(1, 1, 24, 24)
            expected = original(sample)
        real_benchmark = benchmark.benchmark

        def verify_and_benchmark(model, image_size, *args):
            self.assertEqual(image_size, 24)
            self.assertEqual(model.classifier.residual_indices.tolist(), [5, 1])
            self.assertFalse(model.classifier.gate_use_gap)
            with torch.inference_mode():
                torch.testing.assert_close(model(sample), expected)
            return real_benchmark(model, image_size, *args)

        previous_threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as folder:
                run_dir = Path(folder)
                torch.save(original.state_dict(), run_dir / "checkpoint.pt")
                (run_dir / "checkpoint_metadata.json").write_text(
                    json.dumps(
                        {
                            "model": {"name": "efficient_hccr", **options},
                            "preprocess": {"image_size": 24},
                        }
                    ),
                    encoding="utf-8",
                )
                output_path = run_dir / "benchmark.json"
                with (
                    patch(
                        "hccr.commands.benchmark.benchmark",
                        side_effect=verify_and_benchmark,
                    ) as measured,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = main(
                        [
                            "benchmark",
                            "--run-dir",
                            str(run_dir),
                            "--device",
                            "cpu",
                            "--image-size",
                            "96",
                            "--warmup-iterations",
                            "1",
                            "--iterations",
                            "2",
                            "--output",
                            str(output_path),
                        ]
                    )
                self.assertEqual(result, 0)
                measured.assert_called_once()
                report = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(report["image_size"], 24)
                self.assertEqual(report["num_classes"], 7)
                self.assertEqual(report["embedding_dim"], 8)
                self.assertEqual(report["classification_head"], "normalized_softmax")
                self.assertEqual(report["classifier_structure"], "dynamic_residual")
                self.assertIn(
                    "factorized_classifier_with_scalar_norm_cache",
                    report["shared_inference_behavior"],
                )
                self.assertGreater(report["results"]["float32"]["latency_p95_ms"], 0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_run_directory_and_standalone_checkpoint_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "choose run-dir"):
            main(
                [
                    "benchmark",
                    "--device",
                    "cpu",
                    "--run-dir",
                    "run",
                    "--checkpoint",
                    "checkpoint.pt",
                ]
            )
