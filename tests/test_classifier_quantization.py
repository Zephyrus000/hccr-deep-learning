import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear

from hccr.artifacts.checkpoints import file_digest
from hccr.benchmarking.quantization import (
    NormalizedDenseDeploymentHead,
    benchmark_classifier_quantization,
    build_dense_head_pair,
    cpu_quantization_context,
    load_classifier_deployment,
)
from hccr.models import EfficientHCCRNet
from hccr.models.heads import AngularMarginClassifier, FactorizedClassifier


def make_run(root, *, zero_head=False, manifest=None):
    options = {
        "num_classes": 3,
        "width": 4,
        "stage_depths": (1, 1, 1),
        "embedding_dim": 8,
        "dropout": 0.0,
        "classification_head": "cosface",
        "logit_scale": 8.0,
        "angular_margin": 0.1,
    }
    model = EfficientHCCRNet(**options).eval()
    if zero_head:
        with torch.no_grad():
            model.classifier.weight.zero_()
    root.mkdir()
    torch.save(model.state_dict(), root / "checkpoint.pt")
    metadata = {
        "model": {"name": "efficient_hccr", **options},
        "preprocess": {"image_size": 24, "margin": 4},
    }
    if manifest is not None:
        metadata["manifest_digest"] = file_digest(manifest)
    (root / "checkpoint_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (root / "labels.json").write_text(
        json.dumps({"labels": {"0": "一", "1": "丁", "2": "七"}}), encoding="utf-8"
    )
    return model


class ClassifierQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fp32_wrapper_is_exact_target_free_cosine_for_all_objectives(self):
        for kind in ("normalized_softmax", "cosface", "arcface"):
            source = AngularMarginClassifier(2, 3, kind, 16.0, 0.2).eval()
            with torch.no_grad():
                source.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 2.0], [-1.0, 0.0]]))
            deployed = NormalizedDenseDeploymentHead(source).eval()
            samples = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-8.0, 0.0]])
            expected = (
                torch.tensor(
                    [[1 - 1e-7, 0.8, -0.6], [0.0, 0.0, 0.0], [-0.6, 0.0, 1 - 1e-7]]
                )
                * 16
            )
            with torch.no_grad():
                torch.testing.assert_close(deployed(samples), expected)
                torch.testing.assert_close(
                    deployed(samples), source(samples), rtol=0, atol=0
                )
            self.assertEqual(deployed.linear.weight.dtype, torch.float32)
            self.assertNotEqual(
                deployed.linear.weight.data_ptr(), source.weight.data_ptr()
            )

    def test_real_per_channel_int8_kernel_preserves_backbone_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            model = make_run(Path(folder) / "run").train()
            before = copy.deepcopy(model.state_dict())
            previous_engine = torch.backends.quantized.engine
            with cpu_quantization_context(1):
                fp32, int8 = build_dense_head_pair(model)
                self.assertIsInstance(int8.classifier.linear, DynamicLinear)
                weight = int8.classifier.linear.weight()
                self.assertEqual(weight.dtype, torch.qint8)
                self.assertEqual(weight.qscheme(), torch.per_channel_affine)
                self.assertEqual(weight.int_repr().element_size(), 1)
                samples = torch.eye(8)
                with torch.inference_mode(), torch.profiler.profile() as profile:
                    actual = int8.classifier(samples)
                self.assertIn(
                    "quantized::linear_dynamic",
                    {row.key for row in profile.key_averages()},
                )
                torch.testing.assert_close(
                    actual, fp32.classifier(samples), atol=0.1, rtol=0.02
                )
                for name, value in fp32.state_dict().items():
                    if not name.startswith("classifier."):
                        torch.testing.assert_close(
                            value, int8.state_dict()[name], rtol=0, atol=0
                        )
            self.assertEqual(torch.backends.quantized.engine, previous_engine)
            self.assertTrue(model.training)
            for name, value in before.items():
                torch.testing.assert_close(
                    value, model.state_dict()[name], rtol=0, atol=0
                )

    def test_artifact_roundtrip_report_and_threads_restore_without_dataset_reads(self):
        from scripts.benchmark_classifier_quantization import main

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root / "run")
            previous_threads = torch.get_num_threads()
            with patch("hccr.benchmarking.quantization.HCCRDataset") as dataset:
                result = main(
                    [
                        "--source-run",
                        str(root / "run"),
                        "--output-dir",
                        str(root / "out"),
                        "--threads",
                        "2",
                        "--warmup",
                        "1",
                        "--iterations",
                        "2",
                        "--blocks",
                        "2",
                        "--samples",
                        "4",
                    ]
                )
            self.assertEqual(result, 0)
            dataset.assert_not_called()
            self.assertEqual(torch.get_num_threads(), previous_threads)
            report = json.loads((root / "out/report.json").read_text())
            self.assertIsNone(report["accuracy"])
            self.assertEqual(
                report["original_fp32_export_fidelity"]["max_absolute_score_drift"], 0
            )
            self.assertEqual(
                report["quantization"]["quantized_modules"], ["classifier.linear"]
            )
            self.assertEqual(report["timing_protocol"]["batch_size"], 1)
            self.assertEqual(
                report["artifacts"]["fp32"]["backbone_digest"],
                report["artifacts"]["int8"]["backbone_digest"],
            )
            for kind in ("fp32", "int8"):
                info = report["artifacts"][kind]
                path = root / "out" / info["file"]
                self.assertEqual(file_digest(path), info["sha256"])
                self.assertEqual(path.stat().st_size, info["bytes"])
                self.assertGreater(info["head_state_bytes"], 0)
                self.assertGreater(
                    report["head_only_latency"][kind]["median_p50_ms"], 0
                )
                self.assertGreater(
                    report["end_to_end_latency"][kind]["median_p95_ms"], 0
                )
            restored = load_classifier_deployment(root / "out/int8.pt")
            self.assertIsInstance(restored.classifier.linear, DynamicLinear)
            with torch.inference_mode():
                self.assertEqual(
                    tuple(restored(torch.ones(1, 1, 24, 24)).shape), (1, 3)
                )
                fp32 = load_classifier_deployment(root / "out/fp32.pt")
                generator = torch.Generator().manual_seed(7)
                torch.rand(4, 1, 24, 24, generator=generator)
                samples = torch.randn(4, 8, generator=generator)
                reference = torch.cat(
                    [fp32.classifier(row) for row in samples.split(1)]
                )
                candidate = torch.cat(
                    [restored.classifier(row) for row in samples.split(1)]
                )
                self.assertEqual(
                    report["synthetic_head_score_drift"]["max_absolute_score_drift"],
                    float((reference - candidate).abs().max()),
                )
            with self.assertRaises(FileExistsError):
                benchmark_classifier_quantization(root / "run", root / "out")

    def test_validation_only_exact_artifacts_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            Image.new("L", (10, 12), 128).save(root / "validation.png")
            columns = [
                "sample_id",
                "source_file",
                "writer_id",
                "unicode_label",
                "class_id",
                "split",
            ]
            rows = [
                ["v0", "validation.png", "v", "一", 0, "validation"],
                ["v1", "validation.png", "v", "丁", 1, "validation"],
                ["t", "MUST_NOT_OPEN.png", "t", "七", 2, "test"],
                ["i", "MUST_NOT_OPEN_ICDAR.png", "i", "七", 2, "icdar"],
                ["c", "MUST_NOT_OPEN_CAL.png", "c", "七", 2, "calibration"],
            ]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
            make_run(root / "run", zero_head=True, manifest=manifest)
            report = benchmark_classifier_quantization(
                root / "run",
                root / "out",
                warmup=0,
                iterations=1,
                blocks=1,
                samples=2,
                validation_manifest=manifest,
            )
            self.assertEqual(report["accuracy"]["split"], "validation")
            self.assertEqual(report["accuracy"]["samples"], 2)
            self.assertEqual(
                report["accuracy"]["top1_accuracy"], {"fp32": 0.5, "int8": 0.5}
            )
            self.assertEqual(report["accuracy"]["top1_agreement"], 1)

    def test_context_restores_on_failure_and_unsupported_heads_rejected(self):
        original = torch.get_num_threads(), torch.backends.quantized.engine
        with (
            self.assertRaisesRegex(RuntimeError, "forced"),
            cpu_quantization_context(2),
        ):
            raise RuntimeError("forced")
        self.assertEqual(
            (torch.get_num_threads(), torch.backends.quantized.engine), original
        )
        for classifier in (
            nn.Linear(8, 3),
            FactorizedClassifier(8, 3, "cosface", 32.0, 0.1, classifier_rank=2),
        ):
            model = nn.Module()
            model.classifier = classifier
            with self.assertRaisesRegex(ValueError, "normalized dense"):
                build_dense_head_pair(model)
        with (
            self.assertRaisesRegex(ValueError, "Unsupported CPU"),
            cpu_quantization_context(1, "none"),
        ):
            pass

    def test_failure_restores_benchmark_globals_without_publishing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root / "run")
            original = torch.get_num_threads(), torch.backends.quantized.engine
            with (
                patch(
                    "hccr.benchmarking.quantization._timings",
                    side_effect=RuntimeError("forced timing failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "forced timing failure"),
            ):
                benchmark_classifier_quantization(
                    root / "run",
                    root / "out",
                    threads=2,
                    samples=2,
                    warmup=0,
                    iterations=1,
                    blocks=1,
                )
            self.assertEqual(
                (torch.get_num_threads(), torch.backends.quantized.engine), original
            )
            self.assertFalse((root / "out").exists())
            self.assertTrue((root / "run/checkpoint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
