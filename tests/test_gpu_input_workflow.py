from __future__ import annotations

import csv
import json
import tempfile
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

from hccr.artifacts.checkpoints import file_digest, save_checkpoint
from hccr.cli import build_parser
from hccr.commands.train import config_from_arguments
from hccr.data.dataset import HCCRDataset
from hccr.models import build_model, load_model_from_run
from hccr.preprocessing import EvalPreprocessor
from hccr.training.proposal import state_digest
from hccr.training.workflow import (
    TrainingConfig,
    _build_training_model,
    _model_metadata,
    _validate_training_config,
    build_training_cache,
    profile_model,
    run_training,
)


class GPUInputWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        rows = []
        for split in ("train", "calibration", "validation", "test", "icdar"):
            for class_id, label in enumerate(("一", "二", "三")):
                name = f"{split}-{class_id}.png"
                # Unopened holdout and calibration files expose accidental reads.
                if split in {"train", "validation"}:
                    image = Image.new("L", (14, 14), 255)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(
                        (2 + class_id, 3, 7 + class_id, 10), fill=class_id * 50
                    )
                    image.save(self.root / name)
                rows.append(
                    {
                        "sample_id": name,
                        "source_file": name,
                        "writer_id": split,
                        "unicode_label": label,
                        "class_id": class_id,
                        "split": split,
                    }
                )
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def config(self, name, **changes):
        base = TrainingConfig(
            self.manifest,
            self.root / name,
            3,
            epochs=1,
            batch_size=3,
            reference_batch_size=3,
            optimizer_step_policy="configured_epochs",
            learning_rate_scaling="none",
            image_size=16,
            width=4,
            stage_depths=(1, 1, 1),
            embedding_dim=8,
            reparameterize_depthwise=False,
            dropout=0,
            device="cpu",
            evaluation_policy="validation_only",
            scheduler="none",
            early_stopping_patience=None,
            benchmark_warmup_iterations=1,
            benchmark_iterations=1,
            benchmark_repetitions=1,
            reproducibility_mode="strict",
            training_cache_max_gib=0.01,
            training_cache_reserve_gib=0,
            num_workers=0,
        )
        return replace(base, **changes)

    def execute(self, config):
        observed = []
        cache_references = []
        profiling_checks = []

        def prepare_cache(*args, **kwargs):
            cache, metadata = build_training_cache(*args, **kwargs)
            if cache is not None:
                cache_references.extend(
                    weakref.ref(value) for value in (cache, cache.images, cache.targets)
                )
            return cache, metadata

        def profile(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in cache_references))
            profiling_checks.append(True)
            return profile_model(*args, **kwargs)

        def build(*args, **kwargs):
            model = _build_training_model(*args, **kwargs)

            def record(module, inputs):
                if len(inputs) > 1:
                    observed.append(
                        (
                            inputs[0].detach().cpu().clone(),
                            inputs[1].detach().cpu().clone(),
                        )
                    )

            model.register_forward_pre_hook(record)
            return model

        with (
            patch("hccr.training.workflow._build_training_model", side_effect=build),
            patch(
                "hccr.training.workflow.build_training_cache", side_effect=prepare_cache
            ),
            patch("hccr.training.workflow.profile_model", side_effect=profile),
        ):
            run_training(config)
        self.assertEqual(profiling_checks, [True])
        run = next(config.output_dir.glob("20*"))
        metadata = json.loads((run / "checkpoint_metadata.json").read_text())
        pipeline = json.loads((run / "input_pipeline.json").read_text())
        restored = load_model_from_run(run).eval()
        with torch.inference_mode():
            self.assertEqual(tuple(restored(torch.ones(1, 1, 16, 16)).shape), (1, 3))
        self.assertEqual(metadata["training"]["completed_optimizer_steps"], 1)
        self.assertEqual(metadata["test"]["status"], "not_run")
        self.assertEqual(len(observed), 1)
        self.assertTrue((run / "augmentation_gallery.png").is_file())
        self.assertEqual(pipeline["manifest_digest"], file_digest(self.manifest))
        self.assertTrue(pipeline["fit_only"])
        return metadata, pipeline, observed[0], restored

    def test_cli_roundtrip_and_incompatible_cache_modes(self):
        arguments = build_parser().parse_args(
            [
                "train",
                "--manifest",
                str(self.manifest),
                "--num-classes",
                "3",
                "--augmentation-backend",
                "gpu",
                "--training-cache",
                "cpu",
                "--training-cache-max-gib",
                "2",
                "--training-cache-reserve-gib",
                "1",
            ]
        )
        config = config_from_arguments(arguments)
        _validate_training_config(config)
        self.assertEqual(config.augmentation_backend, "gpu")
        self.assertEqual(config.training_cache, "cpu")
        self.assertEqual(config.training_cache_max_gib, 2)
        self.assertEqual(config.training_cache_reserve_gib, 1)
        for values, message in (
            ({"augmentation_backend": "cpu"}, "frozen pilot"),
            ({"distributed": True}, "DDP"),
            ({"training_cache_max_gib": 0}, "budget"),
            ({"training_cache_reserve_gib": float("nan")}, "budget"),
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, message),
            ):
                _validate_training_config(replace(config, **values))
        with self.assertRaisesRegex(ValueError, "CUDA training device"):
            run_training(self.config("invalid-cpu", augmentation_backend="gpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_streaming_and_both_cache_placements_preserve_sample_order(self):
        expected_order = torch.randperm(3, generator=torch.Generator().manual_seed(8))
        reference = None
        for placement in ("none", "cpu", "cuda"):
            with self.subTest(placement=placement):
                metadata, pipeline, batch, _ = self.execute(
                    self.config(
                        f"gpu-{placement}",
                        device="cuda",
                        augmentation_backend="gpu",
                        training_cache=placement,
                        train_blur_probability=0,
                    )
                )
                self.assertEqual(
                    pipeline["augmentation"]["backend"], "tensor_affine_v1"
                )
                self.assertEqual(metadata["training"]["augmentation_backend"], "gpu")
                self.assertEqual(metadata["training"]["training_cache"], placement)
                self.assertTrue(torch.equal(batch[1], expected_order))
                self.assertEqual(batch[0].dtype, torch.float32)
                if reference is None:
                    reference = batch[0]
                else:
                    torch.testing.assert_close(batch[0], reference, rtol=0, atol=1e-6)
                cache = pipeline["cache"]
                if placement == "none":
                    self.assertEqual(cache["status"], "disabled")
                else:
                    self.assertEqual(cache["status"], "cached")
                    self.assertEqual(cache["placement"], placement)
                    self.assertEqual(cache["sample_count"], 3)

    def test_frozen_pilot_cpu_cache_keeps_inputs_and_backbone_unchanged(self):
        config = self.config(
            "frozen",
            training_cache="cpu",
            freeze_backbone=True,
            checkpoint_selection="final",
            initialize_from_run=self.root / "source",
        )
        source = build_model(
            "efficient_hccr",
            num_classes=3,
            width=4,
            stage_depths=(1, 1, 1),
            embedding_dim=8,
            reparameterize_depthwise=False,
            dropout=0,
        )
        config.initialize_from_run.mkdir()
        labels = config.initialize_from_run / "labels.json"
        labels.write_text(
            json.dumps({"labels": {"0": "一", "1": "二", "2": "三"}}), encoding="utf-8"
        )
        save_checkpoint(
            source,
            config.initialize_from_run,
            {
                "model": _model_metadata(source, config, 3),
                "manifest_digest": file_digest(self.manifest),
                "labels_digest": file_digest(labels),
            },
        )
        _, pipeline, batch, restored = self.execute(config)
        self.assertEqual(pipeline["augmentation"]["backend"], "deterministic_eval")
        self.assertEqual(pipeline["cache"]["placement"], "cpu")
        dataset = HCCRDataset(self.manifest, "train", EvalPreprocessor(16))
        self.addCleanup(dataset.close)
        expected = torch.stack([dataset[index][0] for index in batch[1].tolist()])
        torch.testing.assert_close(batch[0], expected, rtol=0, atol=0)
        self.assertEqual(
            state_digest(restored.state_dict(), exclude_classifier=True),
            state_digest(source.state_dict(), exclude_classifier=True),
        )
        _, stream_pipeline, stream_batch, _ = self.execute(
            replace(
                config,
                output_dir=self.root / "frozen-stream",
                training_cache="none",
            )
        )
        self.assertEqual(stream_pipeline["cache"]["status"], "disabled")
        self.assertTrue(torch.equal(stream_batch[1], batch[1]))
        torch.testing.assert_close(stream_batch[0], batch[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
