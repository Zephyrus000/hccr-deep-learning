from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from hccr.artifacts.checkpoints import file_digest, save_checkpoint
from hccr.data.casia_manifest import build_casia_manifest
from hccr.data.dataset import HCCRDataset
from hccr.data.manifest import read_manifest
from hccr.models import build_model
from hccr.models.allocation import load_paired_predictions, select_residual_classes
from hccr.models.artifacts import load_model_from_run
from hccr.preprocessing import EvalPreprocessor
from hccr.training.proposal import state_digest
from hccr.training.proposal_predictions import (
    export_calibration_predictions,
    fit_rows_digest,
)


class ProposalPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for split in ("train", "calibration", "test"):
            for class_id, label in enumerate(("一", "中")):
                name = f"{split}-{class_id}.png"
                Image.new("L", (4, 4), 60 + class_id * 100).save(self.root / name)
                self.rows.append(
                    {
                        "sample_id": name,
                        "source_file": name,
                        "writer_id": split,
                        "unicode_label": label,
                        "class_id": str(class_id),
                        "split": split,
                        "official_partition": "test" if split == "test" else "train",
                    }
                )
        self.write_manifest()
        self.policy = patch(
            "hccr.training.proposal_predictions.validate_official_source"
        )
        self.policy.start()
        self.addCleanup(self.policy.stop)
        self.dense = self.make_run("dense")
        self.low = self.make_run("low_rank")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def write_manifest(self) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def make_run(self, structure: str) -> Path:
        run = self.root / structure
        run.mkdir()
        config = {
            "name": "efficient_hccr",
            "num_classes": 2,
            "width": 4,
            "embedding_dim": 4,
            "stage_depths": [1, 1, 1],
            "classification_head": "cosface",
            "classifier_structure": structure,
            "classifier_rank": 1,
            "logit_scale": 32.0,
            "angular_margin": 0.1,
        }
        torch.manual_seed(7)
        model = build_model(
            "efficient_hccr",
            **{key: value for key, value in config.items() if key != "name"},
        )
        (run / "labels.json").write_text(
            json.dumps({"labels": {"0": "一", "1": "中"}}), encoding="utf-8"
        )
        metadata = {
            "model": config,
            "training": {
                "freeze_backbone": True,
                "seed": 7,
                "optimizer": "adamw",
                "weight_decay": 0.0001,
                "completed_optimizer_steps": 2,
                "total_optimizer_steps": 2,
                "resolved_epochs": 1,
            },
            "selection": {"epoch": 1, "policy": "final"},
            "preprocess": {"image_size": 16, "margin": 2},
            "manifest_digest": file_digest(self.manifest),
            "labels_digest": file_digest(run / "labels.json"),
        }
        save_checkpoint(model, run, metadata)
        (run / "initialization.json").write_text(
            json.dumps(
                {
                    "backbone_digest": state_digest(
                        model.state_dict(), exclude_classifier=True
                    ),
                }
            ),
            encoding="utf-8",
        )
        return run

    def edit_metadata(self, section: str, **values) -> None:
        path = self.dense / "checkpoint_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata[section].update(values)
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_real_checkpoint_dataset_export_pairs_and_controls(self) -> None:
        frequency, direction = (
            self.root / "frequency.json",
            self.root / "direction.json",
        )
        metadata = export_calibration_predictions(
            self.dense,
            self.manifest,
            self.root / "dense.jsonl",
            batch_size=1,
            fit_frequency_output=frequency,
            directional_output=direction,
            directional_rank=1,
        )
        export_calibration_predictions(
            self.low, self.manifest, self.root / "low.jsonl", batch_size=2
        )
        paired = load_paired_predictions(
            self.root / "dense.jsonl", self.root / "low.jsonl"
        )
        self.assertEqual(set(paired.dense), {"calibration-0.png", "calibration-1.png"})
        self.assertEqual(
            metadata["fit_manifest_digest"], fit_rows_digest(list(reversed(self.rows)))
        )
        self.assertEqual(
            metadata["backbone_digest"], paired.low_rank_metadata["backbone_digest"]
        )
        self.assertEqual(paired.low_rank_metadata["classifier_rank"], 1)
        self.assertEqual(
            len(select_residual_classes(paired, 1)["residual_class_indices"]), 1
        )
        self.assertEqual(json.loads(frequency.read_text())["values"], [1, 1])
        self.assertEqual(
            json.loads(frequency.read_text())["manifest_digest"],
            fit_rows_digest(self.rows),
        )
        self.assertEqual(len(json.loads(direction.read_text())["values"]), 2)

    def test_requires_final_complete_frozen_pilot(self) -> None:
        for section, values in (
            ("training", {"freeze_backbone": False}),
            ("training", {"completed_optimizer_steps": 1}),
            ("training", {"max_train_samples": 1}),
            ("selection", {"policy": "best"}),
        ):
            path = self.dense / "checkpoint_metadata.json"
            original = path.read_bytes()
            self.edit_metadata(section, **values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                export_calibration_predictions(
                    self.dense, self.manifest, self.root / "bad.jsonl"
                )
            path.write_bytes(original)
        self.assertFalse((self.root / "bad.jsonl").exists())

    def test_rejects_changed_manifest_or_labels(self) -> None:
        self.manifest.write_text(self.manifest.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            export_calibration_predictions(
                self.dense, self.manifest, self.root / "bad.jsonl"
            )
        self.write_manifest()
        (self.dense / "labels.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "labels digest"):
            export_calibration_predictions(
                self.dense, self.manifest, self.root / "bad.jsonl"
            )

    def test_rejects_changed_frozen_backbone_buffers(self) -> None:
        path = self.dense / "checkpoint.pt"
        state = torch.load(path, weights_only=True)
        key = next(key for key in state if "running_mean" in key)
        state[key] += 1
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "backbone differs"):
            export_calibration_predictions(
                self.dense, self.manifest, self.root / "bad.jsonl"
            )

    def test_rejects_overwriting_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "destinations"):
            export_calibration_predictions(self.dense, self.manifest, self.manifest)
        source = self.root / "calibration-0.png"
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "image inputs"):
            export_calibration_predictions(self.dense, self.manifest, source)
        self.assertEqual(source.read_bytes(), original)

    def test_predictions_use_saved_preprocessing_and_target_free_scores(self) -> None:
        output = self.root / "predictions.jsonl"
        model = load_model_from_run(self.dense).eval()
        dataset = HCCRDataset(self.manifest, "calibration", EvalPreprocessor(16, 2))
        self.addCleanup(dataset.close)
        with torch.inference_mode():
            expected = {
                row["sample_id"]: int(model(dataset[index][0][None]).argmax(1))
                for index, row in enumerate(dataset.rows)
            }
        observed_shapes = []
        handle = model.register_forward_pre_hook(
            lambda module, args: observed_shapes.append(tuple(args[0].shape))
        )
        self.addCleanup(handle.remove)
        with patch(
            "hccr.training.proposal_predictions.load_model_from_run", return_value=model
        ):
            export_calibration_predictions(self.dense, self.manifest, output)
        actual = {
            row["sample_id"]: row["prediction"]
            for row in map(
                json.loads, output.read_text(encoding="utf-8").splitlines()[1:]
            )
        }
        self.assertEqual(actual, expected)
        self.assertEqual(observed_shapes, [(2, 1, 16, 16)])

    def test_invalid_directional_rank_preserves_existing_output(self) -> None:
        output = self.root / "predictions.jsonl"
        output.write_text("existing", encoding="utf-8")
        for rank in (0, 5, True, 1.5):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                export_calibration_predictions(
                    self.dense,
                    self.manifest,
                    output,
                    directional_output=self.root / "direction.json",
                    directional_rank=rank,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")
        self.assertFalse((self.root / "direction.json").exists())

    def test_pairing_rejects_different_fit_update_budgets(self) -> None:
        export_calibration_predictions(self.low, self.manifest, self.root / "low.jsonl")
        self.edit_metadata(
            "training", completed_optimizer_steps=3, total_optimizer_steps=3
        )
        export_calibration_predictions(
            self.dense, self.manifest, self.root / "dense.jsonl"
        )
        with self.assertRaisesRegex(ValueError, "head_fit_budget"):
            load_paired_predictions(self.root / "dense.jsonl", self.root / "low.jsonl")


class OfficialGNTExportTests(unittest.TestCase):
    def test_full_official_vocabulary_uses_only_calibration_gnt_records(self) -> None:
        # Use the canonical 3755-class endpoints so no official validator is mocked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [
                bytes((lead, trail)).decode("gb2312")
                for lead in range(0xB0, 0xD8)
                for trail in range(0xA1, 0xFF if lead < 0xD7 else 0xFA)
            ]
            train, test = [], []
            identity = 0
            for version_index, version in enumerate(("HWDB1.0", "HWDB1.1")):
                for partition, writers, paths in (
                    ("Train", range(3), train),
                    ("Test", range(3, 4), test),
                ):
                    source = root / f"{version}_{partition}"
                    source.mkdir()
                    paths.append(source)
                    for writer in writers:
                        writer_id = 1000 + 100 * version_index + writer
                        (source / f"{writer_id}-f.gnt").write_bytes(
                            b"".join(
                                struct.pack(
                                    "<I2sHHI",
                                    14,
                                    label.encode("gb18030"),
                                    2,
                                    2,
                                    identity + index,
                                )
                                for index, label in enumerate(labels)
                            )
                        )
                        identity += len(labels)
            icdar = root / "ICDAR2013_Test"
            icdar.mkdir()
            (icdar / "C001-f-f.gnt").write_bytes(
                b"".join(
                    struct.pack(
                        "<I2sHHI", 14, label.encode("gb18030"), 2, 2, identity + index
                    )
                    for index, label in enumerate(labels)
                )
            )
            output_dir = root / "processed" / "official"
            build_casia_manifest(
                train,
                test,
                root,
                output_dir,
                protocol="hwdb1.0-1.1-official",
                label_set="gb2312-level1",
                audit_pixel_duplicates=True,
                icdar_dir=icdar,
            )
            manifest = output_dir / "manifest.csv"
            rows = read_manifest(manifest)
            ordered_labels = sorted(labels)
            run = root / "pilot"
            run.mkdir()
            (run / "labels.json").write_text(
                json.dumps(
                    {
                        "labels": {
                            str(index): label
                            for index, label in enumerate(ordered_labels)
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "name": "efficient_hccr",
                "num_classes": 3755,
                "width": 4,
                "embedding_dim": 4,
                "stage_depths": [1, 1, 1],
                "classification_head": "cosface",
                "classifier_structure": "dense",
                "logit_scale": 32.0,
                "angular_margin": 0.1,
            }
            model = build_model(**config)
            save_checkpoint(
                model,
                run,
                {
                    "model": config,
                    "manifest_digest": file_digest(manifest),
                    "labels_digest": file_digest(run / "labels.json"),
                    "preprocess": {"image_size": 16, "margin": 2},
                    "training": {
                        "freeze_backbone": True,
                        "seed": 7,
                        "optimizer": "adamw",
                        "weight_decay": 0.0001,
                        "resolved_epochs": 1,
                        "completed_optimizer_steps": 1,
                        "total_optimizer_steps": 1,
                    },
                    "selection": {"policy": "final", "epoch": 1},
                },
            )
            (run / "initialization.json").write_text(
                json.dumps(
                    {
                        "backbone_digest": state_digest(
                            model.state_dict(), exclude_classifier=True
                        )
                    }
                ),
                encoding="utf-8",
            )
            # A corrupt non-calibration source must never be read by inference.
            excluded = next(row for row in rows if row["split"] == "icdar")
            (root / excluded["source_file"]).unlink()
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                export_calibration_predictions(
                    run,
                    manifest,
                    root / "predictions.jsonl",
                    batch_size=512,
                    storage_backend="filesystem",
                )
            finally:
                torch.set_num_threads(previous_threads)
            contents = list(
                map(
                    json.loads,
                    (root / "predictions.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines(),
                )
            )
            calibration = {
                row["sample_id"] for row in rows if row["split"] == "calibration"
            }
            self.assertEqual(len(calibration), 7510)
            self.assertEqual({row["sample_id"] for row in contents[1:]}, calibration)
            self.assertEqual(contents[0]["class_labels"], ordered_labels)
            self.assertEqual(
                contents[0]["checkpoint_digest"], file_digest(run / "checkpoint.pt")
            )


if __name__ == "__main__":
    unittest.main()
