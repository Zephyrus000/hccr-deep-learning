from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from hccr.artifacts.checkpoints import file_digest, save_checkpoint
from hccr.data.dataset import HCCRDataset
from hccr.evaluation.compression_sensitivity import (
    analyze_compression_sensitivity,
    summarize_compression_sensitivity,
)
from hccr.models import build_model
from hccr.models.allocation import mapping_digest
from hccr.models.artifacts import load_model_from_run
from hccr.training.proposal import fit_rows_digest, pilot_budget, state_digest


class CompressionSummaryTests(unittest.TestCase):
    def test_paired_damage_and_whole_writer_bootstrap(self):
        # Writer A hurts the selected class and helps the outside class.
        # Writer B has the opposite pattern, with unequal class supports.
        records = [
            {
                "writer_id": writer,
                "source_version": "HWDB1.0",
                "class_id": c,
                "dense_correct": dense,
                "low_rank_correct": not dense,
            }
            for writer, c, dense in (
                ("A", 0, True),
                ("A", 0, True),
                ("A", 1, False),
                ("B", 0, False),
                ("B", 1, True),
                ("B", 1, True),
            )
        ]
        report = summarize_compression_sensitivity(
            iter(records),
            [0],
            ["一", "中"],
            bootstrap_repetitions=2000,
            bootstrap_seed=7,
        )
        self.assertEqual(report["groups"]["selected"]["harm"], 2)
        self.assertEqual(report["groups"]["selected"]["help"], 1)
        self.assertEqual(report["groups"]["selected"]["support"], 3)
        self.assertEqual(report["groups"]["selected"]["signed_net_damage_rate"], 1 / 3)
        self.assertEqual(report["signed_net_damage_gap"], 0)
        interval = report["bootstrap"]["intervals"]["signed_net_damage_gap"]
        self.assertEqual((interval["lower"], interval["upper"]), (-2, 2))
        self.assertEqual(interval["valid_repetitions"], 2000)
        self.assertEqual(report["concentration"]["positive_selected_share"], 0.5)
        self.assertEqual(
            report["concentration"]["signed_selected_contribution_ratio"], 0.5
        )
        self.assertEqual(report["bootstrap"]["writer_count"], 2)
        for row in records:
            if row["writer_id"] == "B":
                row["source_version"] = "HWDB1.1"
        stratified = summarize_compression_sensitivity(
            records,
            [0],
            ["一", "中"],
            bootstrap_repetitions=20,
        )
        interval = stratified["bootstrap"]["intervals"]["signed_net_damage_gap"]
        self.assertEqual((interval["lower"], interval["upper"]), (0, 0))
        self.assertEqual(stratified["bootstrap"]["singleton_strata"], 2)

    def test_support_weighting_null_denominators_and_point_only(self):
        records = [
            {
                "writer_id": "A",
                "source_version": "HWDB1.0",
                "class_id": c,
                "dense_correct": False,
                "low_rank_correct": True,
            }
            for c in (0, 1, 1)
        ]
        report = summarize_compression_sensitivity(
            records,
            [0],
            ["一", "中", "人"],
            bootstrap_repetitions=0,
        )
        self.assertEqual(report["signed_net_damage_gap"], 0)
        self.assertEqual(report["groups"]["outside"]["support"], 2)
        self.assertIsNone(report["concentration"]["positive_selected_share"])
        self.assertIsNone(report["concentration"]["signed_selected_contribution_ratio"])
        self.assertIsNone(report["bootstrap"]["intervals"])
        self.assertIsNone(report["per_class"][2]["signed_net_damage_rate"])
        no_outside = summarize_compression_sensitivity(
            records,
            [0, 1, 2],
            ["一", "中", "人"],
            bootstrap_repetitions=2,
        )
        self.assertIsNone(no_outside["signed_net_damage_gap"])
        self.assertEqual(
            no_outside["bootstrap"]["intervals"]["signed_net_damage_gap"][
                "valid_repetitions"
            ],
            0,
        )

    def test_rejects_unknown_writers_invalid_classes_or_bootstrap(self):
        row = {
            "writer_id": "A",
            "source_version": "HWDB1.0",
            "class_id": 0,
            "dense_correct": True,
            "low_rank_correct": False,
        }
        for changes in ({"writer_id": ""}, {"class_id": 2}, {"dense_correct": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summarize_compression_sensitivity(
                    [{**row, **changes}], [0], ["一", "中"]
                )
        with self.assertRaises(ValueError):
            summarize_compression_sensitivity(
                [row], [0], ["一", "中"], bootstrap_repetitions=-1
            )


class CompressionInferenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / "manifest.csv"
        self.rows = []
        for split, writer in (
            ("train", "fit"),
            ("calibration", "cal"),
            ("validation", "v1"),
            ("validation", "v2"),
            ("test", "test"),
        ):
            for c, label in enumerate(("一", "中")):
                name = f"{writer}-{c}.png"
                # Test files deliberately do not exist. No test image may be opened.
                if split != "test":
                    Image.new("L", (4, 4), 80 + 80 * c).save(self.root / name)
                self.rows.append(
                    {
                        "sample_id": name,
                        "source_file": name,
                        "writer_id": writer,
                        "class_id": str(c),
                        "unicode_label": label,
                        "split": split,
                        "official_partition": "test" if split == "test" else "train",
                        "source_version": "HWDB1.0",
                    }
                )
        self.write_manifest()
        self.policy = patch(
            "hccr.evaluation.compression_sensitivity.validate_official_source"
        )
        self.policy.start()
        self.addCleanup(self.policy.stop)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.runs, self.metadata = {}, {}
        for role in ("dense", "low_rank"):
            torch.manual_seed(7)
            model_config = {
                "name": "efficient_hccr",
                "num_classes": 2,
                "width": 4,
                "embedding_dim": 4,
                "stage_depths": [1, 1, 1],
                "classification_head": "cosface",
                "classifier_structure": role,
                "classifier_rank": 1,
                "logit_scale": 32.0,
                "angular_margin": 0.1,
            }
            model = build_model(
                "efficient_hccr",
                **{k: v for k, v in model_config.items() if k != "name"},
            )
            run = self.root / role
            run.mkdir()
            (run / "labels.json").write_text(
                json.dumps({"labels": {"0": "一", "1": "中"}}),
                encoding="utf-8",
            )
            metadata = {
                "model": model_config,
                "training": {
                    "freeze_backbone": True,
                    "seed": 7,
                    "optimizer": "adamw",
                    "weight_decay": 1e-4,
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
            backbone = state_digest(model.state_dict(), exclude_classifier=True)
            (run / "initialization.json").write_text(
                json.dumps({"backbone_digest": backbone}),
                encoding="utf-8",
            )
            self.runs[role], self.metadata[role] = run, metadata
        labels = ["一", "中"]
        provenance = {
            role: {
                "type": "metadata",
                "schema_version": 1,
                "split": "calibration",
                "official_partition": "train",
                "model_role": role,
                "class_labels": labels,
                "class_mapping_digest": mapping_digest(labels),
                "manifest_digest": file_digest(self.manifest),
                "fit_manifest_digest": fit_rows_digest(self.rows),
                "fit_writer_ids": ["fit"],
                "pilot_seed": 7,
                "head_fit_budget": pilot_budget(self.metadata[role]),
                "backbone_digest": backbone,
                "checkpoint_digest": file_digest(run / "checkpoint.pt"),
                "classifier_rank": 1 if role == "low_rank" else None,
            }
            for role, run in self.runs.items()
        }
        self.mask_payload = {
            "class_labels": labels,
            "class_mapping_digest": mapping_digest(labels),
            "manifest_digest": file_digest(self.manifest),
            "fit_manifest_digest": fit_rows_digest(self.rows),
            "residual_class_indices": [0],
            "provenance": {
                **provenance,
                "calibration_writer_ids": ["cal"],
                "prediction_file_digests": {"dense": "a" * 64, "low_rank": "b" * 64},
            },
        }
        self.mask = self.root / "mask.json"
        self.mask.write_text(json.dumps(self.mask_payload), encoding="utf-8")
        self.output = self.root / "report.json"

    def write_manifest(self):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def analyze(self, **options):
        return analyze_compression_sensitivity(
            self.runs["dense"],
            self.runs["low_rank"],
            self.mask,
            self.manifest,
            self.output,
            storage_backend="filesystem",
            **options,
        )

    def test_real_models_read_only_validation_once_with_target_free_scores(self):
        loaded, opened = [], []
        original_load = HCCRDataset.load_image

        def load(run, device):
            model = load_model_from_run(run, device)
            model.register_forward_pre_hook(
                lambda module, args: loaded.append((len(args), len(args[0])))
            )
            return model

        def image(dataset, row):
            opened.append(row["split"])
            return original_load(dataset, row)

        with (
            patch(
                "hccr.evaluation.compression_sensitivity.load_model_from_run",
                side_effect=load,
            ),
            patch.object(HCCRDataset, "load_image", image),
        ):
            report = self.analyze(batch_size=3, bootstrap_repetitions=0)
        self.assertEqual(opened, ["validation"] * 4)
        self.assertEqual(loaded, [(1, 3), (1, 3), (1, 1), (1, 1)])
        self.assertEqual(report["groups"]["overall"]["support"], 4)
        self.assertEqual(report["validation_writer_ids"], ["v1", "v2"])
        self.assertTrue(report["validation_calibration_writer_disjoint"])
        self.assertFalse(report["mask_reselected"])
        self.assertEqual(report["selected_class_ids"], [0])
        self.assertEqual(json.loads(self.output.read_text())["split"], "validation")
        self.assertEqual(len(report["validation_rows_digest"]), 64)

    def test_rejects_wrong_pair_budget_and_changed_backbone(self):
        self.mask_payload["provenance"]["dense"]["checkpoint_digest"] = "c" * 64
        self.mask.write_text(json.dumps(self.mask_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.mask_payload["provenance"]["dense"]["checkpoint_digest"] = file_digest(
            self.runs["dense"] / "checkpoint.pt"
        )
        self.mask.write_text(json.dumps(self.mask_payload), encoding="utf-8")
        metadata_path = self.runs["dense"] / "checkpoint_metadata.json"
        self.metadata["dense"]["training"]["weight_decay"] = 0.1
        metadata_path.write_text(json.dumps(self.metadata["dense"]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "budget"):
            self.analyze()
        self.metadata["dense"]["training"]["weight_decay"] = 1e-4
        metadata_path.write_text(json.dumps(self.metadata["dense"]), encoding="utf-8")
        checkpoint = self.runs["dense"] / "checkpoint.pt"
        state = torch.load(checkpoint, weights_only=True)
        key = next(k for k in state if "running_mean" in k)
        state[key].add_(1)
        torch.save(state, checkpoint)
        self.mask_payload["provenance"]["dense"]["checkpoint_digest"] = file_digest(
            checkpoint
        )
        self.mask.write_text(json.dumps(self.mask_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "backbone"):
            self.analyze()

    def test_rejects_writer_leakage_and_input_overwrite(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            analyze_compression_sensitivity(
                self.runs["dense"],
                self.runs["low_rank"],
                self.mask,
                self.manifest,
                self.mask,
            )
        for row in self.rows:
            if row["split"] == "validation":
                row["writer_id"] = "cal"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "writer leakage"):
            self.analyze()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
