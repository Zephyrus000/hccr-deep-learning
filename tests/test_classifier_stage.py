from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from scripts.prepare_classifier_stage import prepare_classifier_stage

from hccr.artifacts.checkpoints import file_digest
from hccr.cli import build_parser
from hccr.commands.train import config_from_arguments
from hccr.experiments.runner import build_jobs
from hccr.experiments.spec import ExperimentSpec, ExperimentVariant
from hccr.models import build_model
from hccr.models.allocation import mapping_digest
from hccr.training.proposal import fit_rows_digest, pilot_budget, state_digest
from hccr.training.workflow import _validate_training_config


def jobs_for(config):
    return build_jobs(
        ExperimentSpec(
            experiment_id=config["experiment_id"],
            manifest=Path(config["manifest"]),
            output_dir=Path(config["output_dir"]),
            seeds=tuple(config["seeds"]),
            base_args=config["base_args"],
            variants=tuple(
                ExperimentVariant(v["name"], v["args"]) for v in config["variants"]
            ),
        )
    )


class ClassifierStageTests(unittest.TestCase):
    def test_pilots_keep_teacher_identity_and_have_equal_frozen_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            manifest.write_text("frozen", encoding="utf-8")
            source = root / "teacher"
            source.mkdir()
            (source / "checkpoint.pt").write_bytes(b"teacher")
            (source / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "manifest_digest": file_digest(manifest),
                        "model": {
                            "name": "efficient_hccr",
                            "num_classes": 3755,
                            "embedding_dim": 320,
                            "classification_head": "cosface",
                            "classifier_structure": "dense",
                            "gate_use_gap": False,
                            "gate_detach_gap": False,
                        },
                        "training": {"seed": 17, "batch_size": 256},
                        "preprocess": {"image_size": 96},
                    }
                ),
                encoding="utf-8",
            )
            config = prepare_classifier_stage("pilots", source, manifest)
            jobs = jobs_for(config)
            self.assertEqual(len(jobs), 2)
            parsed = [
                config_from_arguments(
                    build_parser().parse_args(
                        list(job.command[job.command.index("train") :])
                    )
                )
                for job in jobs
            ]
            for candidate in parsed:
                _validate_training_config(candidate)
                self.assertTrue(candidate.freeze_backbone)
                self.assertEqual(candidate.initialize_from_run, source)
                self.assertEqual(candidate.checkpoint_selection, "final")
                self.assertIsNone(candidate.early_stopping_patience)
                self.assertEqual(candidate.seed, 17)
                self.assertFalse(candidate.gate_use_gap)
                self.assertFalse(candidate.gate_detach_gap)
            with self.assertRaisesRegex(ValueError, "seed"):
                prepare_classifier_stage("pilots", source, manifest, seed=29)
            with self.assertRaisesRegex(ValueError, "rank"):
                prepare_classifier_stage("pilots", source, manifest, rank=1000)
            budget_config = prepare_classifier_stage(
                "pilots", source, manifest, near_budget_k=500
            )
            self.assertEqual(len(jobs_for(budget_config)), 3)
            self.assertEqual(
                budget_config["variants"][-1]["args"]["classifier_rank"], 104
            )
            self.assertNotEqual(config["experiment_id"], budget_config["experiment_id"])
            other_rank = prepare_classifier_stage("pilots", source, manifest, rank=32)
            self.assertNotEqual(config["experiment_id"], other_rank["experiment_id"])

    def test_residual_configs_include_references_and_do_not_collide_across_masks(self):
        labels = list("一二三四五六七")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                {
                    "class_id": str(i),
                    "unicode_label": label,
                    "writer_id": writer,
                    "split": split,
                    "sample_id": f"{split}-{i}",
                    "source_file": f"{split}-{i}.gnt",
                }
                for split, writer in (("train", "fit"), ("calibration", "cal"))
                for i, label in enumerate(labels)
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            sources, stored_metadata = {}, {}
            near_model = build_model(
                "efficient_hccr",
                num_classes=7,
                width=4,
                stage_depths=(1, 1, 1),
                backbone_output_channels=8,
                embedding_dim=8,
                classification_head="cosface",
                classifier_structure="low_rank",
                classifier_rank=4,
            )
            backbone_digest = state_digest(
                near_model.state_dict(), exclude_classifier=True
            )
            for role in ("dense", "low_rank", "near_budget"):
                path = root / role
                path.mkdir()
                if role == "near_budget":
                    torch.save(near_model.state_dict(), path / "checkpoint.pt")
                else:
                    (path / "checkpoint.pt").write_bytes(role.encode())
                metadata = {
                    "manifest_digest": file_digest(manifest),
                    "model": {
                        "name": "efficient_hccr",
                        "num_classes": 7,
                        "embedding_dim": 8,
                        "width": 4,
                        "stage_depths": [1, 1, 1],
                        "backbone_output_channels": 8,
                        "classification_head": "cosface",
                        "classifier_structure": "dense"
                        if role == "dense"
                        else "low_rank",
                        "classifier_rank": 4 if role == "near_budget" else 2,
                        "logit_scale": 32.0,
                        "angular_margin": 0.1,
                    },
                    "training": {
                        "seed": 7,
                        "batch_size": 3,
                        "freeze_backbone": True,
                        "optimizer": "adamw",
                        "weight_decay": 0.0001,
                        "completed_optimizer_steps": 10,
                        "total_optimizer_steps": 10,
                        "resolved_epochs": 1,
                    },
                    "selection": {"epoch": 1, "policy": "final"},
                    "preprocess": {"image_size": 16, "margin": 2},
                }
                (path / "checkpoint_metadata.json").write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                sources[role] = path
                stored_metadata[role] = metadata
            pilot = {
                "type": "metadata",
                "schema_version": 1,
                "split": "calibration",
                "official_partition": "train",
                "fit_writer_ids": ["fit"],
                "class_labels": labels,
                "class_mapping_digest": mapping_digest(labels),
                "manifest_digest": file_digest(manifest),
                "fit_manifest_digest": fit_rows_digest(rows),
                "backbone_digest": backbone_digest,
                "head_fit_budget": pilot_budget(stored_metadata["dense"]),
                "pilot_seed": 7,
            }
            mask = {
                "class_labels": labels,
                "class_mapping_digest": mapping_digest(labels),
                "manifest_digest": file_digest(manifest),
                "fit_manifest_digest": fit_rows_digest(rows),
                "residual_class_indices": [0],
                "provenance": {
                    "dense": {
                        **pilot,
                        "model_role": "dense",
                        "classifier_rank": None,
                        "checkpoint_digest": file_digest(
                            sources["dense"] / "checkpoint.pt"
                        ),
                    },
                    "low_rank": {
                        **pilot,
                        "model_role": "low_rank",
                        "classifier_rank": 2,
                        "checkpoint_digest": file_digest(
                            sources["low_rank"] / "checkpoint.pt"
                        ),
                    },
                    "calibration_writer_ids": ["cal"],
                    "prediction_file_digests": {
                        "dense": "a" * 64,
                        "low_rank": "b" * 64,
                    },
                },
            }
            mask_path = root / "mask.json"
            mask_path.write_text(json.dumps(mask), encoding="utf-8")
            config = prepare_classifier_stage(
                "residuals",
                sources["low_rank"],
                manifest,
                mask=mask_path,
                dense_pilot_run=sources["dense"],
                near_budget_pilot_run=sources["near_budget"],
            )
            jobs = jobs_for(config)
            self.assertEqual(len(jobs), 5)
            self.assertEqual(
                config["variants"][-1]["args"]["initialize_from_run"],
                str(sources["near_budget"]),
            )
            for job in jobs:
                candidate = config_from_arguments(
                    build_parser().parse_args(
                        list(job.command[job.command.index("train") :])
                    )
                )
                _validate_training_config(candidate)
                self.assertFalse(candidate.freeze_backbone)
                self.assertEqual(candidate.epochs, 30)
            mask["residual_class_indices"] = [1]
            mask_path.write_text(json.dumps(mask), encoding="utf-8")
            other = prepare_classifier_stage(
                "residuals",
                sources["low_rank"],
                manifest,
                mask=mask_path,
                dense_pilot_run=sources["dense"],
                near_budget_pilot_run=sources["near_budget"],
            )
            self.assertNotEqual(config["experiment_id"], other["experiment_id"])
            stored_metadata["near_budget"]["training"]["total_optimizer_steps"] = 11
            (sources["near_budget"] / "checkpoint_metadata.json").write_text(
                json.dumps(stored_metadata["near_budget"]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "budget"):
                prepare_classifier_stage(
                    "residuals",
                    sources["low_rank"],
                    manifest,
                    mask=mask_path,
                    near_budget_pilot_run=sources["near_budget"],
                )


if __name__ == "__main__":
    unittest.main()
