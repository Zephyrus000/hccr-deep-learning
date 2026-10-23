from __future__ import annotations

import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

from hccr.artifacts.checkpoints import file_digest
from hccr.cli import build_parser
from hccr.commands.train import config_from_arguments
from hccr.models import build_model, load_model_from_run
from hccr.models.allocation import (
    load_paired_predictions,
    mapping_digest,
    select_residual_classes,
    write_prediction_file,
)
from hccr.training.proposal import (
    fit_rows_digest,
    initialize_training_model,
    manifest_class_labels,
    pilot_budget,
    resolve_residual_mask,
    state_digest,
    validate_allocation_initialization,
    validate_official_source,
)
from hccr.training.trainer import train_epoch
from hccr.training.workflow import (
    TrainingConfig,
    _model_metadata,
    _validate_training_config,
    run_training,
)


def tiny_model(structure="dense", rank=2, indices=()):
    return build_model(
        "efficient_hccr",
        num_classes=3,
        width=4,
        stage_depths=(1, 1, 1),
        embedding_dim=8,
        reparameterize_depthwise=False,
        classification_head="cosface",
        classifier_structure=structure,
        classifier_rank=rank,
        residual_class_indices=indices,
    )


def save_run(path, model, manifest, labels):
    path.mkdir()
    label_file = path / "labels.json"
    label_file.write_text(
        json.dumps({"labels": dict(enumerate(labels))}), encoding="utf-8"
    )
    config = TrainingConfig(
        manifest,
        path,
        3,
        width=4,
        stage_depths=(1, 1, 1),
        embedding_dim=8,
        reparameterize_depthwise=False,
        classifier_structure=model.classifier_structure,
        classifier_rank=model.classifier_rank,
        residual_class_indices=tuple(model.residual_class_indices),
    )
    metadata = {
        "model": _model_metadata(model, config, 3),
        "manifest_digest": file_digest(manifest),
        "labels_digest": file_digest(label_file),
    }
    (path / "checkpoint_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    torch.save(model.state_dict(), path / "checkpoint.pt")


class ProposalWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_cli_carries_structure_mask_and_frozen_pilot_settings(self):
        args = build_parser().parse_args(
            [
                "train",
                "--manifest",
                "official.csv",
                "--num-classes",
                "3755",
                "--classifier-structure",
                "dynamic_residual",
                "--classifier-rank",
                "64",
                "--residual-mask",
                "mask.json",
                "--gate-kind",
                "cosine",
                "--no-gate-use-gap",
                "--residual-regularization",
                "0.001",
                "--initialize-from-run",
                "pilot",
                "--require-official-data",
            ]
        )
        config = config_from_arguments(args)
        _validate_training_config(config)
        self.assertEqual(config.classifier_structure, "dynamic_residual")
        self.assertEqual(config.residual_mask, Path("mask.json"))
        self.assertFalse(config.gate_use_gap)
        self.assertEqual(config.residual_regularization, 0.001)
        self.assertTrue(config.require_official_data)

    def test_frozen_training_updates_head_but_preserves_every_backbone_tensor(self):
        model = tiny_model("dynamic_residual", indices=(0, 2))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("classifier."))
        before = state_digest(model.state_dict(), exclude_classifier=True)
        head_before = state_digest(model.classifier.state_dict())
        optimizer = torch.optim.SGD(
            (p for p in model.parameters() if p.requires_grad), lr=0.01
        )
        result = train_epoch(
            model,
            [(torch.rand(3, 1, 16, 16), torch.arange(3), ["", "", ""])],
            optimizer,
            "cpu",
            nn.CrossEntropyLoss(),
            freeze_backbone=True,
            residual_regularization=1e-4,
            gate_regularization=1e-4,
        )
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(
            state_digest(model.state_dict(), exclude_classifier=True), before
        )
        self.assertNotEqual(state_digest(model.classifier.state_dict()), head_before)
        self.assertTrue(torch.isfinite(model.classifier.residuals).all())

    def test_teacher_to_low_rank_to_residual_restores_exact_shared_factors(self):
        labels = ["一", "二", "三"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            manifest.write_text("fixed-source-manifest", encoding="utf-8")
            dense = tiny_model()
            save_run(root / "dense", dense, manifest, labels)
            low = tiny_model("low_rank")
            info = initialize_training_model(low, root / "dense", manifest, labels)
            self.assertEqual(info["mode"], "normalized_dense_svd")
            self.assertEqual(
                state_digest(low.state_dict(), exclude_classifier=True),
                state_digest(dense.state_dict(), exclude_classifier=True),
            )
            save_run(root / "low", low, manifest, labels)
            static = tiny_model("static_residual", indices=(0, 2))
            dynamic = tiny_model("dynamic_residual", indices=(0, 2))
            for candidate in (static, dynamic):
                info = initialize_training_model(
                    candidate, root / "low", manifest, labels
                )
                self.assertEqual(info["mode"], "shared_low_rank_pilot")
                torch.testing.assert_close(
                    candidate.classifier.coefficients,
                    low.classifier.coefficients,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    candidate.classifier.basis, low.classifier.basis, rtol=0, atol=0
                )
                self.assertEqual(
                    candidate.classifier.residuals.count_nonzero().item(), 0
                )
            sample = torch.rand(2, 1, 16, 16)
            with torch.inference_mode():
                torch.testing.assert_close(
                    static.eval()(sample), dynamic.eval()(sample)
                )
            with self.assertRaisesRegex(ValueError, "rank"):
                initialize_training_model(
                    tiny_model("static_residual", rank=1, indices=(0,)),
                    root / "low",
                    manifest,
                    labels,
                )
            with self.assertRaisesRegex(ValueError, "class mapping"):
                initialize_training_model(
                    tiny_model("low_rank"), root / "dense", manifest, labels[::-1]
                )

    def test_frozen_low_rank_pilot_preserves_inactive_parameters_and_backbone(self):
        labels = ["一", "二", "三"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            rows = []
            for split in ("train", "validation"):
                for class_id, label in enumerate(labels):
                    name = f"{split}-{class_id}.png"
                    image = Image.new("L", (12, 12), color=255)
                    image.putpixel((class_id + 3, 4), 0)
                    image.save(root / name)
                    rows.append(
                        dict(
                            sample_id=name,
                            source_file=name,
                            writer_id=split,
                            unicode_label=label,
                            class_id=class_id,
                            split=split,
                        )
                    )
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            save_run(root / "teacher", tiny_model(), manifest, labels)
            observed = {}

            def observe_before_ddp(model, context):
                self.assertFalse(context.enabled)
                self.assertFalse(model.classifier.residuals.requires_grad)
                self.assertEqual(model.classifier.residuals.numel(), 0)
                self.assertTrue(model.classifier.coefficients.requires_grad)
                self.assertTrue(model.classifier.basis.requires_grad)
                observed["model"] = model
                observed["backbone"] = state_digest(
                    model.state_dict(), exclude_classifier=True
                )
                observed["coefficients"] = (
                    model.classifier.coefficients.detach().clone()
                )
                observed["basis"] = model.classifier.basis.detach().clone()
                return model

            config = TrainingConfig(
                manifest,
                root / "runs",
                3,
                epochs=1,
                batch_size=3,
                reference_batch_size=3,
                optimizer_step_policy="configured_epochs",
                learning_rate_scaling="none",
                width=4,
                stage_depths=(1, 1, 1),
                embedding_dim=8,
                reparameterize_depthwise=False,
                classifier_structure="low_rank",
                classifier_rank=2,
                initialize_from_run=root / "teacher",
                freeze_backbone=True,
                checkpoint_selection="final",
                image_size=16,
                device="cpu",
                evaluation_policy="validation_only",
                scheduler="none",
                early_stopping_patience=None,
                benchmark_warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
            )
            with patch(
                "hccr.training.workflow.wrap_model_for_training",
                side_effect=observe_before_ddp,
            ):
                metrics = run_training(config)
            self.assertIn("top1", metrics)
            model = observed["model"]
            self.assertFalse(model.classifier.residuals.requires_grad)
            self.assertIsNone(model.classifier.residuals.grad)
            self.assertEqual(
                state_digest(model.state_dict(), exclude_classifier=True),
                observed["backbone"],
            )
            self.assertFalse(
                torch.equal(model.classifier.coefficients, observed["coefficients"])
            )
            self.assertFalse(torch.equal(model.classifier.basis, observed["basis"]))

    def test_official_guard_rejects_image_export_without_writer_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            path.write_text("old-export", encoding="utf-8")
            (path.parent / "audit_report.json").write_text(
                json.dumps({"source_kind": "png", "manifest_digest": file_digest(path)})
            )
            with self.assertRaisesRegex(ValueError, "GNT"):
                validate_official_source(path, [])

    def test_allocation_binds_the_actual_source_checkpoint_and_pilot_protocol(self):
        labels = ["一", "二", "三"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            manifest.write_text("frozen-source", encoding="utf-8")
            source_run = root / "low_rank"
            source = tiny_model("low_rank")
            save_run(source_run, source, manifest, labels)
            metadata_path = source_run / "checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "training": {
                        "seed": 7,
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
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            initialization = initialize_training_model(
                tiny_model("dynamic_residual", indices=(0, 2)),
                source_run,
                manifest,
                labels,
            )
            allocation = {
                "provenance": {
                    "low_rank": {
                        "checkpoint_digest": file_digest(source_run / "checkpoint.pt"),
                        "backbone_digest": state_digest(
                            source.state_dict(), exclude_classifier=True
                        ),
                        "classifier_rank": 2,
                        "manifest_digest": file_digest(manifest),
                        "pilot_seed": 7,
                        "head_fit_budget": pilot_budget(metadata),
                    }
                }
            }
            validate_allocation_initialization(allocation, source_run, initialization)
            for field in ("checkpoint_digest", "backbone_digest"):
                changed = {**initialization, field: "0" * 64}
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, field.split("_")[0]),
                ):
                    validate_allocation_initialization(allocation, source_run, changed)
            for section, field, value, error in (
                ("model", "classifier_structure", "dense", "low-rank"),
                ("model", "classifier_rank", 1, "low-rank"),
                ("training", "seed", 17, "seed"),
                ("training", "weight_decay", 0.01, "budget"),
                ("training", "completed_optimizer_steps", 9, "complete"),
            ):
                changed = deepcopy(metadata)
                changed[section][field] = value
                metadata_path.write_text(json.dumps(changed), encoding="utf-8")
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    validate_allocation_initialization(
                        allocation, source_run, initialization
                    )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            checkpoint = source_run / "checkpoint.pt"
            state = torch.load(checkpoint, weights_only=True)
            state["classifier.basis"] += 1
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_allocation_initialization(
                    allocation, source_run, initialization
                )

    def test_dynamic_training_writes_reloadable_checkpoint_and_actual_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            rows = []
            for split in ("train", "calibration", "validation", "test"):
                for class_id, label in enumerate(("一", "二", "三")):
                    name = f"{split}-{class_id}.png"
                    image = Image.new("L", (12, 12), color=255)
                    image.putpixel((class_id + 3, 4), 0)
                    image.save(root / name)
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
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = TrainingConfig(
                manifest,
                root / "runs",
                3,
                epochs=1,
                batch_size=3,
                reference_batch_size=3,
                optimizer_step_policy="configured_epochs",
                learning_rate_scaling="none",
                width=4,
                stage_depths=(1, 1, 1),
                embedding_dim=8,
                reparameterize_depthwise=False,
                classifier_structure="dynamic_residual",
                classifier_rank=2,
                residual_class_indices=(0, 2),
                image_size=16,
                device="cpu",
                residual_regularization=1e-4,
                gate_regularization=1e-4,
                evaluation_policy="validation_only",
                scheduler="none",
                early_stopping_patience=None,
                benchmark_warmup_iterations=1,
                benchmark_iterations=1,
                benchmark_repetitions=1,
            )
            metrics = run_training(config)
            self.assertIn("top1", metrics)
            run = next((root / "runs").glob("20*"))
            metadata = json.loads((run / "checkpoint_metadata.json").read_text())
            self.assertEqual(metadata["training"]["completed_optimizer_steps"], 1)
            self.assertEqual(metadata["selection"]["epoch"], 1)
            self.assertEqual(metadata["test"]["evaluation_policy"], "validation_only")
            restored = load_model_from_run(run).eval()
            self.assertEqual(restored.classifier_structure, "dynamic_residual")
            self.assertGreater(restored.classifier.residuals.count_nonzero().item(), 0)
            with torch.inference_mode():
                self.assertTrue(
                    torch.isfinite(restored(torch.ones(1, 1, 16, 16))).all()
                )
            profile = json.loads((run / "resource_profile.json").read_text())
            self.assertEqual(profile["classifier_structure"], "dynamic_residual")
            self.assertEqual(
                profile["estimated_classifier_macs"],
                restored.classifier.inference_macs(),
            )

    def test_allocation_mapping_and_writer_provenance_are_checked(self):
        labels = ["一", "二", "三"]
        rows = [
            {
                "class_id": str(c),
                "unicode_label": label,
                "split": split,
                "writer_id": writer,
                "sample_id": f"{split}-{c}",
            }
            for split, writer in (("train", "fit-a"), ("calibration", "cal-a"))
            for c, label in enumerate(labels)
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            manifest.write_text("frozen", encoding="utf-8")
            path = root / "mask.json"
            pilot = {
                "type": "metadata",
                "schema_version": 1,
                "split": "calibration",
                "official_partition": "train",
                "fit_writer_ids": ["fit-a"],
                "class_labels": labels,
                "class_mapping_digest": mapping_digest(labels),
                "manifest_digest": file_digest(manifest),
                "fit_manifest_digest": fit_rows_digest(rows),
                "backbone_digest": "a" * 64,
                "pilot_seed": 7,
                "head_fit_budget": pilot_budget(
                    {
                        "training": {
                            "seed": 7,
                            "freeze_backbone": True,
                            "optimizer": "adamw",
                            "weight_decay": 0.0001,
                            "completed_optimizer_steps": 10,
                            "total_optimizer_steps": 10,
                            "resolved_epochs": 1,
                        },
                        "selection": {"epoch": 1, "policy": "final"},
                        "model": {
                            "classification_head": "cosface",
                            "logit_scale": 32.0,
                            "angular_margin": 0.1,
                        },
                        "preprocess": {"image_size": 16, "margin": 2},
                    }
                ),
            }
            predictions = [
                {
                    "sample_id": f"calibration-{c}",
                    "writer_id": "cal-a",
                    "class_id": c,
                    "prediction": c,
                }
                for c in range(len(labels))
            ]
            for role in ("dense", "low_rank"):
                write_prediction_file(
                    root / f"{role}.jsonl",
                    {
                        **pilot,
                        "model_role": role,
                        "checkpoint_digest": ("d" if role == "dense" else "b") * 64,
                        "classifier_rank": 2 if role == "low_rank" else None,
                    },
                    predictions,
                )
            low_path = root / "low_rank.jsonl"
            mask = select_residual_classes(
                load_paired_predictions(root / "dense.jsonl", low_path), 2
            )
            path.write_text(json.dumps(mask), encoding="utf-8")
            self.assertEqual(
                resolve_residual_mask(path, manifest, rows)["residual_class_indices"],
                [0, 2],
            )
            mask["provenance"]["calibration_writer_ids"] = ["test-writer"]
            path.write_text(json.dumps(mask), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration writers"):
                resolve_residual_mask(path, manifest, rows)
            mask["provenance"]["calibration_writer_ids"] = ["cal-a"]
            for keys, value, error in (
                (("provenance", "dense", "type"), None, "metadata"),
                (("provenance", "low_rank", "model_role"), "dense", "roles"),
                (("provenance", "low_rank", "classifier_rank"), 0, "rank"),
                (("provenance", "low_rank", "backbone_digest"), "c" * 64, "backbone"),
                (("provenance", "low_rank", "checkpoint_digest"), "claimed", "SHA-256"),
                (("provenance", "dense", "manifest_digest"), "c" * 64, "manifest"),
                (("provenance", "dense", "fit_manifest_digest"), "c" * 64, "manifest"),
                (("provenance", "dense", "fit_writer_ids"), ["wrong"], "fit writers"),
                (("provenance", "dense", "pilot_seed"), 17, "seed"),
                (
                    (
                        "provenance",
                        "dense",
                        "head_fit_budget",
                        "completed_optimizer_steps",
                    ),
                    9,
                    "complete",
                ),
                (
                    ("provenance", "dense", "head_fit_budget", "freeze_backbone"),
                    False,
                    "frozen",
                ),
                (
                    (
                        "provenance",
                        "dense",
                        "head_fit_budget",
                        "checkpoint_selection",
                        "policy",
                    ),
                    "best_validation",
                    "final",
                ),
                (
                    ("provenance", "dense", "head_fit_budget", "max_train_samples"),
                    1,
                    "full",
                ),
                (
                    ("provenance", "dense", "head_fit_budget", "preprocess", "margin"),
                    8,
                    "preprocessing",
                ),
                (
                    ("provenance", "dense", "head_fit_budget", "weight_decay"),
                    0.01,
                    "head_fit_budget",
                ),
                (
                    ("provenance", "prediction_file_digests", "dense"),
                    "claimed",
                    "SHA-256",
                ),
            ):
                changed = deepcopy(mask)
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, error):
                    resolve_residual_mask(path, manifest, rows)
            path.write_text(json.dumps(mask), encoding="utf-8")
            altered_rows = deepcopy(rows)
            altered_rows[0]["sample_id"] = "different-fit-record"
            with self.assertRaisesRegex(ValueError, "fit rows"):
                resolve_residual_mask(path, manifest, altered_rows)
        self.assertEqual(manifest_class_labels(rows), labels)


if __name__ == "__main__":
    unittest.main()
