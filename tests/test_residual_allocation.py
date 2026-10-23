from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from hccr.models.allocation import (
    directional_svd_errors,
    load_paired_predictions,
    mapping_digest,
    select_residual_classes,
    write_prediction_file,
)


class ResidualAllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = ["木", "一", "二"]
        self.metadata = {
            "type": "metadata",
            "schema_version": 1,
            "split": "calibration",
            "official_partition": "train",
            "class_labels": self.labels,
            "class_mapping_digest": mapping_digest(self.labels),
            "manifest_digest": "calibration-sha",
            "fit_manifest_digest": "fit-sha",
            "fit_writer_ids": ["fit-1"],
            "backbone_digest": "shared-backbone",
            "head_fit_budget": {"steps": 10, "optimizer": "adamw"},
            "pilot_seed": 7,
            "model_role": "dense",
            "checkpoint_digest": "dense-sha",
        }
        self.dense = [
            {"sample_id": "a", "writer_id": "cal-1", "class_id": 0, "prediction": 0},
            {"sample_id": "b", "writer_id": "cal-2", "class_id": 0, "prediction": 1},
            {"sample_id": "c", "writer_id": "cal-1", "class_id": 1, "prediction": 1},
            {"sample_id": "d", "writer_id": "cal-1", "class_id": 2, "prediction": 2},
        ]
        self.low = [dict(row) for row in self.dense]
        self.low[0]["prediction"] = 1
        self.low[1]["prediction"] = 0
        self.low[2]["prediction"] = 2

    def paired(self):
        dense_path, low_path = self.root / "dense.jsonl", self.root / "low.jsonl"
        write_prediction_file(dense_path, self.metadata, self.dense)
        write_prediction_file(
            low_path,
            {**self.metadata, "model_role": "low_rank", "checkpoint_digest": "low-sha"},
            reversed(self.low),
        )
        return load_paired_predictions(dense_path, low_path)

    def test_paired_discordance_and_unicode_ties(self) -> None:
        mask = select_residual_classes(self.paired(), 2)
        self.assertEqual(mask["selection_order"], [1, 2])
        self.assertEqual(mask["residual_class_indices"], [1, 2])
        self.assertEqual(mask["zero_penalty_selected"], [2])
        stats = mask["class_statistics"]
        self.assertEqual(
            (stats[0]["support"], stats[0]["harmed"], stats[0]["helped"]), (2, 1, 1)
        )
        self.assertEqual(stats[0]["signed_delta"], 0)
        self.assertAlmostEqual(stats[1]["signed_delta"], 1 / 3)
        self.assertEqual(
            mask["provenance"]["calibration_writer_ids"], ["cal-1", "cal-2"]
        )
        self.assertEqual(mask["class_mapping_digest"], mapping_digest(self.labels))

    def test_negative_signed_delta_is_retained(self) -> None:
        self.low[0]["prediction"] = 0
        mask = select_residual_classes(self.paired(), 1)
        self.assertEqual(mask["class_statistics"][0]["signed_delta"], -1 / 4)
        self.assertEqual(mask["class_statistics"][0]["penalty"], 0)

    def test_pair_penalty_counts_only_new_confusion_and_symmetrizes(self) -> None:
        mask = select_residual_classes(self.paired(), 3, method="pair", eta=0.5)
        stats = mask["class_statistics"]
        self.assertEqual(stats[0]["pair_score"], 0)
        self.assertAlmostEqual(stats[1]["pair_score"], 1 / 6)
        self.assertAlmostEqual(stats[2]["pair_score"], 1 / 6)
        self.assertAlmostEqual(stats[1]["selection_score"], 1 / 3 + 1 / 12)

    def test_controls_and_seed_reproducibility(self) -> None:
        paired = self.paired()
        first = select_residual_classes(paired, 2, method="random", seed=29)
        self.assertEqual(
            first, select_residual_classes(paired, 2, method="random", seed=29)
        )
        self.assertEqual(len(set(first["residual_class_indices"])), 2)
        self.assertEqual(
            select_residual_classes(
                paired, 1, method="frequency", frequencies=[9, 2, 1]
            )["residual_class_indices"],
            [0],
        )
        self.assertEqual(
            select_residual_classes(
                paired, 1, method="directional_svd", directional_errors=[0.1, 0.2, 0.8]
            )["residual_class_indices"],
            [2],
        )
        self.assertAlmostEqual(
            select_residual_classes(paired, 1, method="raw_difficulty")[
                "class_statistics"
            ][0]["selection_score"],
            1 / 4,
        )
        with self.assertRaisesRegex(ValueError, "one value per class"):
            select_residual_classes(paired, 1, method="frequency")

    def test_test_validation_and_fit_overlap_are_rejected(self) -> None:
        for split in ("test", "validation", "fit"):
            with (
                self.subTest(split=split),
                self.assertRaisesRegex(ValueError, "calibration"),
            ):
                write_prediction_file(
                    self.root / "bad.jsonl",
                    {**self.metadata, "split": split},
                    self.dense,
                )
        with self.assertRaisesRegex(ValueError, "official train"):
            write_prediction_file(
                self.root / "bad.jsonl",
                {**self.metadata, "official_partition": "test"},
                self.dense,
            )
        self.dense[0]["writer_id"] = "fit-1"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.paired()

    def test_pair_mismatches_are_rejected(self) -> None:
        for field, value in (
            ("writer_id", "different-writer"),
            ("class_id", 2),
            ("sample_id", "other-id"),
        ):
            with self.subTest(field=field):
                original = self.low[0][field]
                self.low[0][field] = value
                with self.assertRaisesRegex(ValueError, "mismatch|differ"):
                    self.paired()
                self.low[0][field] = original
        self.paired()
        path = self.root / "low.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = json.loads(lines[0])
        metadata["backbone_digest"] = "different"
        path.write_text("\n".join([json.dumps(metadata), *lines[1:]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "backbone_digest"):
            load_paired_predictions(self.root / "dense.jsonl", path)

    def test_duplicate_and_missing_class_support_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            write_prediction_file(
                self.root / "bad.jsonl", self.metadata, [*self.dense, self.dense[0]]
            )
        with self.assertRaisesRegex(ValueError, "support"):
            write_prediction_file(
                self.root / "bad.jsonl", self.metadata, self.dense[:-1]
            )

    def test_directional_svd_is_scale_invariant(self) -> None:
        weights = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        errors = directional_svd_errors(weights, 1)
        scaled = directional_svd_errors(
            weights * torch.tensor([[2.0], [4.0], [8.0]]), 1
        )
        torch.testing.assert_close(torch.tensor(errors), torch.tensor(scaled))
        torch.testing.assert_close(
            torch.tensor(directional_svd_errors(weights, 2)),
            torch.zeros(3),
            atol=1e-6,
            rtol=0,
        )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            directional_svd_errors(torch.zeros(3, 2), 1)

    def test_cli_writes_provenance_and_rejects_test_frequency(self) -> None:
        self.paired()
        command = [
            sys.executable,
            "scripts/select_residual_classes.py",
            "--dense-predictions",
            str(self.root / "dense.jsonl"),
            "--low-rank-predictions",
            str(self.root / "low.jsonl"),
            "--output",
            str(self.root / "mask.json"),
            "--k",
            "2",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        mask = json.loads((self.root / "mask.json").read_text(encoding="utf-8"))
        self.assertEqual(mask["residual_class_indices"], [1, 2])
        self.assertEqual(
            len(mask["provenance"]["prediction_file_digests"]["dense"]), 64
        )
        control = self.root / "control.json"
        control.write_text(
            json.dumps(
                {
                    "class_mapping_digest": mapping_digest(self.labels),
                    "values": [10, 1, 2],
                    "provenance": {"source": "fixture"},
                    "split": "test",
                    "official_partition": "test",
                    "manifest_digest": "fit-sha",
                }
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            command + ["--method", "frequency", "--control-scores", str(control)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("paired fit manifest", result.stderr)

    def test_directional_cli_requires_matching_checkpoint_and_pilot_rank(self) -> None:
        self.paired()
        low_path = self.root / "low.jsonl"
        lines = low_path.read_text(encoding="utf-8").splitlines()
        metadata = json.loads(lines[0])
        metadata["classifier_rank"] = 1
        low_path.write_text(
            "\n".join([json.dumps(metadata), *lines[1:]]), encoding="utf-8"
        )
        control = self.root / "control.json"
        command = [
            sys.executable,
            "scripts/select_residual_classes.py",
            "--dense-predictions",
            str(self.root / "dense.jsonl"),
            "--low-rank-predictions",
            str(low_path),
            "--output",
            str(self.root / "mask.json"),
            "--k",
            "1",
            "--method",
            "directional_svd",
            "--control-scores",
            str(control),
        ]
        for checkpoint, rank, success in (
            ("wrong-dense", 1, False),
            ("dense-sha", 2, False),
            ("dense-sha", 1, True),
        ):
            control.write_text(
                json.dumps(
                    {
                        "class_mapping_digest": mapping_digest(self.labels),
                        "values": [0.1, 0.2, 0.8],
                        "rank": rank,
                        "provenance": {"checkpoint_digest": checkpoint},
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, success, result.stderr)
        mask = json.loads((self.root / "mask.json").read_text(encoding="utf-8"))
        self.assertEqual(mask["residual_class_indices"], [2])


if __name__ == "__main__":
    unittest.main()
