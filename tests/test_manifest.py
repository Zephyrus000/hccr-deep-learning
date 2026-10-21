from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hccr.data.manifest import (
    audit_manifest,
    validate_training_manifest,
    writer_provenance,
)
from hccr.data.splitter import WriterDisjointSplitter


def row(
    sample_id: str, source_file: str, split: str, writer_id: str = ""
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "source_file": source_file,
        "writer_id": writer_id,
        "unicode_label": "一",
        "class_id": "0",
        "split": split,
    }


class ManifestTests(unittest.TestCase):
    def test_four_split_audit_checks_every_writer_pair(self) -> None:
        from itertools import combinations

        splits = ("train", "validation", "test", "icdar")
        rows = [
            row(str(i), f"{i}.png", split, f"writer-{i}")
            for i, split in enumerate(splits)
        ]
        self.assertEqual(
            audit_manifest(rows).split_counts, {split: 1 for split in splits}
        )
        provenance = validate_training_manifest(Path("unused.csv"), rows)
        self.assertIs(provenance["writer_disjoint_verified"], True)
        self.assertEqual(len(provenance["pairwise_writer_overlap_counts"]), 6)
        for left, right in combinations(range(4), 2):
            with self.subTest(left=splits[left], right=splits[right]):
                changed = [dict(item) for item in rows]
                changed[right]["writer_id"] = changed[left]["writer_id"]
                with self.assertRaisesRegex(ValueError, "writer leakage"):
                    validate_training_manifest(Path("unused.csv"), changed)

    def test_training_api_refuses_leakage_before_loading_images(self) -> None:
        from hccr.training.workflow import TrainingConfig, run_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
                "a,a.png,writer-1,A,0,train\n"
                "b,b.png,writer-1,A,0,validation\n"
                "c,c.png,writer-2,A,0,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "writer leakage"):
                run_training(
                    TrainingConfig(
                        manifest_path=manifest,
                        output_dir=root / "runs",
                        num_classes=2,
                        device="cpu",
                    )
                )
            self.assertFalse((root / "runs").exists())

    def test_train_validation_writer_leakage_fails_preflight(self) -> None:
        rows = [
            row("1", "a.png", "train", "same-writer"),
            row("2", "b.png", "validation", "same-writer"),
            row("3", "c.png", "test", "other-writer"),
        ]
        info = writer_provenance(rows)
        self.assertIs(info["writer_disjoint_verified"], False)
        self.assertEqual(info["pairwise_writer_overlap_counts"]["train/validation"], 1)
        with self.assertRaisesRegex(ValueError, "writer leakage"):
            validate_training_manifest(Path("unused.csv"), rows)

    def test_audit_counts_samples_and_empty_writer_metadata(self) -> None:
        audit = audit_manifest([row("1", "a.png", "train"), row("2", "b.png", "test")])
        self.assertEqual(audit.sample_count, 2)
        self.assertEqual(audit.writer_overlap_count, 0)

    def test_audit_rejects_duplicate_source_file(self) -> None:
        with self.assertRaises(ValueError):
            audit_manifest([row("1", "a.png", "train"), row("2", "a.png", "test")])

    def test_writer_split_is_reproducible(self) -> None:
        splitter = WriterDisjointSplitter(seed=7)
        self.assertEqual(
            splitter.validation_writers(["a", "b", "c", "d"]),
            splitter.validation_writers(["d", "c", "b", "a"]),
        )

    def test_missing_writer_metadata_is_not_reported_as_disjoint(self) -> None:
        provenance = writer_provenance(
            [row("1", "a.png", "train"), row("2", "b.png", "test")]
        )
        self.assertEqual(provenance["availability"], "unavailable")
        self.assertEqual(provenance["overlap_check"], "not_verifiable")
        self.assertIsNone(provenance["writer_disjoint_verified"])

    def test_complete_writer_metadata_reports_overlap_truthfully(self) -> None:
        disjoint = writer_provenance(
            [
                row("1", "a.png", "train", "writer-a"),
                row("2", "b.png", "test", "writer-b"),
            ]
        )
        overlapping = writer_provenance(
            [
                row("1", "a.png", "train", "writer-a"),
                row("2", "b.png", "test", "writer-a"),
            ]
        )
        self.assertIs(disjoint["writer_disjoint_verified"], True)
        self.assertIs(overlapping["writer_disjoint_verified"], False)
        self.assertEqual(overlapping["train_validation_test_overlap_count"], 1)
