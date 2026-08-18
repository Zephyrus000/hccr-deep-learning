from __future__ import annotations

import unittest

from hccr.data.manifest import audit_manifest
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
