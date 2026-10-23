from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.build_casia_manifest import main as build_manifest_cli
from scripts.export_hwdb_source_versions import (
    GROUPS,
    VERSIONS,
    export_hwdb_source_versions,
)

from hccr.artifacts.checkpoints import file_digest
from hccr.data.casia_manifest import source_version


class HWDBSourceVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "raw"
        self.output = self.root / "bundle"
        for index, version in enumerate(VERSIONS):
            for partition in ("Train", "Test"):
                directory = self.data / f"{version}_{partition}"
                directory.mkdir(parents=True)
                (directory / f"{100 + index}-f.gnt").write_bytes(
                    f"{version}-{partition}".encode()
                )
        for group, versions in GROUPS.items():
            for partition in ("Train", "Test"):
                target = self.data / f"{group}_{partition}"
                target.mkdir()
                for version in versions:
                    for path in (self.data / f"{version}_{partition}").iterdir():
                        shutil.copyfile(path, target / path.name)

    def test_plan_never_hashes_or_creates_outputs(self):
        with patch("scripts.export_hwdb_source_versions._hash") as hashing:
            report = export_hwdb_source_versions(self.data, self.output)
        hashing.assert_not_called()
        self.assertFalse(report["hashing_performed"])
        self.assertEqual(len(report["inventory"]), 10)
        self.assertFalse(self.output.exists())

    def test_content_matched_mapping_is_builder_compatible_and_audited(self):
        report = export_hwdb_source_versions(self.data, self.output, execute=True)
        mapping_path = self.output / "source_versions.json"
        mapping = json.loads(mapping_path.read_text())
        self.assertEqual(len(mapping), 16)
        self.assertEqual(report["mapping_sha256"], file_digest(mapping_path))
        self.assertEqual(len(report["matches"]), 10)
        for key, version in mapping.items():
            self.assertEqual(
                source_version(self.data / key, self.data, mapping), version
            )
        with (
            patch(
                "scripts.build_casia_manifest.build_casia_manifest", return_value={}
            ) as build,
            redirect_stdout(StringIO()),
        ):
            build_manifest_cli(
                [
                    "--protocol",
                    "custom",
                    "--data-root",
                    str(self.data),
                    "--train-dir",
                    str(self.data / "HWDB1.0-1.1_Train"),
                    "--test-dir",
                    str(self.data / "HWDB1.0-1.1_Test"),
                    "--without-icdar",
                    "--source-version-map",
                    str(mapping_path),
                    "--output-dir",
                    str(self.root / "manifest"),
                ]
            )
        self.assertEqual(build.call_args.kwargs["source_versions"], mapping)
        self.assertEqual(
            json.loads((self.output / "audit.json").read_text())["status"], "completed"
        )
        with self.assertRaises(FileExistsError):
            export_hwdb_source_versions(self.data, self.output, execute=True)

    def test_mismatched_bytes_reject_without_publishing(self):
        path = self.data / "HWDB1.0-1.1_Train" / "100-f.gnt"
        path.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "basename\\+SHA256"):
            export_hwdb_source_versions(self.data, self.output, execute=True)
        self.assertFalse(self.output.exists())

    def test_missing_merged_source_rejects_incomplete_coverage(self):
        (self.data / "HWDB1.0-1.2_Test" / "102-f.gnt").unlink()
        with self.assertRaisesRegex(ValueError, "missing separated sources"):
            export_hwdb_source_versions(self.data, self.output, execute=True)
        self.assertFalse(self.output.exists())

    def test_ambiguous_versions_reject_even_identical_basename_and_bytes(self):
        original = self.data / "HWDB1.0_Train" / "100-f.gnt"
        shutil.copyfile(original, self.data / "HWDB1.1_Train" / original.name)
        with self.assertRaisesRegex(ValueError, "found 2"):
            export_hwdb_source_versions(self.data, self.output, execute=True)

    def test_duplicate_nested_basename_and_missing_partition_are_rejected(self):
        directory = self.data / "HWDB1.0_Train"
        nested = directory / "duplicate"
        nested.mkdir()
        shutil.copyfile(directory / "100-f.gnt", nested / "100-f.gnt")
        with self.assertRaisesRegex(ValueError, "Duplicate basename"):
            export_hwdb_source_versions(self.data, self.output)
        shutil.rmtree(nested)
        shutil.rmtree(self.data / "HWDB1.2_Test")
        with self.assertRaisesRegex(ValueError, "Missing"):
            export_hwdb_source_versions(self.data, self.output)

    def test_wrong_partition_same_basename_is_not_a_match(self):
        shutil.copyfile(
            self.data / "HWDB1.0_Test" / "100-f.gnt",
            self.data / "HWDB1.0-1.1_Train" / "100-f.gnt",
        )
        with self.assertRaisesRegex(ValueError, "found 0"):
            export_hwdb_source_versions(self.data, self.output, execute=True)

    def test_duplicate_content_and_hash_interruption_leave_no_bundle(self):
        original = self.data / "HWDB1.0_Train" / "100-f.gnt"
        duplicate = self.data / "HWDB1.0_Train" / "999-f.gnt"
        shutil.copyfile(original, duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate source content"):
            export_hwdb_source_versions(self.data, self.output, execute=True)
        self.assertFalse(self.output.exists())
        duplicate.unlink()
        with (
            patch(
                "scripts.export_hwdb_source_versions._hash",
                side_effect=OSError("interrupted"),
            ),
            self.assertRaisesRegex(OSError, "interrupted"),
        ):
            export_hwdb_source_versions(self.data, self.output, execute=True)
        self.assertFalse(self.output.exists())

    def test_atomic_publication_failure_preserves_verified_staging_evidence(self):
        with (
            patch(
                "scripts.export_hwdb_source_versions.os.rename",
                side_effect=PermissionError("WinError 5"),
            ),
            self.assertRaises(PermissionError),
        ):
            export_hwdb_source_versions(self.data, self.output, execute=True)
        self.assertFalse(self.output.exists())
        staged = list(self.root.glob(".hwdb-source-versions-*/bundle"))
        self.assertEqual(len(staged), 1)
        mapping = staged[0] / "source_versions.json"
        audit = json.loads((staged[0] / "audit.json").read_text())
        self.assertEqual(audit["mapping_sha256"], file_digest(mapping))


if __name__ == "__main__":
    unittest.main()
