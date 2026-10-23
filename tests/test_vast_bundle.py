from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from scripts.pack_vast_workspace import MAPPINGS, main, pack_workspace


class VastBundleTests(unittest.TestCase):
    def fixture(self, root):
        for name, payload in {
            "pyproject.toml": "[project]\nname='fixture'\n",
            "README.md": "Public source\n",
            "src/hccr/new_untracked.py": "VALUE = 1\n",
            "scripts/launch.sh": "#!/bin/sh\ntrue\n",
            "configs/public.yaml": "seed: 7\n",
            "docs/classifier_experiments.md": "Instructions\n",
            "docs/vastai_bootstrap.md": "Bootstrap\n",
            ".env": "PRIVATE_ENV",
            ".git/config": "PRIVATE_GIT",
            ".venv/local.py": "PRIVATE_VENV",
            "configs/credentials.yaml": "PRIVATE_CREDENTIALS",
            "configs/private/local.yaml": "PRIVATE_CONFIG",
            "data/image.png": "PRIVATE_IMAGE",
            "weights/model.pt": "PRIVATE_WEIGHTS",
            "requirements-lock.txt": "PRIVATE_LOCK",
        }.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")

    def test_current_untracked_source_manifest_exclusions_and_byte_identical_replay(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            # The fixture's .git is intentionally not a real repository.
            first = pack_workspace(root, root / "first.zip")
            os.utime(root / "src/hccr/new_untracked.py", (1_000_000_000, 1_000_000_000))
            second = pack_workspace(root, root / "second.zip")
            self.assertEqual(
                (root / "first.zip").read_bytes(), (root / "second.zip").read_bytes()
            )
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            with ZipFile(root / "first.zip") as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
                self.assertEqual(names, sorted(names))
                self.assertIn("src/hccr/new_untracked.py", names)
                self.assertEqual(len(names), 8)
                self.assertNotIn("BUNDLE_MANIFEST.json", manifest["files"])
                self.assertTrue(
                    all("PRIVATE" not in archive.read(n).decode() for n in names)
                )
                for name, digest in manifest["files"].items():
                    self.assertEqual(
                        hashlib.sha256(archive.read(name)).hexdigest(), digest
                    )
                canonical = json.dumps(
                    manifest["files"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
                self.assertEqual(
                    manifest["source_tree_sha256"],
                    hashlib.sha256(canonical).hexdigest(),
                )
                self.assertEqual(
                    archive.getinfo("scripts/launch.sh").external_attr >> 16, 0o100755
                )
                self.assertTrue(
                    all(
                        i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist()
                    )
                )
            self.assertEqual(
                first["archive_sha256"],
                hashlib.sha256((root / "first.zip").read_bytes()).hexdigest(),
            )

    def test_modified_working_source_changes_tree_and_archive_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            before = pack_workspace(root, root / "before.zip")
            (root / "src/hccr/new_untracked.py").write_text("VALUE = 2\n")
            after = pack_workspace(root, root / "after.zip")
            self.assertNotEqual(
                before["source_tree_sha256"], after["source_tree_sha256"]
            )
            self.assertNotEqual(before["archive_sha256"], after["archive_sha256"])

    def test_verified_source_versions_are_packaged_and_changed_maps_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            metadata = root / "artifacts/source-versions"
            metadata.mkdir(parents=True)
            mapping = metadata / "source_versions.json"
            mapping.write_text(json.dumps({"HWDB1.0-1.1_Train/001-f.gnt": "HWDB1.0"}))
            (metadata / "audit.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "hashing_performed": True,
                        "mapping_sha256": hashlib.sha256(
                            mapping.read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            pack_workspace(root, root / "bundle.zip", source_version_bundle=metadata)
            with ZipFile(root / "bundle.zip") as archive:
                self.assertEqual(
                    archive.read("metadata/source_versions.json"), mapping.read_bytes()
                )
                self.assertIn("metadata/audit.json", archive.namelist())
                self.assertNotIn(".env", archive.namelist())
            mapping.write_text(mapping.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "mapping audit"):
                pack_workspace(
                    root, root / "changed.zip", source_version_bundle=metadata
                )
            self.assertFalse((root / "changed.zip").exists())

    def test_source_symlinks_are_rejected_without_following_them(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            linked = root / "src/hccr/linked.py"
            try:
                linked.symlink_to(root / ".env")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "linked paths"):
                pack_workspace(root, root / "bundle.zip")
            self.assertFalse((root / "bundle.zip").exists())

    def test_link_metadata_rejects_a_source_before_reading_its_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            flagged = root / "src/hccr/new_untracked.py"
            with (
                patch.object(Path, "is_symlink", lambda path: path == flagged),
                self.assertRaisesRegex(ValueError, "linked paths"),
            ):
                pack_workspace(root, root / "bundle.zip")
            self.assertFalse((root / "bundle.zip").exists())

    def test_only_audited_gnt_mappings_are_included_and_hash_tampering_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            name = "icdar2013-hwdb1_0-1_1_four_splits"
            protocol, count = MAPPINGS[name]
            directory = root / "data/processed" / name
            directory.mkdir(parents=True)
            labels = sorted(
                {
                    bytes((a, b)).decode("gb2312")
                    for a in range(0xB0, 0xD8)
                    for b in range(0xA1, 0xFF)
                    if (a, b) <= (0xD7, 0xF9)
                }
            )
            mapping = directory / "class_mapping_full.json"
            mapping.write_text(
                json.dumps(
                    {
                        "labels": [
                            dict(class_id=i, unicode_label=label)
                            for i, label in enumerate(labels)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (directory / "labels.json").write_text(
                json.dumps({"labels": dict(enumerate(labels))}), encoding="utf-8"
            )
            audit = dict(
                protocol=protocol,
                source_kind="gnt",
                class_count=count,
                label_mapping_digest=hashlib.sha256(mapping.read_bytes()).hexdigest(),
            )
            (directory / "audit_report.json").write_text(json.dumps(audit))
            (directory / "manifest.csv").write_text("PRIVATE_ROWS")
            legacy = root / "data/processed/casia_hwdb"
            legacy.mkdir()
            (legacy / "class_mapping_full.json").write_text("PRIVATE_KAGGLE")
            pack_workspace(root, root / "maps.zip", include_mappings=True)
            with ZipFile(root / "maps.zip") as archive:
                data_names = [n for n in archive.namelist() if n.startswith("data/")]
            self.assertEqual(len(data_names), 3)
            self.assertFalse(
                any("manifest.csv" in n or "casia_hwdb" in n for n in data_names)
            )
            mapping.write_text(mapping.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "unverified GNT"):
                pack_workspace(root, root / "tampered.zip", include_mappings=True)

    def test_cli_writes_archive_hash_report_outside_the_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            self.assertEqual(main(["--root", str(root)]), 0)
            output = root / "artifacts/vastai/hccr-source.zip"
            report = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(
                report["archive_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            with ZipFile(output) as archive:
                self.assertFalse(
                    any(name.endswith(".report.json") for name in archive.namelist())
                )
