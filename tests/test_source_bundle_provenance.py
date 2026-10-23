from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hccr.artifacts.provenance import _source_bundle_state, initialize_run


class SourceBundleProvenanceTests(unittest.TestCase):
    def fixture(self, root, files=None):
        source = root / "src/example.py"
        source.parent.mkdir(exist_ok=True)
        source.write_text("VALUE = 7\n", encoding="utf-8")
        files = files or {
            "src/example.py": hashlib.sha256(source.read_bytes()).hexdigest()
        }
        payload = dict(
            schema_version=1,
            files=files,
            git_commit="a" * 40,
            source_tree_sha256=hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        (root / "BUNDLE_MANIFEST.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    def test_verified_bundle_without_git_is_retained_in_run_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = self.fixture(root)
            self.assertFalse((root / ".git").exists())
            state = _source_bundle_state(root)
            self.assertEqual(state["origin_git_commit"], "a" * 40)
            self.assertEqual(state["verified_files"], 1)
            self.assertEqual(state["source_tree_sha256"], payload["source_tree_sha256"])
            with (
                patch(
                    "hccr.artifacts.provenance._source_bundle_state", return_value=state
                ),
                patch(
                    "hccr.artifacts.provenance._git_state",
                    return_value={"commit": None},
                ),
                patch(
                    "hccr.artifacts.provenance._environment_metadata", return_value={}
                ),
            ):
                initialize_run(root / "run", {}, {}, run_id="fixture")
            metadata = json.loads((root / "run/metadata.json").read_text())
            self.assertIsNone(metadata["git_commit"])
            self.assertEqual(metadata["source_bundle"], state)

    def test_changed_and_missing_source_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            source = root / "src/example.py"
            source.write_text("VALUE = 8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                _source_bundle_state(root)
            source.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                _source_bundle_state(root)

    def test_unsafe_paths_and_manifest_self_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in (
                "../outside.py",
                "/outside.py",
                "C:\\outside.py",
                "BUNDLE_MANIFEST.json",
            ):
                with self.subTest(name=name):
                    self.fixture(root, {name: "0" * 64})
                    with self.assertRaises(ValueError):
                        _source_bundle_state(root)

    def test_absent_bundle_manifest_preserves_ordinary_repository_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(_source_bundle_state(Path(folder)))
