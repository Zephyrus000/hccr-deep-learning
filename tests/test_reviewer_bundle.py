"""Behavior checks for portable reviewer copies and their integrity boundary."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_reviewer_bundle.py"
_SPEC = importlib.util.spec_from_file_location("reviewer_bundle_tests", _SCRIPT)
bundle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bundle
_SPEC.loader.exec_module(bundle)


class ReviewerBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_tempdir = tempfile.tempdir
        workspace = _SCRIPT.parents[1].resolve()
        temporary_root = workspace / "artifacts/test-tmp/reviewer-bundle"
        if not temporary_root.resolve().is_relative_to(workspace):
            raise ValueError("Test temporary path must remain in the workspace")
        temporary_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temporary_root)

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir = cls.previous_tempdir

    def test_snapshot_copies_bytes_without_modifying_or_linking_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "checkpoint.pt"; source.write_bytes(b"frozen weights")
            plan = bundle.BundlePlan(root)
            plan.add(source, "checkpoints/seed-07/checkpoint.pt", "checkpoint", expected_sha256=bundle.sha256(source))
            destination = root / "reviewer"; destination.mkdir()
            record = bundle.copy_entry(next(iter(plan.entries.values())), destination, plan)
            self.assertEqual(source.read_bytes(), b"frozen weights")
            self.assertEqual((destination / record["path"]).read_bytes(), b"frozen weights")
            source.write_bytes(b"later working copy")
            self.assertEqual((destination / record["path"]).read_bytes(), b"frozen weights")

    def test_frozen_hash_mismatch_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "checkpoint.pt"; source.write_bytes(b"changed")
            plan = bundle.BundlePlan(root); plan.add(source, "checkpoint.pt", "checkpoint", expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "Frozen source hash mismatch"):
                bundle.copy_entry(next(iter(plan.entries.values())), root / "bundle", plan)

    def test_credentials_raw_data_and_escape_paths_are_not_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = bundle.BundlePlan(root)
            for filename in (".env", "id_ed25519_hccr_vastai", "images.lmdb", "source.zip"):
                source = root / filename; source.write_bytes(b"excluded")
                with self.assertRaises(ValueError):
                    plan.add(source, filename, "not_allowed")
            source = root / "metrics.json"; source.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                plan.add(source, "../outside.json", "not_allowed")

    def test_conflicting_file_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = bundle.BundlePlan(root)
            for name in ("a.json", "b.json"):
                (root / name).write_text("{}", encoding="utf-8")
            plan.add(root / "a.json", "metrics.json", "metrics")
            with self.assertRaisesRegex(ValueError, "Target collision"):
                plan.add(root / "b.json", "metrics.json", "metrics")

    def test_local_document_links_become_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report = root / "report.json"; report.write_text("{}", encoding="utf-8")
            original = root / "notes.md"; original.write_text("", encoding="utf-8")
            plan = bundle.BundlePlan(root); plan.add(report, "analysis/report.json", "analysis")
            text = f"[Report]({report.as_posix()}) [Unavailable](missing.json) [Paper](https://example.org/paper)"
            rewritten = bundle.rebase_markdown(text, original, "paper/notes.md", plan)
            self.assertEqual(rewritten, "[Report](../analysis/report.json) Unavailable [Paper](https://example.org/paper)")
            line_link = f"[Report]({report.as_posix()}:12)"
            self.assertEqual(bundle.rebase_markdown(line_link, original, "paper/notes.md", plan),
                             "[Report](../analysis/report.json)")

    def test_integrity_checker_detects_changed_weights_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); weight = root / "checkpoint.pt"; weight.write_bytes(b"frozen")
            bundle.write_json(root / "MANIFEST.json", {"coverage": {"runs": 1, "deployment_completed": 0},
                              "files": [{"path": "checkpoint.pt", "sha256": bundle.sha256(weight), "bytes": 6}]})
            self.assertEqual(bundle.verify_bundle(root)["status"], "verified")
            weight.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "integrity failed"):
                bundle.verify_bundle(root)
            weight.write_bytes(b"frozen"); (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted"):
                bundle.verify_bundle(root)

    def test_variant_paths_keep_random_mask_seed_distinct_from_training_seed(self):
        self.assertEqual(bundle.variant_name({"variant": "static_residual__final-random-29"}),
                         "static-residual-random-mask-29")
        self.assertEqual(bundle.variant_name({"variant": "efficient_hccr__cosface"}),
                         "efficienthccrnet-cosface")

    def test_calibration_files_match_dense_and_rank64_pilots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); plan = bundle.BundlePlan(root); catalog = []
            for seed in (7, 17, 29):
                for filename, variant, role in (("dense", "dense", "dense"), ("low-rank", "low-rank-r064", "low_rank")):
                    run_id = f"pilots/{variant}/seed-{seed:02d}"
                    checkpoint = root / "weights" / run_id / "checkpoint.pt"
                    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(run_id.encode())
                    target = f"checkpoints/{run_id}/checkpoint.pt"
                    plan.add(checkpoint, target, "checkpoint", run_id, bundle.sha256(checkpoint))
                    catalog.append({"run_id": run_id, "checkpoint": target})
                    prediction = root / f"seed-{seed}/{filename}.jsonl"
                    prediction.parent.mkdir(exist_ok=True)
                    header = {"type": "metadata", "split": "calibration", "official_partition": "train",
                              "pilot_seed": seed, "model_role": role, "classifier_rank": 64 if role == "low_rank" else None,
                              "checkpoint_digest": bundle.sha256(checkpoint), "manifest_digest": "frozen-manifest"}
                    prediction.write_text(json.dumps(header) + "\n", encoding="utf-8")
            records = bundle.add_calibration_predictions(plan, root, catalog)
            self.assertEqual(len(records), 6)
            self.assertEqual(records[1]["predictions"], "inference/calibration/low-rank-r064/seed-07/predictions.jsonl")
            self.assertEqual({row["precision"] for row in records}, {"float32"})
            prediction = root / "seed-7/low-rank.jsonl"
            header = json.loads(prediction.read_text(encoding="utf-8"))
            header["checkpoint_digest"] = "rank104-or-unrelated-checkpoint"
            prediction.write_text(json.dumps(header) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint/protocol mismatch"):
                bundle.add_calibration_predictions(plan, root, catalog)


if __name__ == "__main__":
    unittest.main()
