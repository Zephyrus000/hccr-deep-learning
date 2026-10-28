"""Behavior checks for the isolated frozen deployment runner."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location("frozen_heldouts", Path(__file__).resolve().parents[1] / "scripts/evaluate_frozen_deployment_heldouts.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_frozen_deployment_heldouts as resume


class FrozenDeploymentTests(unittest.TestCase):
    def test_source_offset_mapping_uses_merged_test_path(self):
        row = {"source_file": "HWDB1.1_Test/1001-f.gnt", "record_offset": "123", "sample_id": "local",
               "official_partition": "test", "source_version": "HWDB1.1"}
        mapped = runner.canonical(row)
        self.assertEqual(mapped["source_file"], "HWDB1.0-1.1_Test/1001-f.gnt")
        self.assertEqual(mapped["sample_id"], hashlib.sha256(b"HWDB1.0-1.1_Test/1001-f.gnt\x00123").hexdigest())
        self.assertEqual(row["sample_id"], "local")

    def test_icdar_identity_is_preserved_and_invalid_identity_is_rejected(self):
        source = "ICDAR2013_Test/C001-f-f.gnt"
        identity = hashlib.sha256((source + "\0" + "10").encode()).hexdigest()
        row = {"source_file": source, "record_offset": "10", "sample_id": identity, "writer_id": "icdar2013:C001"}
        mapped = runner.canonical(row)
        self.assertEqual(mapped["sample_id"], identity)
        self.assertEqual(mapped["source_writer_id"], "icdar2013:C001")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            runner.canonical({**row, "sample_id": "bad"})

    def test_prediction_resume_keeps_committed_rows_and_detects_corruption(self):
        runner.OUT.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="test-resume-", dir=runner.OUT)
        with patch.dict(runner.COUNTS, {"test": 4}):
            root = Path(directory)
            artifact = {"seed": 7, "family": "native_fp32"}
            unit = runner.unit_state(root, artifact, "test")
            logits = torch.full((2, 3755), -10.)
            logits[0, 17], logits[1, 23] = 10., 10.
            runner.save_scores(unit, 0, logits, np.array([17, 4]))
            runner.flush_unit(unit, 1.0)
            resumed = runner.unit_state(root, artifact, "test")
            self.assertEqual(resumed["completed"], 2)
            self.assertEqual(resumed["arrays"]["top5"][:2, 0].tolist(), [17, 23])
            resumed["arrays"]["top5"][0, 0] = 0
            resumed["arrays"]["top5"].flush()
            with self.assertRaisesRegex(ValueError, "block changed"):
                runner.unit_state(root, artifact, "test")

    def test_backbone_equality_check_rejects_changed_weights(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Linear(2, 2)
                self.classifier = torch.nn.Linear(2, 2)
        a, b = Model(), Model()
        b.load_state_dict(a.state_dict())
        self.assertEqual(len(runner.pair_identity(a, b)), 64)
        with torch.no_grad():
            b.stem.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "backbone differs"):
            runner.pair_identity(a, b)

    def test_completed_metrics_are_recomputed_from_predictions(self):
        runner.OUT.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="test-metrics-", dir=runner.OUT))
        with patch.dict(runner.COUNTS, {"test": 4}):
            (root / "cache").mkdir()
            runner.atomic(root / "protocol.json", {"test_fixture": True})
            targets = np.array([17, 4, 5, 6], dtype="int64")
            np.save(root / "cache/test-targets.npy", targets)
            (root / "cache/test-rows.csv").write_text("source_version\nHWDB1.0\nHWDB1.0\nHWDB1.1\nHWDB1.1\n", encoding="utf-8")
            artifact = {"seed": 7, "family": "native_fp32"}
            unit = runner.unit_state(root, artifact, "test")
            logits = torch.full((4, 3755), -100.)
            logits[0, 17] = 10.
            logits[1, 23], logits[1, 4] = 10., 9.
            logits[2, 5] = 10.
            logits[3, [7, 8, 9, 10, 11, 6]] = torch.tensor([10., 9., 8., 7., 6., 5.])
            runner.save_scores(unit, 0, logits, targets)
            runner.flush_unit(unit, 1.0)
            runner.finish_unit(root, unit, artifact, "test", {"test": {"sample_identity_sha256": "fixture", "targets_sha256": "fixture"}}, {"sha256": "fixture"})
            metrics = runner.read(unit["directory"] / "metrics.json")
            self.assertEqual(metrics["samples"], 4)
            self.assertEqual(metrics["top1"], 0.5)
            self.assertEqual(metrics["top5"], 0.75)
            self.assertEqual(metrics["macro_recall_present_classes"], 0.5)
            self.assertEqual(metrics["by_source_version"]["HWDB1.0"]["top5"], 1.0)
            self.assertEqual(metrics["by_source_version"]["HWDB1.1"]["top5"], 0.5)
            self.assertTrue(runner.unit_state(root, artifact, "test")["done"])

    def test_resume_rejects_corrupted_derived_targets(self):
        runner.OUT.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="test-target-resume-", dir=runner.OUT))
        (root / "cache").mkdir()
        runner.atomic(root / "protocol.json", {"test_fixture": True})
        np.save(root / "cache/test-targets.npy", np.array([17, 4, 5, 6], dtype="int64"))
        (root / "cache/test-rows.csv").write_text("fixture rows\n", encoding="utf-8")
        runner.atomic(root / "cache/reconciliation.json", {
            "protocol_file_sha256": runner.sha(root / "protocol.json"),
            "test": {"rows_sha256": runner.sha(root / "cache/test-rows.csv"),
                     "targets_sha256": runner.sha(root / "cache/test-targets.npy")}})
        with patch.dict(resume.scoring.COUNTS, {"test": 4}, clear=True):
            self.assertIn("test-targets.npy", resume.verify_derived(root))
            np.save(root / "cache/test-targets.npy", np.array([0, 4, 5, 6], dtype="int64"))
            with self.assertRaisesRegex(ValueError, "identity changed: test-targets"):
                resume.verify_derived(root)


if __name__ == "__main__":
    unittest.main()
