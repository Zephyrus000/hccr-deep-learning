from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hccr.evidence import (
    build_evidence_bundle,
    normalize_resource_profile,
    validate_evidence_bundle,
)


class EvidenceBundleTests(unittest.TestCase):
    def test_normalizes_missing_legacy_flops_without_inventing_values(self) -> None:
        normalized = normalize_resource_profile(
            {
                "model_name": "efficient_hccr",
                "parameter_count": 10,
                "estimated_macs": 20,
                "mac_coverage": {"complete": True},
                "inference_benchmarks": [
                    {
                        "batch_size": 1,
                        "latency_p95_ms": 2.0,
                        "peak_cuda_memory_mib": 3.0,
                    }
                ],
            }
        )
        self.assertEqual(normalized["source_variant"], "resource-profile-v1")
        self.assertEqual(normalized["complexity"]["macs"]["total"], 20)
        self.assertEqual(
            normalized["complexity"]["flops"]["availability"], "not_measured"
        )
        self.assertIsNone(normalized["complexity"]["flops"]["total"])
        self.assertEqual(
            normalized["memory"]["peak_inference_cuda_memory_mib"], 3.0
        )

    def test_builds_and_validates_bundle_without_copying_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            run = experiments / "run-1"
            run.mkdir(parents=True)
            _write_json(run / "config.json", {"seed": 7})
            _write_json(run / "metadata.json", {"git_commit": "abc"})
            _write_json(
                run / "checkpoint_metadata.json",
                {
                    "schema_version": 8,
                    "model": {"name": "efficient_hccr"},
                    "preprocess": {"image_size": 96, "margin": 4},
                    "manifest_digest": "sha256:manifest",
                },
            )
            _write_json(
                run / "metrics.json",
                {"schema_version": 2, "validation": {}, "test": {}},
            )
            _write_json(
                run / "resource_profile.json",
                {
                    "estimated_macs": 20,
                    "mac_coverage": {"complete": True},
                    "inference_benchmarks": [],
                },
            )
            (run / "checkpoint.pt").write_bytes(b"checkpoint")
            output = root / "paper-evidence-v1"

            manifest = build_evidence_bundle(experiments, output)
            result = validate_evidence_bundle(output)

            self.assertEqual(len(manifest["runs"]), 1)
            self.assertEqual(result["runs"], 1)
            self.assertFalse(
                (output / "raw" / "runs" / "run-1" / "checkpoint.pt").exists()
            )
            evidence = _read_json(output / "runs" / "run-1" / "run_evidence.json")
            self.assertTrue(evidence["checkpoint"]["sha256"].startswith("sha256:"))
            self.assertEqual(
                evidence["results"]["resources"]["complexity"]["flops"][
                    "availability"
                ],
                "not_measured",
            )

    def test_validation_detects_tampered_raw_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            run = experiments / "run-1"
            run.mkdir(parents=True)
            for name, payload in {
                "config.json": {},
                "metadata.json": {},
                "checkpoint_metadata.json": {"model": {}, "preprocess": {}},
                "metrics.json": {},
                "resource_profile.json": {"inference_benchmarks": []},
            }.items():
                _write_json(run / name, payload)
            output = root / "paper-evidence-v1"
            build_evidence_bundle(experiments, output)
            (output / "raw" / "runs" / "run-1" / "config.json").write_text(
                '{"tampered": true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_evidence_bundle(output)

    def test_validation_rejects_unknown_normalized_schema(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            run = experiments / "run-1"
            run.mkdir(parents=True)
            for name, payload in {
                "config.json": {},
                "metadata.json": {},
                "checkpoint_metadata.json": {"model": {}, "preprocess": {}},
                "metrics.json": {},
                "resource_profile.json": {"inference_benchmarks": []},
            }.items():
                _write_json(run / name, payload)
            output = root / "paper-evidence-v1"
            build_evidence_bundle(experiments, output)
            evidence_path = output / "runs" / "run-1" / "run_evidence.json"
            evidence = _read_json(evidence_path)
            evidence["schema_version"] = 999
            _write_json(evidence_path, evidence)
            with self.assertRaisesRegex(ValueError, "unsupported schema version"):
                validate_evidence_bundle(output)

    def test_normalizes_derived_reprofile_collection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            experiments.mkdir()
            derived = root / "profiling-host-a"
            profile_dir = derived / "runs" / "run-1" / "cpu"
            profile_dir.mkdir(parents=True)
            _write_json(
                profile_dir / "resource_profile.json",
                {
                    "estimated_macs": 20,
                    "estimated_flops": 40,
                    "flop_coverage": {"complete": True},
                    "inference_benchmarks": [],
                },
            )
            output = root / "paper-evidence-v1"
            manifest = build_evidence_bundle(
                experiments, output, derived_roots=[derived]
            )
            evidence = _read_json(
                output
                / "derived"
                / "profiling-host-a"
                / "derived_evidence.json"
            )
        self.assertEqual(len(manifest["derived_collections"]), 1)
        self.assertEqual(evidence["profiles"][0]["run_id"], "run-1")
        self.assertEqual(
            evidence["profiles"][0]["resources"]["complexity"]["flops"][
                "total"
            ],
            40,
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
