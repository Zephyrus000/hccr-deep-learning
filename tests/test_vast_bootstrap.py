from __future__ import annotations

import hashlib
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import yaml

from hccr.ops.vast_bootstrap import (
    Bootstrap,
    BootstrapError,
    build_parser,
    configure_runtime,
    download_archive,
    drive_file_id,
    extract_archive,
    main,
    processed_artifact_inventory,
    read_download_environment,
    zip_inventory,
)


class VastBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.secret = "https://drive.google.com/file/d/private_file_123456/view"

    def archive(self, name="source.zip", member="HWDB1.0-1.1_Train/1001-f.gnt"):
        path = self.root / name
        with ZipFile(path, "w") as archive:
            archive.writestr(member, b"synthetic-gnt")
        return path

    def hardware(self):
        return {
            "cpu_count": 32,
            "memory_limit_bytes": 128 * 1024**3,
            "gpus": [{"free_vram_mib": 96 * 1024}],
            "torch": {"cuda_available": True, "bf16_supported": True},
        }

    def template(self):
        path = self.root / "template.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "base_args": {
                        "epochs": 20,
                        "batch_size": 256,
                        "reference_batch_size": 256,
                        "precision": "bfloat16",
                        "num_workers": 12,
                    },
                    "variants": [{"name": "dense", "args": {}}],
                }
            )
        )
        return path

    def test_plan_never_creates_outputs_or_reports_environment_values(self):
        environment = self.root / ".env"
        environment.write_text(f'gdown_hwdb10_11_benchmark="{self.secret}"\n')
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--root", str(self.root)])
        self.assertEqual(code, 0)
        self.assertNotIn(self.secret, output.getvalue())
        self.assertNotIn("private_file", output.getvalue())
        self.assertFalse((self.root / "artifacts").exists())
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(
            json.loads(output.getvalue())["stages"],
            ["check", "download", "extract", "manifest", "lmdb", "config"],
        )

    def test_environment_parser_does_not_execute_shell_or_read_unneeded_values(self):
        environment = self.root / ".env"
        environment.write_text(
            f"gdown_hwdb10_11_benchmark=\"gdown --fuzzy '{self.secret}' -O data.zip\"\n"
            "server_password='arbitrary unclosed quote\n"
            "vastai_api_key=$(touch SHOULD_NOT_EXIST)\n"
        )
        values = read_download_environment(environment)
        self.assertEqual(set(values), {"gdown_hwdb10_11_benchmark"})
        self.assertEqual(
            drive_file_id(values["gdown_hwdb10_11_benchmark"]), "private_file_123456"
        )

    def test_complete_local_version_directories_skip_download_and_extraction(self):
        raw = self.root / "data/raw"
        for version in ("HWDB1.0", "HWDB1.1", "HWDB1.2"):
            for partition in ("Train", "Test"):
                (raw / f"{version}_{partition}").mkdir(parents=True)
        (raw / "ICDAR2013_Test").mkdir()
        for dataset, version_count in (("primary", 2), ("secondary", 3)):
            with self.subTest(dataset=dataset):
                bootstrap = Bootstrap(
                    build_parser().parse_args(
                        [
                            "--root",
                            str(self.root),
                            "--dataset",
                            dataset,
                        ]
                    )
                )
                plan = bootstrap.plan()
                self.assertEqual(
                    plan["stages"], ["check", "manifest", "lmdb", "config"]
                )
                self.assertEqual(plan["source_layout"]["kind"], "version_separated")
                self.assertEqual(
                    len(plan["source_layout"]["train_directories"]), version_count
                )
                self.assertTrue(plan["source_layout"]["directories_present"])
                self.assertTrue(plan["reuse_local_raw"])
                self.assertEqual(plan["extraction_root"], str(raw.resolve()))
        self.assertFalse((self.root / "artifacts").exists())

    def test_separated_manifest_uses_directory_versions_without_external_map(self):
        raw = self.root / "separated"
        for version in ("HWDB1.0", "HWDB1.1"):
            for partition in ("Train", "Test"):
                (raw / f"{version}_{partition}").mkdir(parents=True)
        (raw / "ICDAR2013_Test").mkdir()
        bootstrap = Bootstrap(
            build_parser().parse_args(
                [
                    "--root",
                    str(self.root),
                    "--raw-root",
                    str(raw),
                ]
            )
        )
        with (
            patch(
                "hccr.data.casia_manifest.build_casia_manifest",
                return_value={
                    "manifest_digest": "audited",
                    "class_count": 3755,
                },
            ) as build,
            patch("hccr.data.manifest.read_manifest", return_value=[]),
            patch("hccr.training.proposal.validate_official_source"),
        ):
            result = bootstrap.stage_manifest()
        self.assertEqual(result["class_count"], 3755)
        self.assertEqual(
            build.call_args.args[0], [raw / "HWDB1.0_Train", raw / "HWDB1.1_Train"]
        )
        self.assertEqual(
            build.call_args.args[1], [raw / "HWDB1.0_Test", raw / "HWDB1.1_Test"]
        )
        self.assertEqual(build.call_args.kwargs["source_versions"], {})
        self.assertEqual(build.call_args.kwargs["icdar_dir"], raw / "ICDAR2013_Test")
        self.assertTrue(build.call_args.kwargs["audit_pixel_duplicates"])
        self.assertEqual(build.call_args.kwargs["expected_development_writers"], 576)

    def test_incomplete_version_layout_does_not_silently_promote_sources(self):
        raw = self.root / "incomplete"
        (raw / "HWDB1.0_Train").mkdir(parents=True)
        bootstrap = Bootstrap(
            build_parser().parse_args(
                [
                    "--root",
                    str(self.root),
                    "--raw-root",
                    str(raw),
                ]
            )
        )
        self.assertFalse(bootstrap.plan()["source_layout"]["directories_present"])
        with self.assertRaisesRegex(BootstrapError, "source_version_map"):
            bootstrap.stage_manifest()

    def test_bundled_mapping_is_detected_and_checks_destination_source_bytes(self):
        raw = self.root / "raw"
        inventory, versions = {}, {}
        for partition in ("Train", "Test"):
            name = f"HWDB1.0-1.1_{partition}"
            directory = raw / name
            directory.mkdir(parents=True)
            path = directory / "001-f.gnt"
            path.write_bytes(partition.encode())
            relative = f"{name}/001-f.gnt"
            versions[relative] = "HWDB1.0"
            inventory[name] = [
                {
                    "source_file": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ]
        (raw / "ICDAR2013_Test").mkdir()
        metadata = self.root / "metadata"
        metadata.mkdir()
        mapping = metadata / "source_versions.json"
        mapping.write_text(json.dumps(versions))
        (metadata / "audit.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "hashing_performed": True,
                    "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
                    "inventory": inventory,
                }
            )
        )
        bootstrap = Bootstrap(
            build_parser().parse_args(
                [
                    "--root",
                    str(self.root),
                    "--raw-root",
                    str(raw),
                ]
            )
        )
        self.assertTrue(bootstrap.plan()["source_version_map_supplied"])
        with (
            patch(
                "hccr.data.casia_manifest.build_casia_manifest",
                return_value={
                    "manifest_digest": "audited",
                    "class_count": 3755,
                },
            ) as build,
            patch("hccr.data.manifest.read_manifest", return_value=[]),
            patch("hccr.training.proposal.validate_official_source"),
        ):
            bootstrap.stage_manifest()
            build.assert_called_once()
            (raw / "HWDB1.0-1.1_Train/001-f.gnt").write_bytes(b"changed")
            with self.assertRaisesRegex(BootstrapError, "audited_content_mismatch"):
                bootstrap.stage_manifest()
            build.assert_called_once()
        mapping.write_text(mapping.read_text() + " ")
        with self.assertRaisesRegex(BootstrapError, "mapping_audit_mismatch"):
            bootstrap.stage_manifest()

    def test_drive_reference_whitelist_rejects_commands_and_non_drive_hosts(self):
        for value in (
            f"gdown {self.secret} && touch injected",
            "$(touch injected)",
            "https://example.com/private",
            f"gdown --no-check-certificate {self.secret}",
        ):
            with self.subTest(value=value), self.assertRaises(BootstrapError):
                drive_file_id(value)

    def test_fake_download_is_quiet_validates_digest_and_reuses_archive(self):
        source = self.archive()
        payload = source.read_bytes()
        output = self.root / "download.zip"
        calls = []

        def downloader(**options):
            calls.append(options)
            print(self.secret)
            Path(options["output"]).write_bytes(payload)
            return options["output"]

        with redirect_stdout(io.StringIO()) as log:
            report = download_archive(self.secret, output, downloader=downloader)
        self.assertEqual(log.getvalue(), "")
        self.assertEqual(report["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["resume"])
        download_archive(
            self.secret, output, downloader=lambda **_: self.fail("redownloaded")
        )
        with self.assertRaisesRegex(BootstrapError, "archive_sha256_mismatch"):
            download_archive(self.secret, output, expected_sha256="0" * 64)

    def test_download_exception_does_not_expose_url(self):
        def downloader(**options):
            print(self.secret)
            raise ValueError(self.secret)

        with (
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(BootstrapError) as failure,
        ):
            download_archive(
                self.secret, self.root / "failed.zip", downloader=downloader
            )
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn(self.secret, str(failure.exception))

    def test_stage_failure_and_persisted_status_redact_exception_details(self):
        args = build_parser().parse_args(
            ["--execute", "--root", str(self.root), "--stage", "download"]
        )
        with patch.object(
            Bootstrap, "stage_download", side_effect=RuntimeError(self.secret)
        ):
            report = Bootstrap(args).run()
        self.assertEqual(report["status"], "failed")
        self.assertNotIn(self.secret, json.dumps(report))
        status = self.root / "artifacts/vast-bootstrap-primary/status.json"
        self.assertNotIn(self.secret, status.read_text())

    def test_extract_is_atomic_reusable_and_detects_changed_content(self):
        archive = self.archive()
        output = self.root / "extracted"
        self.assertFalse(extract_archive(archive, output)["reused"])
        self.assertTrue(extract_archive(archive, output)["reused"])
        (output / "HWDB1.0-1.1_Train/1001-f.gnt").write_bytes(b"changed")
        with self.assertRaisesRegex(BootstrapError, "extracted_source_changed"):
            extract_archive(archive, output)

    def test_zip_traversal_symlink_and_case_collisions_are_rejected(self):
        for member in (
            "../outside",
            "/outside",
            "C:/outside",
            "folder/../../outside",
            "..\\outside",
        ):
            with self.subTest(member=member):
                archive = self.archive("bad.zip", member)
                with self.assertRaisesRegex(BootstrapError, "unsafe_zip_member"):
                    extract_archive(archive, self.root / "bad-extraction")
                self.assertFalse((self.root / "bad-extraction").exists())
        path = self.root / "symlink.zip"
        with ZipFile(path, "w") as archive:
            entry = ZipInfo("link")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "outside")
        with self.assertRaisesRegex(BootstrapError, "unsafe_zip_member"):
            zip_inventory(path)
        path = self.root / "collision.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("File.gnt", b"a")
            archive.writestr("file.gnt", b"b")
        with self.assertRaisesRegex(BootstrapError, "duplicate_zip_member"):
            zip_inventory(path)

    def test_extract_checks_disk_before_creating_staging(self):
        archive = self.archive()
        with (
            patch(
                "hccr.ops.vast_bootstrap.require_space",
                side_effect=BootstrapError("insufficient_disk_space"),
            ),
            self.assertRaisesRegex(BootstrapError, "insufficient_disk_space"),
        ):
            extract_archive(archive, self.root / "extracted")
        self.assertFalse((self.root / "extracted.bootstrap-extracting").exists())

    def test_interrupted_extraction_restarts_only_owned_staging(self):
        archive = self.archive()
        output = self.root / "extracted"
        with (
            patch(
                "hccr.ops.vast_bootstrap.ZipFile.open",
                side_effect=OSError("interrupted"),
            ),
            self.assertRaises(OSError),
        ):
            extract_archive(archive, output)
        self.assertFalse(output.exists())
        self.assertTrue(
            (
                self.root / "extracted.bootstrap-extracting/.bootstrap-owner.json"
            ).is_file()
        )
        extract_archive(archive, output)
        self.assertTrue((output / "HWDB1.0-1.1_Train/1001-f.gnt").is_file())

    def test_failed_extraction_publication_retains_ownership_for_retry(self):
        archive = self.archive()
        output = self.root / "extracted"
        with (
            patch(
                "pathlib.Path.rename", side_effect=PermissionError("publish blocked")
            ),
            self.assertRaises(PermissionError),
        ):
            extract_archive(archive, output)
        owner = self.root / "extracted.bootstrap-extracting/.bootstrap-owner.json"
        self.assertTrue(owner.is_file())
        self.assertEqual(
            json.loads(owner.read_text())["archive_sha256"],
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        self.assertFalse(output.exists())

    def test_small_shared_memory_limits_workers_without_changing_epochs(self):
        hardware = self.hardware()
        hardware["shared_memory_free_bytes"] = 64 * 1024**2
        template = self.template()
        config = yaml.safe_load(template.read_text())
        config["base_args"]["epochs"] = 100
        template.write_text(yaml.safe_dump(config))
        output = self.root / "runtime.yaml"
        configure_runtime(
            template,
            output,
            self.root / "manifest.csv",
            hardware,
            preset="strict",
            concurrent_jobs=2,
            cache_backend=None,
            classes=3755,
        )
        config = yaml.safe_load(output.read_text())
        self.assertEqual(config["base_args"]["epochs"], 100)
        self.assertEqual(config["base_args"]["num_workers"], 1)

    def test_bf16_config_preserves_budget_and_sets_cache_runtime_fields(self):
        output = self.root / "runtime.yaml"
        report = configure_runtime(
            self.template(),
            output,
            self.root / "manifest.csv",
            self.hardware(),
            preset="bf16",
            concurrent_jobs=2,
            cache_backend=None,
            classes=3755,
        )
        config = yaml.safe_load(output.read_text())
        self.assertEqual(config["base_args"]["epochs"], 20)
        self.assertEqual(config["base_args"]["batch_size"], 256)
        self.assertEqual(config["base_args"]["reference_batch_size"], 256)
        self.assertEqual(config["base_args"]["precision"], "bfloat16")
        self.assertEqual(config["base_args"]["augmentation_backend"], "gpu")
        self.assertEqual(config["base_args"]["training_cache"], "auto")
        self.assertEqual(config["base_args"]["num_workers"], 8)
        self.assertFalse(report["manifest_ready"])
        self.assertFalse(report["training_started"])

    def test_strict_runtime_and_explicit_cache_rules(self):
        output = self.root / "runtime.yaml"
        configure_runtime(
            self.template(),
            output,
            self.root / "manifest.csv",
            self.hardware(),
            preset="strict",
            concurrent_jobs=2,
            cache_backend=None,
            classes=3755,
        )
        base = yaml.safe_load(output.read_text())["base_args"]
        self.assertEqual(
            (base["precision"], base["augmentation_backend"], base["training_cache"]),
            ("float32", "cpu", "none"),
        )
        with self.assertRaisesRegex(BootstrapError, "strict_runtime"):
            configure_runtime(
                self.template(),
                output,
                self.root / "manifest.csv",
                self.hardware(),
                preset="strict",
                concurrent_jobs=2,
                cache_backend="cuda",
                classes=3755,
            )

    def test_independent_config_executes_when_official_manifest_is_blocked(self):
        template = self.template()
        args = build_parser().parse_args(
            [
                "--execute",
                "--root",
                str(self.root),
                "--stage",
                "config",
                "--config-template",
                str(template),
            ]
        )
        report = Bootstrap(args, hardware=self.hardware()).run()
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["stages"]["config"]["result"]["manifest_ready"])
        self.assertFalse(report["training_started"])

    def test_manifest_stage_reports_missing_version_map_and_can_resume_config(self):
        args = build_parser().parse_args(
            ["--execute", "--root", str(self.root), "--stage", "manifest"]
        )
        report = Bootstrap(args).run()
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            report["stages"]["manifest"]["error_code"],
            "official_sources_require_evidence_backed_source_version_map",
        )
        args.stage = ["config"]
        args.config_template = self.template()
        report = Bootstrap(args, hardware=self.hardware()).run()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["stages"]["manifest"]["status"], "blocked")

    def test_existing_mapping_reuse_does_not_promote_legacy_manifest(
        self,
    ):
        directory = self.root / "data/processed/existing"
        directory.mkdir(parents=True)
        (directory / "manifest.csv").write_text(
            "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
        )
        mapping = directory / "class_mapping_full.json"
        mapping.write_text(
            json.dumps(
                {
                    "labels": [
                        {"class_id": index, "unicode_label": chr(0x4E00 + index)}
                        for index in range(7185)
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / "audit_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "protocol": "hwdb1.0-1.2",
                    "class_count": 7185,
                    "label_mapping_digest": hashlib.sha256(
                        mapping.read_bytes()
                    ).hexdigest(),
                    "split_counts": {"train": 1, "validation": 1, "test": 1},
                }
            ),
            encoding="utf-8-sig",
        )
        args = build_parser().parse_args(
            ["--root", str(self.root), "--dataset", "secondary"]
        )
        plan = Bootstrap(args).plan()
        self.assertEqual(plan["vocabulary_path"], str(mapping))
        self.assertTrue(plan["existing_artifacts"][0]["mapping_digest_verified"])
        self.assertFalse(plan["existing_artifacts"][0]["official_schema_candidate"])
        self.assertFalse(plan["existing_artifacts"][0]["has_calibration"])
        (directory / "manifest.csv").unlink()
        mapping_only = Bootstrap(
            build_parser().parse_args(
                ["--root", str(self.root), "--dataset", "secondary"]
            )
        ).plan()
        self.assertEqual(mapping_only["vocabulary_path"], str(mapping))
        self.assertFalse(mapping_only["existing_artifacts"][0]["manifest_present"])
        self.assertTrue(
            mapping_only["existing_artifacts"][0]["mapping_digest_verified"]
        )
        self.assertFalse(
            mapping_only["existing_artifacts"][0]["official_schema_candidate"]
        )
        mapping.write_text(mapping.read_text(encoding="utf-8") + " ", encoding="utf-8")
        inventory = processed_artifact_inventory(self.root)
        self.assertFalse(inventory[0]["mapping_digest_verified"])
        self.assertFalse(inventory[0]["secondary_vocabulary_candidate"])

    def test_explicit_existing_legacy_manifest_cannot_bypass_official_preflight(self):
        manifest = self.root / "legacy.csv"
        manifest.write_text(
            "sample_id,source_file,writer_id,unicode_label,class_id,split\n"
        )
        args = build_parser().parse_args(
            [
                "--execute",
                "--root",
                str(self.root),
                "--existing-manifest",
                str(manifest),
                "--stage",
                "download",
            ]
        )
        report = Bootstrap(
            args, downloader=lambda **_: self.fail("download attempted")
        ).run()
        self.assertEqual(report["status"], "failed")
        self.assertFalse((self.root / "data/archives").exists())

    def test_bf16_cpu_cache_is_allowed_with_gpu_augmentation_and_unfrozen_backbone(
        self,
    ):
        output = self.root / "runtime.yaml"
        configure_runtime(
            self.template(),
            output,
            self.root / "manifest.csv",
            self.hardware(),
            preset="bf16",
            concurrent_jobs=2,
            cache_backend="cpu",
            classes=3755,
        )
        base = yaml.safe_load(output.read_text())["base_args"]
        self.assertEqual(
            (base["augmentation_backend"], base["training_cache"]), ("gpu", "cpu")
        )


if __name__ == "__main__":
    unittest.main()
