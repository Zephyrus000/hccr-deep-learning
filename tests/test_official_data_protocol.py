from __future__ import annotations

import copy
import csv
import json
import shutil
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from hccr.data.casia_manifest import build_casia_manifest, source_version
from hccr.data.lmdb_store import manifest_digest
from hccr.data.manifest import (
    audit_manifest,
    read_manifest,
    validate_training_manifest,
    writer_provenance,
)


def records(labels: list[str], identity: int) -> bytes:
    return b"".join(
        struct.pack("<I2sHHI", 14, label.encode("gb18030"), 2, 2, identity + index)
        for index, label in enumerate(labels)
    )


class OfficialPrimaryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        labels = [
            bytes((lead, trail)).decode("gb2312")
            for lead in range(0xB0, 0xD8)
            for trail in range(0xA1, 0xFF if lead < 0xD7 else 0xFA)
        ]
        cls.train, cls.test = [], []
        identity = 0
        for version_index, version in enumerate(("HWDB1.0", "HWDB1.1")):
            for partition, writers, target in (
                ("Train", range(10), cls.train),
                ("Test", range(10, 11), cls.test),
            ):
                directory = cls.root / f"{version}_{partition}"
                directory.mkdir()
                target.append(directory)
                for writer in writers:
                    writer_id = 1000 + 100 * version_index + writer
                    (directory / f"{writer_id}-f.gnt").write_bytes(
                        records(labels, identity)
                    )
                    identity += len(labels)
        cls.output = cls.root / "output"
        cls.icdar = cls.root / "ICDAR2013_Test"
        cls.icdar.mkdir()
        (cls.icdar / "C001-f-f.gnt").write_bytes(records(labels, identity))
        cls.report = build_casia_manifest(
            cls.train,
            cls.test,
            cls.root,
            cls.output,
            protocol="hwdb1.0-1.1-official",
            label_set="gb2312-level1",
            seed=20261022,
            calibration_fraction=0.1,
            audit_pixel_duplicates=True,
            icdar_dir=cls.icdar,
        )
        cls.manifest = cls.output / "manifest.csv"
        cls.rows = read_manifest(cls.manifest)

    def test_full_primary_vocabulary_has_frozen_disjoint_roles_and_provenance(
        self,
    ) -> None:
        info = validate_training_manifest(self.manifest, self.rows)
        self.assertTrue(info["official_source_provenance_required"])
        self.assertIsNone(info["writer_disjoint_verified"])
        self.assertTrue(info["hwdb_writer_disjoint_verified"])
        self.assertEqual(
            info["icdar_identity_reconciliation"], "unverified_source_namespace_ids"
        )
        self.assertEqual(self.report["class_count"], 3755)
        self.assertEqual(
            self.report["writer_counts"],
            {
                "train": 16,
                "calibration": 2,
                "validation": 2,
                "test": 2,
                "icdar": 1,
            },
        )
        self.assertEqual(audit_manifest(self.rows).writer_overlap_count, 0)
        self.assertEqual(
            len(writer_provenance(self.rows)["pairwise_writer_overlap_counts"]), 10
        )
        for split in ("train", "calibration"):
            self.assertTrue(
                all(self.report["class_sample_counts_by_split"][split].values())
            )
        for row in self.rows:
            expected = (
                "train"
                if row["split"] in {"train", "calibration", "validation"}
                else row["split"]
            )
            self.assertEqual(row["official_partition"], expected)
            self.assertIn(row["source_version"], {"HWDB1.0", "HWDB1.1", "ICDAR2013"})
        self.assertEqual(
            self.report["sample_counts_by_version_and_split"]["HWDB1.0"]["test"], 3755
        )

    def test_training_rejects_changed_rows_and_removed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / "manifest.csv"
            manifest.write_bytes(self.manifest.read_bytes())
            with self.assertRaisesRegex(ValueError, "matching audit_report"):
                validate_training_manifest(manifest, self.rows)
            (output / "class_mapping_full.json").write_bytes(
                (self.output / "class_mapping_full.json").read_bytes()
            )
            report = dict(self.report)
            changed = [dict(row) for row in self.rows]
            changed[0]["source_version"] = "unknown"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(changed[0]))
                writer.writeheader()
                writer.writerows(changed)
            report["manifest_digest"] = manifest_digest(manifest)
            (output / "audit_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unknown source version"):
                validate_training_manifest(manifest, changed)

    def test_full_builder_output_passes_official_training_preflight(self) -> None:
        from hccr.training.proposal import validate_official_source

        validate_official_source(self.manifest, self.rows)

    def test_holdouts_may_lack_classes_while_fit_and_calibration_stay_complete(
        self,
    ) -> None:
        from hccr.training.proposal import validate_official_source

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in self.train + self.test + [self.icdar]:
                shutil.copytree(source, root / source.name)
            for source in self.report["sources"]:
                if source["assigned_split"] in {"validation", "test", "icdar"}:
                    path = root / source["source_file"]
                    path.write_bytes(path.read_bytes()[:14])
            output = root / "output"
            report = build_casia_manifest(
                [root / source.name for source in self.train],
                [root / source.name for source in self.test],
                root,
                output,
                protocol="hwdb1.0-1.1-official",
                label_set="gb2312-level1",
                seed=20261022,
                calibration_fraction=0.1,
                audit_pixel_duplicates=True,
                icdar_dir=root / self.icdar.name,
            )
            for split in ("validation", "test", "icdar"):
                self.assertEqual(report["class_counts_by_split"][split], 1)
            for split in ("train", "calibration"):
                self.assertEqual(report["class_counts_by_split"][split], 3755)
            manifest = output / "manifest.csv"
            validate_official_source(manifest, read_manifest(manifest))

    def test_training_rejects_incomplete_or_inconsistent_audit_inventories(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = output / "manifest.csv"
            manifest.write_bytes(self.manifest.read_bytes())
            (output / "class_mapping_full.json").write_bytes(
                (self.output / "class_mapping_full.json").read_bytes()
            )
            for corruption, message in (
                ("source_hash", "source SHA-256"),
                ("source_split", "partition disagrees"),
                ("version_split", "per-version sample counts"),
                ("class_support", "class support counts"),
                ("class_count", "class count"),
                ("class_mapping_hash", "class mapping changed"),
            ):
                with self.subTest(corruption=corruption):
                    report = copy.deepcopy(self.report)
                    if corruption == "source_hash":
                        report["sources"][0].pop("sha256")
                    elif corruption == "source_split":
                        report["sources"][0]["assigned_split"] = "test"
                    elif corruption == "version_split":
                        report["sample_counts_by_version_and_split"]["HWDB1.0"].pop(
                            "validation"
                        )
                    elif corruption == "class_support":
                        report["class_sample_counts_by_split"]["train"]["0"] += 1
                    elif corruption == "class_count":
                        report["class_count"] -= 1
                    else:
                        report["label_mapping_digest"] = "0" * 64
                    (output / "audit_report.json").write_text(
                        json.dumps(report), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        validate_training_manifest(manifest, self.rows)


class OfficialDataBoundaryTests(unittest.TestCase):
    def test_merged_directories_need_explicit_version_mapping(self) -> None:
        root = Path("root")
        path = root / "HWDB1.0-1.1_Train" / "1001-f.gnt"
        self.assertEqual(source_version(path, root, {}), "unknown")
        self.assertEqual(
            source_version(
                path,
                root,
                {
                    "HWDB1.0-1.1_Train/1001-f.gnt": "HWDB1.1",
                },
            ),
            "HWDB1.1",
        )

    def test_writer_repeated_in_versions_remains_one_global_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = []
            for index, version in enumerate(("HWDB1.0", "HWDB1.1")):
                folder = root / f"{version}_Train"
                folder.mkdir()
                train.append(folder)
                for writer in range(6):
                    (folder / f"{writer + 1000}-f.gnt").write_bytes(
                        records(["一", "中"], 100 * index + writer * 2)
                    )
            test = root / "HWDB1.1_Test"
            test.mkdir()
            (test / "2000-f.gnt").write_bytes(records(["一", "中"], 1000))
            output = root / "output"
            build_casia_manifest(
                train,
                test,
                root,
                output,
                calibration_fraction=0.1,
                audit_pixel_duplicates=True,
            )
            rows = read_manifest(output / "manifest.csv")
            audit = audit_manifest(rows)
            self.assertEqual(audit.writer_overlap_count, 0)
            self.assertEqual(
                audit.split_counts,
                {"train": 16, "calibration": 4, "validation": 4, "test": 2},
            )

    def test_canonical_identity_mapping_detects_competition_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test, icdar = (root / name for name in ("train", "test", "icdar"))
            for folder in (train, test, icdar):
                folder.mkdir()
            (train / "1000-f.gnt").write_bytes(records(["一"], 0))
            (train / "1001-f.gnt").write_bytes(records(["一"], 1))
            (test / "2000-f.gnt").write_bytes(records(["一"], 2))
            (icdar / "C001-f-f.gnt").write_bytes(records(["一"], 3))
            with self.assertRaisesRegex(ValueError, "writer overlap"):
                build_casia_manifest(
                    train,
                    test,
                    root,
                    root / "output",
                    icdar_dir=icdar,
                    audit_pixel_duplicates=True,
                    writer_identity_map={"icdar2013:C001": "casia:1000"},
                )

    def test_secondary_requires_an_explicit_audited_vocabulary(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires audited vocabulary"):
            build_casia_manifest(
                Path("unused"),
                Path("unused"),
                Path("."),
                Path("unused"),
                protocol="hwdb1.0-1.2-official",
                calibration_fraction=0.1,
                audit_pixel_duplicates=True,
            )

    def test_official_cli_sets_primary_protocol_without_reusing_legacy_defaults(
        self,
    ) -> None:
        from scripts.build_casia_manifest import main

        with (
            patch(
                "scripts.build_casia_manifest.build_casia_manifest", return_value={}
            ) as builder,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(["--protocol", "hwdb1.0-1.1-official", "--without-icdar"]), 0
            )
        args, options = builder.call_args
        self.assertEqual(
            args[0], [Path("data/raw/HWDB1.0_Train"), Path("data/raw/HWDB1.1_Train")]
        )
        self.assertEqual(options["seed"], 20261022)
        self.assertEqual(options["calibration_fraction"], 0.1)
        self.assertEqual(options["expected_class_count"], 3755)
        self.assertEqual(options["expected_development_writers"], 576)

    def test_default_rebuild_plan_only_contains_primary(self) -> None:
        from scripts.run_hwdb_manifest_rebuild import main

        buffer = StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["--plan-only"]), 0)
        self.assertEqual(len(json.loads(buffer.getvalue())), 1)


if __name__ == "__main__":
    unittest.main()
