from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from hccr.data import HCCRDataset, audit_manifest, build_lmdb_image_store, read_manifest
from hccr.data.casia_manifest import build_casia_manifest, gnt_writer_id
from hccr.data.gnt import iter_gnt_records, read_gnt_image
from hccr.data.manifest import validate_training_manifest, writer_provenance


def gnt_bytes(label: str, pixels: bytes = b"\x00\x40\x80\xff") -> bytes:
    return struct.pack("<I2sHH", 14, label.encode("gb18030"), 2, 2) + pixels


class CASIAGNTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        self.raw = self.root / "raw"
        self.train = self.raw / "HWDB1.1_Train"
        self.test = self.raw / "HWDB1.1_Test"
        self.output = self.root / "processed" / "fixture"
        self.train.mkdir(parents=True)
        self.test.mkdir()
        samples = gnt_bytes("中") + gnt_bytes("一") + gnt_bytes("龠") + gnt_bytes("!")
        for writer in ("1001-f", "1002-f"):
            (self.train / f"{writer}.gnt").write_bytes(samples)
        (self.test / "1241-f.gnt").write_bytes(samples)

    def build(self, **options) -> dict:
        return build_casia_manifest(
            self.train, self.test, self.raw, self.output, **options
        )

    def test_preserves_record_offsets_labels_and_grayscale(self) -> None:
        path = self.train / "1001-f.gnt"
        records = list(iter_gnt_records(path))
        self.assertEqual([record.offset for record in records], [0, 14, 28, 42])
        self.assertEqual(records[0].tag, "中".encode("gb18030"))
        image = read_gnt_image(path, 14)
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(image.tobytes(), b"\x00\x40\x80\xff")

    def test_manifest_has_shared_mapping_writer_split_and_unique_records(self) -> None:
        report = self.build(seed=7)
        rows = read_manifest(self.output / "manifest.csv")
        audit = audit_manifest(rows)
        self.assertEqual(audit.sample_count, 9)
        self.assertEqual(audit.class_count, 3)
        self.assertEqual(audit.writer_overlap_count, 0)
        self.assertEqual(audit.split_counts, {"train": 3, "validation": 3, "test": 3})
        self.assertIs(writer_provenance(rows)["writer_disjoint_verified"], True)
        self.assertEqual(report["excluded_by_label_set"], 3)
        self.assertEqual(
            report["writer_counts"], {"train": 1, "validation": 1, "test": 1}
        )
        mapping = json.loads(
            (self.output / "class_mapping_full.json").read_text("utf-8")
        )
        self.assertEqual(
            mapping["labels"],
            [
                {"class_id": 0, "unicode_label": "一"},
                {"class_id": 1, "unicode_label": "中"},
                {"class_id": 2, "unicode_label": "龠"},
            ],
        )
        for split in ("train", "validation", "test"):
            self.assertEqual(
                {
                    row["unicode_label"]: int(row["class_id"])
                    for row in rows
                    if row["split"] == split
                },
                {"一": 0, "中": 1, "龠": 2},
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_manifest(rows + [dict(rows[0], sample_id="duplicate-record")])

    def test_level1_filter_excludes_level2_and_symbols(self) -> None:
        report = self.build(label_set="gb2312-level1", expected_class_count=2)
        self.assertEqual(report["class_count"], 2)
        self.assertEqual(report["excluded_by_label_set"], 6)
        self.assertEqual(
            {
                row["unicode_label"]
                for row in read_manifest(self.output / "manifest.csv")
            },
            {"一", "中"},
        )

    def test_lmdb_and_filesystem_return_identical_pixels_targets_and_metadata(
        self,
    ) -> None:
        self.build()
        manifest = self.output / "manifest.csv"
        rows = read_manifest(manifest)
        report = build_lmdb_image_store(
            manifest, rows, self.raw, self.output / "images.lmdb"
        )
        self.assertEqual(report["sample_count"], 9)
        self.assertEqual(
            report["value_encoding"], "encoded_images_with_lossless_png_for_gnt_records"
        )
        filesystem = HCCRDataset(manifest, "train", storage_backend="filesystem")
        stored = HCCRDataset(manifest, "train", storage_backend="lmdb")
        self.addCleanup(stored.close)
        for index in range(len(filesystem)):
            actual, target, metadata = stored[index]
            expected, expected_target, expected_metadata = filesystem[index]
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual((target, metadata), (expected_target, expected_metadata))
            self.assertEqual(stored.load_image(index).tobytes(), b"\x00\x40\x80\xff")

    def test_rebuild_is_deterministic_and_does_not_overwrite(self) -> None:
        first = self.build(seed=17)
        with self.assertRaises(FileExistsError):
            self.build(seed=17)
        other = self.output.with_name("second")
        second = build_casia_manifest(self.train, self.test, self.raw, other, seed=17)
        self.assertEqual(first["manifest_digest"], second["manifest_digest"])

    def test_rejects_writer_overlap_even_with_different_zero_padding(self) -> None:
        (self.test / "01001-t.gnt").write_bytes(gnt_bytes("中"))
        with self.assertRaisesRegex(ValueError, "writer overlap"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_test_label_without_training_samples(self) -> None:
        (self.test / "1241-f.gnt").write_bytes(gnt_bytes("国"))
        with self.assertRaisesRegex(ValueError, "test labels absent"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_expected_class_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 3755 classes, found 3"):
            self.build(expected_class_count=3755)
        self.assertFalse(self.output.exists())

    def test_rejects_truncated_and_invalid_gnt_records(self) -> None:
        path = self.raw / "broken.gnt"
        for payload in (
            b"\x00",
            gnt_bytes("中")[:-1],
            struct.pack("<I2sHH", 10, b"aa", 0, 0),
        ):
            with self.subTest(payload=payload):
                path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "GNT"):
                    list(iter_gnt_records(path))

    def competition_fixture(self) -> Path:
        competition = self.raw / "ICDAR2013_Test"
        competition.mkdir()
        for writer, path in enumerate(
            [
                self.train / "1001-f.gnt",
                self.train / "1002-f.gnt",
                self.test / "1241-f.gnt",
                competition / "C001-f-f.gnt",
            ]
        ):
            path.write_bytes(
                b"".join(
                    gnt_bytes(label, bytes([writer * 10 + index, 64, 128, 255]))
                    for index, label in enumerate(["中", "一", "龠"])
                )
            )
        return competition

    def test_icdar_writer_parser_preserves_source_identifier(self) -> None:
        self.assertEqual(gnt_writer_id(Path("C001-f-f.gnt")), "icdar2013:C001")
        self.assertEqual(gnt_writer_id(Path("C01-f.gnt")), "icdar2013:C001")
        self.assertEqual(gnt_writer_id(Path("001-f.gnt")), "casia:1")

    def test_competition_build_has_pixel_audit_and_training_guard(self) -> None:
        competition = self.competition_fixture()
        report = build_casia_manifest(
            self.train,
            self.test,
            self.raw,
            self.output,
            icdar_dir=competition,
            protocol="icdar2013-fixture",
            audit_pixel_duplicates=True,
        )
        self.assertEqual(report["image_overlap_check"]["status"], "passed")
        self.assertEqual(report["image_overlap_check"]["checked_samples"], 12)
        self.assertEqual(
            report["writer_counts"],
            {"train": 1, "validation": 1, "test": 1, "icdar": 1},
        )
        self.assertEqual(
            report["split_counts"], {"train": 3, "validation": 3, "test": 3, "icdar": 3}
        )
        self.assertEqual(len(report["pairwise_writer_overlap_counts"]), 6)
        self.assertTrue(
            all(
                value == 0
                for value in report["pairwise_writer_overlap_counts"].values()
            )
        )
        self.assertTrue(
            all(len(source["sha256"]) == 64 for source in report["sources"])
        )
        manifest = self.output / "manifest.csv"
        rows = read_manifest(manifest)
        provenance = validate_training_manifest(manifest, rows)
        self.assertEqual(provenance["dataset_protocol"], "icdar2013-fixture")
        self.assertTrue(
            all(
                row["writer_id"] == "icdar2013:C001"
                for row in rows
                if row["split"] == "icdar"
            )
        )
        self.assertEqual(
            {row["writer_id"] for row in rows if row["split"] == "test"}, {"casia:1241"}
        )
        for split in ("train", "validation", "test", "icdar"):
            self.assertEqual(
                {
                    (row["unicode_label"], row["class_id"])
                    for row in rows
                    if row["split"] == split
                },
                {("一", "0"), ("中", "1"), ("龠", "2")},
            )
        build_lmdb_image_store(manifest, rows, self.raw, self.output / "images.lmdb")
        for split in ("train", "validation", "test", "icdar"):
            stored = HCCRDataset(manifest, split, storage_backend="lmdb")
            try:
                self.assertEqual(len(stored), 3)
                self.assertEqual({item["split"] for item in stored.rows}, {split})
                filesystem = HCCRDataset(manifest, split, storage_backend="filesystem")
                self.assertTrue(torch.equal(stored[0][0], filesystem[0][0]))
                self.assertEqual(stored[0][1], filesystem[0][1])
            finally:
                stored.close()
        self.assertTrue(all(len(row["pixel_sha256"]) == 64 for row in rows))
        manifest.write_text(manifest.read_text("utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            validate_training_manifest(manifest, rows)

    def test_cross_split_duplicate_image_is_rejected_under_renamed_writer(self) -> None:
        competition = self.competition_fixture()
        (competition / "C001-f-f.gnt").write_bytes(
            (self.train / "1001-f.gnt").read_bytes()
        )
        with self.assertRaisesRegex(ValueError, "exact-image leakage"):
            build_casia_manifest(
                self.train,
                self.test,
                self.raw,
                self.output,
                icdar_dir=competition,
                protocol="icdar2013-fixture",
                audit_pixel_duplicates=True,
            )
        self.assertFalse(self.output.exists())

    def test_competition_requires_pixel_audit_and_expected_test_count(self) -> None:
        competition = self.competition_fixture()
        with self.assertRaisesRegex(ValueError, "require the exact-image"):
            build_casia_manifest(
                self.train,
                self.test,
                self.raw,
                self.output,
                icdar_dir=competition,
                protocol="icdar2013-fixture",
            )
        with self.assertRaisesRegex(
            ValueError, "expected 224419 icdar samples, found 3"
        ):
            build_casia_manifest(
                self.train,
                self.test,
                self.raw,
                self.output,
                icdar_dir=competition,
                expected_icdar_samples=224419,
                audit_pixel_duplicates=True,
            )

    def test_changed_label_does_not_hide_duplicate_pixels(self) -> None:
        competition = self.competition_fixture()
        original = (self.train / "1001-f.gnt").read_bytes()
        (competition / "C001-f-f.gnt").write_bytes(gnt_bytes("一", original[10:14]))
        with self.assertRaisesRegex(ValueError, "exact-image leakage"):
            build_casia_manifest(
                self.train,
                self.test,
                self.raw,
                self.output,
                icdar_dir=competition,
                protocol="icdar2013-fixture",
                audit_pixel_duplicates=True,
            )

    def test_same_split_duplicate_is_retained_and_counted(self) -> None:
        competition = self.competition_fixture()
        path = self.train / "1001-f.gnt"
        contents = path.read_bytes()
        path.write_bytes(contents + contents[:14])
        report = build_casia_manifest(
            self.train,
            self.test,
            self.raw,
            self.output,
            icdar_dir=competition,
            protocol="icdar2013-fixture",
            audit_pixel_duplicates=True,
        )
        self.assertEqual(report["image_overlap_check"]["checked_samples"], 13)
        self.assertEqual(report["image_overlap_check"]["same_split_duplicates"], 1)
        self.assertEqual(len(read_manifest(self.output / "manifest.csv")), 13)

    def test_class_mapping_is_pinned_by_dataset_audit(self) -> None:
        competition = self.competition_fixture()
        build_casia_manifest(
            self.train,
            self.test,
            self.raw,
            self.output,
            icdar_dir=competition,
            protocol="icdar2013-fixture",
            audit_pixel_duplicates=True,
        )
        mapping = self.output / "class_mapping_full.json"
        mapping.write_text(mapping.read_text("utf-8") + "\n", encoding="utf-8")
        manifest = self.output / "manifest.csv"
        with self.assertRaisesRegex(ValueError, "class mapping changed"):
            validate_training_manifest(manifest, read_manifest(manifest))

    def test_empty_record_is_counted_and_excluded_explicitly(self) -> None:
        for directory in (self.train, self.test):
            for path in directory.glob("*.gnt"):
                with path.open("ab") as stream:
                    stream.write(
                        struct.pack("<I2sHH", 10, "中".encode("gb18030"), 0, 0)
                    )
        report = self.build()
        self.assertEqual(report["empty_records"], 3)
        self.assertEqual(report["records_seen"], 15)
        self.assertEqual(len(read_manifest(self.output / "manifest.csv")), 9)

    def test_development_writer_filter_excludes_extra_dataset_writers(self) -> None:
        (self.train / "501-f.gnt").write_bytes(gnt_bytes("国"))
        report = self.build(
            development_writer_ids={"casia:1001", "casia:1002"},
            expected_development_writers=2,
        )
        self.assertEqual(report["class_count"], 3)
        self.assertEqual(
            report["excluded_development_sources"], ["HWDB1.1_Train/501-f.gnt"]
        )

    def test_competition_cannot_replace_casia_test(self) -> None:
        competition = self.competition_fixture()
        with self.assertRaisesRegex(ValueError, "separate icdar_dir and CASIA test"):
            build_casia_manifest(
                self.train,
                competition,
                self.raw,
                self.output,
                protocol="icdar2013-fixture",
                audit_pixel_duplicates=True,
            )
        self.assertFalse(self.output.exists())

    def test_multiple_official_test_sources_stay_in_test(self) -> None:
        extra = self.raw / "HWDB1.0_Test"
        extra.mkdir()
        (extra / "301-f.gnt").write_bytes(gnt_bytes("中"))
        report = build_casia_manifest(
            self.train, [self.test, extra], self.raw, self.output
        )
        self.assertEqual(
            report["writer_counts"], {"train": 1, "validation": 1, "test": 2}
        )
        rows = read_manifest(self.output / "manifest.csv")
        self.assertEqual(
            {item["writer_id"] for item in rows if item["split"] == "test"},
            {"casia:301", "casia:1241"},
        )

    def test_cli_presets_preserve_official_partitions(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        from scripts.build_casia_manifest import main

        for protocol, subsets, development_writers, test_writers in (
            ("icdar2013-hwdb1.1", ["HWDB1.1"], 240, 60),
            ("icdar2013-hwdb1.0-1.1", ["HWDB1.0-1.1"], 576, 144),
            ("icdar2013-hwdb1.0-1.2", ["HWDB1.0-1.2"], 816, 204),
        ):
            with (
                self.subTest(protocol=protocol),
                patch(
                    "scripts.build_casia_manifest.build_casia_manifest",
                    return_value={"sources": []},
                ) as builder,
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    main(["--protocol", protocol, "--data-root", str(self.raw)]), 0
                )
                args, options = builder.call_args
                self.assertEqual(
                    args[0], [self.raw / f"{name}_Train" for name in subsets]
                )
                self.assertEqual(
                    args[1], [self.raw / f"{name}_Test" for name in subsets]
                )
                self.assertEqual(options["icdar_dir"], self.raw / "ICDAR2013_Test")
                self.assertEqual(
                    options["expected_development_writers"], development_writers
                )
                self.assertEqual(options["expected_test_writers"], test_writers)
                self.assertEqual(options["expected_icdar_writers"], 60)
                self.assertEqual(options["expected_icdar_samples"], 224419)
                self.assertEqual(options["expected_class_count"], 3755)

    def test_full_hanzi_hwdb1_0_1_2_can_keep_icdar_separate(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        from scripts.build_casia_manifest import main

        with (
            patch(
                "scripts.build_casia_manifest.build_casia_manifest",
                return_value={"sources": []},
            ) as builder,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(
                    [
                        "--protocol",
                        "hwdb1.0-1.2",
                        "--data-root",
                        str(self.raw),
                        "--icdar-dir",
                        str(self.raw / "ICDAR2013_Test"),
                    ]
                ),
                0,
            )
        args, options = builder.call_args
        self.assertEqual(args[0], [self.raw / "HWDB1.0-1.2_Train"])
        self.assertEqual(args[1], [self.raw / "HWDB1.0-1.2_Test"])
        self.assertEqual(options["label_set"], "all-hanzi")
        self.assertIsNone(options["expected_class_count"])
        self.assertEqual(options["expected_development_writers"], 816)
        self.assertEqual(options["expected_test_writers"], 204)
        self.assertEqual(options["expected_icdar_writers"], 60)
        self.assertEqual(options["expected_icdar_samples"], 224419)
