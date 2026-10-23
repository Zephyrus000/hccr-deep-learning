"""Create a protocol-specific manifest from official CASIA GNT downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.data.casia_manifest import OFFICIAL_PROTOCOLS, build_casia_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=(
            "hwdb1.1",
            "hwdb1.0-1.2",
            "icdar2013-hwdb1.1",
            "icdar2013-hwdb1.0-1.1",
            "icdar2013-hwdb1.0-1.2",
            "custom",
            *OFFICIAL_PROTOCOLS,
        ),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-dir", type=Path, action="append")
    parser.add_argument("--test-dir", type=Path, action="append")
    parser.add_argument("--icdar-dir", type=Path)
    parser.add_argument("--without-icdar", action="store_true")
    parser.add_argument(
        "--source-version-map",
        type=Path,
        help="JSON source file/directory to HWDB version, relative to data root",
    )
    parser.add_argument(
        "--writer-identity-map",
        type=Path,
        help="JSON filename writer IDs to canonical identities from source metadata",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        help="Audited JSON list of Unicode labels, or class_mapping_full.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label-set", choices=("gb2312-level1", "all-hanzi"))
    parser.add_argument("--encoding", default="gb18030")
    parser.add_argument("--expected-class-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float)
    args = parser.parse_args(argv)
    if args.protocol == "custom" and (args.train_dir is None or args.test_dir is None):
        parser.error("custom protocol requires --train-dir and --test-dir")
    name = "HWDB1.1" if args.protocol == "hwdb1.1" else "HWDB1.0-1.2"
    label_set = args.label_set or (
        "gb2312-level1" if args.protocol == "hwdb1.1" else "all-hanzi"
    )
    expected = args.expected_class_count
    if expected is None and args.protocol == "hwdb1.1" and label_set == "gb2312-level1":
        expected = 3755
    options = {}
    train_dirs = args.train_dir or [args.data_root / f"{name}_Train"]
    test_dirs = args.test_dir or [args.data_root / f"{name}_Test"]
    icdar_dir = args.icdar_dir
    official = args.protocol in OFFICIAL_PROTOCOLS
    if official:
        versions, fixed_label_set, fixed_count = OFFICIAL_PROTOCOLS[args.protocol]
        if args.label_set not in (None, fixed_label_set) or expected not in (
            None,
            fixed_count,
        ):
            parser.error(
                "official proposal protocol has a fixed vocabulary and class count"
            )
        label_set, expected = fixed_label_set, fixed_count
        train_dirs = args.train_dir or [
            args.data_root / f"{version}_Train" for version in versions
        ]
        test_dirs = args.test_dir or [
            args.data_root / f"{version}_Test" for version in versions
        ]
        primary = len(versions) == 2
        if primary and not args.without_icdar:
            icdar_dir = icdar_dir or args.data_root / "ICDAR2013_Test"
        development_writers, test_writers = (576, 144) if primary else (816, 204)
        options = {
            "expected_development_writers": development_writers,
            "expected_test_writers": test_writers,
        }
        if icdar_dir is not None:
            options.update(expected_icdar_writers=60, expected_icdar_samples=224419)
    elif args.protocol.startswith("icdar2013"):
        if args.label_set not in (None, "gb2312-level1") or expected not in (
            None,
            3755,
        ):
            parser.error(
                "ICDAR benchmark requires the fixed 3755 GB2312 level-1 classes"
            )
        label_set = "gb2312-level1"
        expected = 3755
        if args.protocol == "icdar2013-hwdb1.1":
            train_dirs = args.train_dir or [
                args.data_root / "HWDB1.1_Train",
            ]
            test_dirs = args.test_dir or [args.data_root / "HWDB1.1_Test"]
            development_writers, test_writers = 240, 60
        else:
            dataset = (
                "HWDB1.0-1.1"
                if args.protocol == "icdar2013-hwdb1.0-1.1"
                else "HWDB1.0-1.2"
            )
            train_dirs = args.train_dir or [args.data_root / f"{dataset}_Train"]
            test_dirs = args.test_dir or [args.data_root / f"{dataset}_Test"]
            development_writers, test_writers = (
                (576, 144) if dataset == "HWDB1.0-1.1" else (816, 204)
            )
        icdar_dir = args.icdar_dir or args.data_root / "ICDAR2013_Test"
        options = {
            "expected_development_writers": development_writers,
            "expected_test_writers": test_writers,
            "expected_icdar_writers": 60,
            "expected_icdar_samples": 224419,
        }
    elif args.protocol == "hwdb1.0-1.2" and icdar_dir is not None:
        options = {
            "expected_development_writers": 816,
            "expected_test_writers": 204,
            "expected_icdar_writers": 60,
            "expected_icdar_samples": 224419,
        }
    if args.without_icdar:
        if args.icdar_dir is not None or args.protocol.startswith("icdar2013"):
            parser.error("--without-icdar conflicts with an ICDAR directory/protocol")
        icdar_dir = None
    for option, destination in (
        (args.source_version_map, "source_versions"),
        (args.writer_identity_map, "writer_identity_map"),
    ):
        if option is not None:
            value = json.loads(option.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in value.items()
            ):
                parser.error(f"{option} must contain a JSON object of strings")
            options[destination] = value
    if args.vocabulary is not None:
        value = json.loads(args.vocabulary.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = [item["unicode_label"] for item in value["labels"]]
        if not isinstance(value, list) or any(
            not isinstance(label, str) or len(label) != 1 for label in value
        ):
            parser.error("--vocabulary requires one Unicode character per label")
        options["vocabulary_labels"] = value
    report = build_casia_manifest(
        train_dirs,
        test_dirs,
        args.data_root,
        args.output_dir
        or Path("data/processed")
        / (
            args.protocol.replace(".", "_").replace("-official", "_official")
            + (
                ("_five_splits" if icdar_dir else "_four_splits")
                if official
                else ("_four_splits" if icdar_dir else "")
            )
        ),
        protocol=args.protocol,
        label_set=label_set,
        encoding=args.encoding,
        seed=args.seed
        if args.seed is not None
        else (20261022 if official else 20260817),
        validation_fraction=args.validation_fraction,
        calibration_fraction=(
            args.calibration_fraction
            if args.calibration_fraction is not None
            else (0.1 if official else 0.0)
        ),
        expected_class_count=expected,
        show_progress=True,
        audit_pixel_duplicates=True,
        icdar_dir=icdar_dir,
        **options,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "sources"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
