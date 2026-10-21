"""Create a protocol-specific manifest from official CASIA GNT downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.data.casia_manifest import build_casia_manifest


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
        ),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--train-dir", type=Path, action="append")
    parser.add_argument("--test-dir", type=Path, action="append")
    parser.add_argument("--icdar-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label-set", choices=("gb2312-level1", "all-hanzi"))
    parser.add_argument("--encoding", default="gb18030")
    parser.add_argument("--expected-class-count", type=int)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
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
    if args.protocol.startswith("icdar2013"):
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
    report = build_casia_manifest(
        train_dirs,
        test_dirs,
        args.data_root,
        args.output_dir
        or Path("data/processed")
        / (args.protocol.replace(".", "_") + ("_four_splits" if icdar_dir else "")),
        protocol=args.protocol,
        label_set=label_set,
        encoding=args.encoding,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
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
