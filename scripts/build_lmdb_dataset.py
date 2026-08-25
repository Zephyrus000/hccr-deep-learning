"""Pack manifest images into a single-file LMDB dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.data import (
    build_lmdb_image_store,
    build_lmdb_image_store_from_zip,
    read_manifest,
    resolve_data_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-root", type=Path)
    source.add_argument(
        "--source-zip",
        type=Path,
        help="Read images directly from this ZIP without extracting them.",
    )
    parser.add_argument(
        "--zip-prefix",
        default="",
        help="Optional archive path before each manifest source_file, e.g. raw.",
    )
    parser.add_argument("--map-size-gib", type=float)
    parser.add_argument("--commit-interval", type=int, default=10_000)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.zip_prefix and arguments.source_zip is None:
        parser.error("--zip-prefix requires --source-zip")
    rows = read_manifest(arguments.manifest)
    output = arguments.output or arguments.manifest.with_name("images.lmdb")
    map_size = (
        int(arguments.map_size_gib * 1024**3)
        if arguments.map_size_gib is not None
        else None
    )
    common_options = {
        "map_size": map_size,
        "commit_interval": arguments.commit_interval,
        "show_progress": True,
    }
    if arguments.source_zip is not None:
        report = build_lmdb_image_store_from_zip(
            arguments.manifest,
            rows,
            arguments.source_zip,
            output,
            zip_prefix=arguments.zip_prefix,
            **common_options,
        )
    else:
        source_root = arguments.source_root or resolve_data_root(
            arguments.manifest, rows
        )
        report = build_lmdb_image_store(
            arguments.manifest,
            rows,
            source_root,
            output,
            **common_options,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
