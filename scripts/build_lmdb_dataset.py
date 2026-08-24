"""Pack manifest images into a single-file LMDB dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.data import build_lmdb_image_store, read_manifest, resolve_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--map-size-gib", type=float)
    parser.add_argument("--commit-interval", type=int, default=10_000)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    rows = read_manifest(arguments.manifest)
    source_root = arguments.source_root or resolve_data_root(arguments.manifest, rows)
    output = arguments.output or arguments.manifest.with_name("images.lmdb")
    map_size = (
        int(arguments.map_size_gib * 1024**3)
        if arguments.map_size_gib is not None
        else None
    )
    report = build_lmdb_image_store(
        arguments.manifest,
        rows,
        source_root,
        output,
        map_size=map_size,
        commit_interval=arguments.commit_interval,
        show_progress=True,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
