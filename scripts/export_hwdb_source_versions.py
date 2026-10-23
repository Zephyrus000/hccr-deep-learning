"""Match merged HWDB GNTs to version-separated sources by basename and SHA-256.

The separated directory labels must come from the original archive inventory.
This verifies file identity, not archive authenticity or writer-number ranges.
Mapping keys are POSIX paths relative to --data-root. In a container, preserve
those relative paths or prepend the same new prefix to every mapping key when
using a higher --data-root. Transfer audit hashes with the mapping and verify
the destination files against them. Plan mode reads directory/stat data only.
Execute publishes a new output directory containing both JSON files atomically;
an existing destination is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

VERSIONS = ("HWDB1.0", "HWDB1.1", "HWDB1.2")
GROUPS = {"HWDB1.0-1.1": VERSIONS[:2], "HWDB1.0-1.2": VERSIONS}


def _inventory(root: Path, name: str) -> list[dict]:
    directory = root / name
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Missing or symlinked source directory: {directory}")
    result, basenames = [], set()
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinked source is not supported: {path}")
        if not path.is_file() or path.suffix.lower() != ".gnt":
            continue
        basename = path.name.casefold()
        if basename in basenames:
            raise ValueError(f"Duplicate basename in {name}: {path.name}")
        if not re.fullmatch(r"\d+(?:-[A-Za-z]+)*\.gnt", path.name, re.I):
            raise ValueError(f"Unexpected HWDB filename: {path}")
        info = path.stat()
        if info.st_size <= 0:
            raise ValueError(f"Empty GNT source: {path}")
        basenames.add(basename)
        result.append(
            {
                "source_file": path.relative_to(root).as_posix(),
                "basename": path.name,
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        )
    if not result:
        raise ValueError(f"Source directory contains no GNT files: {directory}")
    return result


def _hash(root: Path, item: dict) -> str:
    path = root / item["source_file"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    info = path.stat()
    if (info.st_size, info.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
        raise ValueError(f"Source changed while hashing: {path}")
    return digest.hexdigest()


def export_hwdb_source_versions(
    data_root: Path, output_dir: Path, *, execute: bool = False
) -> dict:
    """Inventory without hashing, or verify and atomically publish a new bundle."""
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    names = [
        f"{version}_{partition}"
        for version in VERSIONS
        for partition in ("Train", "Test")
    ]
    names.extend(
        f"{group}_{partition}" for group in GROUPS for partition in ("Train", "Test")
    )
    inventory = {name: _inventory(root, name) for name in names}
    report = {
        "schema_version": 1,
        "status": "planned",
        "hashing_performed": False,
        "data_root": str(root),
        "output_dir": str(output),
        "inventory": inventory,
        "assumptions": __doc__.strip(),
    }
    if not execute:
        return report
    if output.exists():
        raise FileExistsError(f"Output bundle already exists: {output}")
    if output == root or output.is_relative_to(root):
        raise ValueError("Output bundle must be outside the source data root")
    for entries in inventory.values():
        for item in entries:
            item["sha256"] = _hash(root, item)
    separated_hashes = {}
    for version in VERSIONS:
        for partition in ("Train", "Test"):
            for item in inventory[f"{version}_{partition}"]:
                previous = separated_hashes.get(item["sha256"])
                if previous and previous != (item["basename"], partition):
                    raise ValueError(
                        "Duplicate source content under different names or partitions: "
                        f"{item['source_file']}"
                    )
                separated_hashes[item["sha256"]] = (item["basename"], partition)
    mapping, matches = {}, []
    for version in VERSIONS:
        for partition in ("Train", "Test"):
            for item in inventory[f"{version}_{partition}"]:
                mapping[item["source_file"]] = version
    for group, versions in GROUPS.items():
        for partition in ("Train", "Test"):
            authoritative = [
                (version, item)
                for version in versions
                for item in inventory[f"{version}_{partition}"]
            ]
            expected, covered = (
                {item["source_file"] for _, item in authoritative},
                set(),
            )
            for item in inventory[f"{group}_{partition}"]:
                candidates = [
                    (version, source)
                    for version, source in authoritative
                    if source["basename"] == item["basename"]
                    and source["sha256"] == item["sha256"]
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Expected one basename+SHA256 match for "
                        f"{item['source_file']}; found {len(candidates)}"
                    )
                version, source = candidates[0]
                covered.add(source["source_file"])
                mapping[item["source_file"]] = version
                matches.append(
                    {
                        "merged_source": item["source_file"],
                        "separated_source": source["source_file"],
                        "source_version": version,
                        "official_partition": partition.lower(),
                        "sha256": item["sha256"],
                    }
                )
            if covered != expected:
                raise ValueError(
                    f"Merged {group}_{partition} is missing separated sources: "
                    f"{sorted(expected - covered)}"
                )
    # Detect additions/removals and metadata changes before publication.
    for name in names:
        before = [
            {key: value for key, value in item.items() if key != "sha256"}
            for item in inventory[name]
        ]
        if _inventory(root, name) != before:
            raise ValueError(f"Source inventory changed during export: {name}")
    payload = dict(sorted(mapping.items()))
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    report.update(
        status="completed",
        hashing_performed=True,
        matches=matches,
        mapping_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".hwdb-source-versions-", dir=output.parent))
    stage_path = stage / "bundle"
    stage_path.mkdir()
    (stage_path / "source_versions.json").write_text(
        encoded, encoding="utf-8", newline="\n"
    )
    (stage_path / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Retain both verified files in stage_path if atomic publication fails.
    os.rename(stage_path, output)
    stage.rmdir()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = export_hwdb_source_versions(
        args.data_root, args.output_dir, execute=args.execute
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "hashing_performed": report["hashing_performed"],
                "output_dir": report["output_dir"],
                "file_counts": {
                    name: len(items) for name, items in report["inventory"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
