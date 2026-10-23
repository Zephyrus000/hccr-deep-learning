"""Create a deterministic source bundle from the current working files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

MAPPINGS = {
    "hwdb1_0-1_2_four_splits": ("hwdb1.0-1.2", 7185),
    "icdar2013-hwdb1_0-1_1_four_splits": ("icdar2013-hwdb1.0-1.1", 3755),
    "icdar2013-hwdb1_0-1_2_four_splits": ("icdar2013-hwdb1.0-1.2", 3755),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe(root: Path, path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor == root.parent:
            break
        if ancestor.is_symlink() or ancestor.is_junction():
            raise ValueError(
                f"source bundle refuses linked paths: {path.relative_to(root)}"
            )


def _verified_mappings(root: Path) -> list[Path]:
    selected = []
    for folder, (protocol, count) in MAPPINGS.items():
        directory = root / "data" / "processed" / folder
        _safe(root, directory)
        if not directory.exists():
            continue
        paths = [
            directory / name
            for name in ("class_mapping_full.json", "labels.json", "audit_report.json")
        ]
        for path in paths:
            _safe(root, path)
        mapping, labels, audit = [
            json.loads(p.read_text(encoding="utf-8-sig")) for p in paths
        ]
        entries = mapping["labels"]
        by_id = {str(entry["class_id"]): entry["unicode_label"] for entry in entries}

        def hanzi(label):
            return (
                isinstance(label, str)
                and len(label) == 1
                and (
                    0x3400 <= ord(label) <= 0x4DBF
                    or 0x4E00 <= ord(label) <= 0x9FFF
                    or 0xF900 <= ord(label) <= 0xFAFF
                    or 0x20000 <= ord(label) <= 0x323AF
                    or ord(label) == 0x3007
                )
            )

        if not (
            audit.get("source_kind") == "gnt"
            and audit.get("protocol") == protocol
            and audit.get("class_count") == count
            and len(entries) == count
            and set(by_id) == {str(i) for i in range(count)}
            and len(set(by_id.values())) == count
            and all(map(hanzi, by_id.values()))
            and labels.get("labels") == by_id
            and audit.get("label_mapping_digest") == _sha256(paths[0].read_bytes())
        ):
            raise ValueError(f"unverified GNT vocabulary: {folder}")
        if count == 3755:
            level1 = {
                bytes((lead, trail)).decode("gb2312")
                for lead in range(0xB0, 0xD8)
                for trail in range(0xA1, 0xFF)
                if (lead, trail) <= (0xD7, 0xF9)
            }
            if set(by_id.values()) != level1:
                raise ValueError(f"noncanonical 3755-class vocabulary: {folder}")
        selected.extend(paths)
    return selected


def pack_workspace(
    root: Path,
    output: Path,
    *,
    include_mappings: bool = False,
    source_version_bundle: Path | None = None,
) -> dict:
    """Package whitelisted public files and report their content identities."""
    if root.is_symlink() or root.is_junction():
        raise ValueError("source bundle root must not be linked")
    root, output = root.resolve(), output.absolute()
    if output.suffix.lower() != ".zip":
        raise ValueError("source bundle output must be a ZIP file")
    paths = [root / "pyproject.toml", root / "README.md"]
    for folder, extensions in (
        ("src", {".py"}),
        ("scripts", {".py", ".sh"}),
        ("configs", {".yaml"}),
    ):
        _safe(root, root / folder)
        for path in sorted((root / folder).rglob("*")):
            _safe(root, path)
            relative = path.relative_to(root)
            private = any(
                part.startswith(".")
                or part.lower() in {"private", "secrets", "credentials"}
                for part in relative.parts
            )
            private |= path.stem.lower() in {
                "credentials",
                "secrets",
                "private",
                "passwords",
            }
            if path.is_file() and path.suffix in extensions and not private:
                paths.append(path)
    paths.extend(
        root / "docs" / name
        for name in ("classifier_experiments.md", "vastai_bootstrap.md")
        if (root / "docs" / name).exists()
    )
    if include_mappings:
        paths.extend(_verified_mappings(root))
    payloads = {}
    for path in sorted(paths):
        _safe(root, path)
        payloads[path.relative_to(root).as_posix()] = path.read_bytes()
    if source_version_bundle is not None:
        folder = (
            source_version_bundle
            if source_version_bundle.is_absolute()
            else root / source_version_bundle
        )
        folder = folder.absolute()
        if not folder.is_relative_to(root):
            raise ValueError("source version bundle must be inside the workspace")
        metadata = {}
        for name in ("source_versions.json", "audit.json"):
            path = folder / name
            _safe(root, path)
            metadata[name] = path.read_bytes()
        mapping = json.loads(metadata["source_versions.json"])
        audit = json.loads(metadata["audit.json"])
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(key, str)
                or value not in {"HWDB1.0", "HWDB1.1", "HWDB1.2"}
                for key, value in mapping.items()
            )
            or audit.get("status") != "completed"
            or audit.get("hashing_performed") is not True
            or audit.get("mapping_sha256") != _sha256(metadata["source_versions.json"])
        ):
            raise ValueError("source version bundle failed its mapping audit")
        payloads.update(
            {"metadata/" + name: content for name, content in metadata.items()}
        )
    files = {name: _sha256(payload) for name, payload in payloads.items()}
    commit = None
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    tree_sha = _sha256(
        json.dumps(
            files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )
    manifest = {
        "schema_version": 1,
        "files": files,
        "git_commit": commit,
        "source_tree_sha256": tree_sha,
    }
    payloads["BUNDLE_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode()
    if output.is_symlink() or any(
        p.is_symlink() or p.is_junction() for p in output.parents
    ):
        raise ValueError("bundle output must not follow linked paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=".hccr-bundle-", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w") as archive:
            for name, payload in sorted(payloads.items()):
                member = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                member.create_system = 3
                member.external_attr = (
                    0o100755 if name.endswith(".sh") else 0o100644
                ) << 16
                archive.writestr(
                    member, payload, compress_type=ZIP_DEFLATED, compresslevel=9
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(output),
        "archive_sha256": _sha256(output.read_bytes()),
        "source_tree_sha256": tree_sha,
        "file_count": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/vastai/hccr-source.zip")
    )
    parser.add_argument("--include-mappings", action="store_true")
    parser.add_argument("--source-version-bundle", type=Path)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    if output.with_suffix(".report.json").is_symlink():
        raise ValueError("bundle report must not follow linked paths")
    report = pack_workspace(
        args.root,
        output,
        include_mappings=args.include_mappings,
        source_version_bundle=args.source_version_bundle,
    )
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
