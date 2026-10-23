"""Structured artifacts for reproducible training analysis."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from hccr.artifacts.serialization import write_json


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def initialize_run(
    output_dir: Path,
    config: Any,
    metadata: dict[str, Any],
    run_id: str | None = None,
) -> str:
    """Write reproducibility metadata and return the unique run identifier."""
    run_id = run_id or new_run_id()
    bundle = _source_bundle_state()
    if bundle is not None:
        metadata = {**metadata, "source_bundle": bundle}
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "metadata.json",
        {
            **metadata,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": (git_state := _git_state())["commit"],
            "git": git_state,
            "environment": _environment_metadata(),
        },
    )
    return run_id


def write_curves(output_dir: Path, epochs: list[dict[str, float]]) -> None:
    write_json(output_dir / "curves.json", {"epochs": epochs})


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_bundle_state(root: Path | None = None) -> dict[str, Any] | None:
    """Verify deployed working-tree files when a source bundle is present."""
    root = (root or Path(__file__).resolve().parents[3]).resolve()
    path = root / "BUNDLE_MANIFEST.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        raise ValueError("invalid source bundle manifest")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tree_digest = hashlib.sha256(canonical).hexdigest()
    if payload.get("source_tree_sha256") != tree_digest:
        raise ValueError("source bundle file inventory changed")
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
        ):
            raise ValueError("unsafe source bundle path")
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError("source bundle file is missing")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("deployed source differs from its bundle manifest")
    return {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_tree_sha256": tree_digest,
        "origin_git_commit": payload.get("git_commit"),
        "verified_files": len(files),
    }


def _git_state() -> dict[str, Any]:
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        commit = _git_commit()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "working_tree_digest": None}
    digest = hashlib.sha256()
    digest.update(b"status\0" + status + b"diff\0" + diff + b"untracked\0")
    for encoded_path in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(encoded_path + b"\0")
        path = root / encoded_path.decode("utf-8")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "commit": commit,
        "dirty": bool(status),
        "working_tree_digest": f"sha256:{digest.hexdigest()}",
    }


def _environment_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "python": platform.python_version(),
            "torch": None,
            "cuda": None,
            "packages": _package_versions(),
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "packages": _package_versions(),
    }


def _package_versions() -> dict[str, str | None]:
    names = (
        "hccr",
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "gdown",
        "Pillow",
        "PyYAML",
        "lmdb",
        "matplotlib",
        "seaborn",
        "tqdm",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
