"""Structured artifacts for reproducible training analysis."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


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
    except OSError, subprocess.CalledProcessError:
        return None


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
    except OSError, subprocess.CalledProcessError:
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
