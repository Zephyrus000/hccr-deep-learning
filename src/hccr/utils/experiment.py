"""Structured artifacts for reproducible training analysis."""

from __future__ import annotations

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
            "git_commit": _git_commit(),
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


def _environment_metadata() -> dict[str, str | bool | None]:
    try:
        import torch
    except ImportError:
        return {"python": platform.python_version(), "torch": None, "cuda": None}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
