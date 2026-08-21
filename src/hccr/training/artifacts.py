"""Production checkpoint and metadata persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "checkpoint.pt")
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_recalibrated_checkpoint(
    model, output_dir: Path, metrics: dict, recalibration: dict
) -> None:
    """Persist a deployable BN-recalibrated variant beside the raw checkpoint."""
    metadata_path = output_dir / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metrics"] = metrics
    metadata["batch_norm_recalibration"] = recalibration
    torch.save(model.state_dict(), output_dir / "checkpoint_recalibrated.pt")
    (output_dir / "checkpoint_recalibrated_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def update_checkpoint_metrics(output_dir: Path, metrics: dict) -> None:
    """Replace checkpoint metrics after the diagnostics pass adds calibration data."""
    metadata_path = output_dir / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metrics"] = metrics
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
