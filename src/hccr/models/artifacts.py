"""Public checkpoint-artifact reconstruction for training and audit tools."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from hccr.models.efficient_hccr import build_model

_EFFICIENT_HCCR_KEYS = {
    "in_channels",
    "width",
    "backbone_output_channels",
    "embedding_dim",
    "stage_depths",
    "stem_stride",
    "reparameterize_depthwise",
    "dropout",
    "classification_head",
    "logit_scale",
    "angular_margin",
}
_BASELINE_KEYS = {
    "in_channels",
    "classification_head",
    "embedding_dim",
    "logit_scale",
    "angular_margin",
}


def load_model_from_run(
    run_dir: Path,
    device: str = "cpu",
    num_classes_override: int | None = None,
) -> torch.nn.Module:
    """Reconstruct and load a model from versioned run artifacts."""
    run_dir = Path(run_dir)
    metadata = read_checkpoint_metadata(run_dir)
    stored = metadata["model"]
    original_num_classes = int(stored["num_classes"])
    target_num_classes = num_classes_override or original_num_classes
    model = build_model_from_metadata(metadata, num_classes=target_num_classes)
    state = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    if target_num_classes == original_num_classes:
        model.load_state_dict(state)
    else:
        backbone_state = {
            name: value
            for name, value in state.items()
            if not name.startswith("classifier.")
        }
        incompatible = model.load_state_dict(backbone_state, strict=False)
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("classifier.")
        ]
        if incompatible.unexpected_keys or invalid_missing:
            raise RuntimeError(
                "class-count override changed non-classifier checkpoint keys"
            )
    return model.to(device)


def build_model_from_metadata(
    metadata: Mapping[str, Any], *, num_classes: int | None = None
) -> torch.nn.Module:
    """Build the exact architecture declared by checkpoint metadata."""
    stored = metadata.get("model")
    if not isinstance(stored, Mapping):
        raise ValueError("checkpoint metadata must contain a model object")
    model_name = str(stored["name"])
    model_keys = (
        _EFFICIENT_HCCR_KEYS if model_name == "efficient_hccr" else _BASELINE_KEYS
    )
    kwargs = {key: stored[key] for key in model_keys if key in stored}
    if "stage_depths" in kwargs:
        kwargs["stage_depths"] = tuple(kwargs["stage_depths"])
    resolved_classes = int(num_classes or stored["num_classes"])
    return build_model(model_name, num_classes=resolved_classes, **kwargs)


def read_checkpoint_metadata(run_dir: Path) -> dict[str, Any]:
    """Read and minimally validate one run's checkpoint metadata."""
    path = Path(run_dir) / "checkpoint_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid checkpoint metadata: {path}")
    return payload
