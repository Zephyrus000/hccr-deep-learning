"""Automatic mixed-precision policy shared by training and evaluation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

PRECISION_NAMES = ("auto", "float32", "float16", "bfloat16")


@dataclass
class PrecisionContext:
    """Resolved autocast and gradient-scaling state for one process."""

    requested: str
    resolved: str
    device_type: str
    autocast_dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None

    @property
    def autocast_enabled(self) -> bool:
        """Whether model forward passes execute in a reduced precision."""
        return self.autocast_dtype is not None

    @property
    def uses_grad_scaler(self) -> bool:
        """Whether FP16 gradients must be scaled before their optimizer step."""
        return self.scaler is not None

    def autocast(self) -> AbstractContextManager[object]:
        """Return the forward-pass autocast context, or a no-op in FP32."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a loss while preserving FP16 scaling semantics."""
        if self.scaler is None:
            loss.backward()
            return
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Expose real gradients to diagnostics before an FP16 optimizer step."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step the optimizer and report whether an FP16 overflow skipped it."""
        if self.scaler is None:
            optimizer.step()
            return True
        previous_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        return self.scaler.get_scale() >= previous_scale

    def metadata(self) -> dict[str, object]:
        """Return requested and effective settings for comparable run artifacts."""
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": (
                str(self.autocast_dtype).removeprefix("torch.")
                if self.autocast_dtype is not None
                else None
            ),
            "grad_scaler": self.uses_grad_scaler,
        }


def resolve_precision(requested: str, device: str) -> PrecisionContext:
    """Resolve a reproducible precision policy for a selected device."""
    if requested not in PRECISION_NAMES:
        raise ValueError(f"precision must be one of: {', '.join(PRECISION_NAMES)}")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    resolved = _resolve_precision_name(requested, device_type)
    if resolved == "float32":
        return PrecisionContext(requested, resolved, device_type, None, None)
    if device_type != "cuda":
        raise ValueError(f"precision={requested} requires CUDA")
    if resolved == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "bfloat16 was requested but is not supported by this CUDA GPU"
        )
    dtype = torch.float16 if resolved == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if resolved == "float16" else None
    return PrecisionContext(requested, resolved, device_type, dtype, scaler)


def _resolve_precision_name(requested: str, device_type: str) -> str:
    if requested != "auto":
        return requested
    if device_type != "cuda":
        return "float32"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
