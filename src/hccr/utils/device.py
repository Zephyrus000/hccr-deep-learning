"""Device selection that remains usable before PyTorch is installed."""

from __future__ import annotations


def resolve_device(preferred: str = "auto") -> str:
    """Resolve an explicit or automatic device without importing the CLI."""
    if preferred not in {"auto", "cpu", "cuda"}:
        raise ValueError("preferred device must be one of: auto, cpu, cuda")
    if preferred == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if preferred == "cuda":
            raise RuntimeError(
                "CUDA was requested but PyTorch is not installed"
            ) from None
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if preferred == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if cuda_available and preferred in {"auto", "cuda"} else "cpu"
