"""Single-node PyTorch DDP lifecycle and collective helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """The process-local state for an optional distributed training run."""

    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"
    backend: str | None = None
    owns_process_group: bool = False

    @property
    def is_main_process(self) -> bool:
        """Whether this rank owns user-visible output and artifacts."""
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize all ranks when distributed training is enabled."""
        if self.enabled:
            dist.barrier()

    def broadcast_object(self, value: Any) -> Any:
        """Broadcast a rank-zero Python value to all ranks."""
        if not self.enabled:
            return value
        values = [value if self.is_main_process else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def all_gather_object(self, value: Any) -> list[Any]:
        """Gather one small Python object from each rank."""
        if not self.enabled:
            return [value]
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values


def initialize_distributed(
    enabled: bool, device: str, backend: str = "auto"
) -> DistributedContext:
    """Initialize a single-node DDP process from a ``torchrun`` environment."""
    observed_world_size = _environment_int("WORLD_SIZE", default=1)
    if not enabled:
        if observed_world_size > 1:
            raise ValueError(
                "torchrun requires --distributed for WORLD_SIZE greater than 1"
            )
        return DistributedContext(enabled=False, device=device)
    if device != "cuda":
        raise ValueError("--distributed requires CUDA; use --device cuda or auto")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    rank = _environment_int("RANK")
    world_size = _environment_int("WORLD_SIZE")
    local_rank = _environment_int("LOCAL_RANK")
    local_world_size = _environment_int("LOCAL_WORLD_SIZE", default=world_size)
    if world_size < 2:
        raise ValueError("--distributed requires torchrun with at least two processes")
    if local_world_size != world_size:
        raise ValueError("only single-node distributed training is supported")
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise ValueError("torchrun rank environment is inconsistent")
    if not torch.cuda.is_available():
        raise RuntimeError("--distributed requires at least one CUDA device")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} exceeds visible CUDA devices "
            f"({torch.cuda.device_count()})"
        )

    resolved_backend = _resolve_backend(backend)
    torch.cuda.set_device(local_rank)
    if dist.is_initialized():
        if dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise RuntimeError("existing process group does not match torchrun ranks")
        owns_process_group = False
    else:
        dist.init_process_group(backend=resolved_backend)
        owns_process_group = True
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=f"cuda:{local_rank}",
        backend=resolved_backend,
        owns_process_group=owns_process_group,
    )


def destroy_distributed(context: DistributedContext) -> None:
    """Release a process group created by :func:`initialize_distributed`."""
    if context.owns_process_group and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_for_training(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap a CUDA model in DDP only when the process group is active."""
    if not context.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the checkpoint/evaluation model below an optional DDP wrapper."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _environment_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"--distributed requires torchrun environment variable {name}")
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"torchrun environment variable {name} must be an integer"
        ) from error


def _resolve_backend(backend: str) -> str:
    if backend not in {"auto", "nccl", "gloo"}:
        raise ValueError("distributed backend must be auto, nccl or gloo")
    resolved_backend = "nccl" if backend == "auto" else backend
    if resolved_backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError("NCCL is unavailable; use --distributed-backend gloo")
    return resolved_backend
