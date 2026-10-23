"""Bounded, deterministic caches of prepared fit images for one training process."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hccr.preprocessing import EvalPreprocessor

_GIB = 1024**3


def estimate_trainable_state_bytes(model: nn.Module | None) -> int:
    """Reserve future gradients and two Adam state tensors at least in FP32."""
    if model is None:
        return 0
    return 3 * sum(
        parameter.numel() * max(parameter.element_size(), 4)
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def plan_cache_admission(
    required_bytes: int,
    available_bytes: int | None,
    max_bytes: int,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Check the hard cache budget and remaining free memory before decoding."""
    if min(required_bytes, max_bytes, reserve_bytes) < 0:
        raise ValueError("memory sizes must be nonnegative")
    usable = (
        max(0, available_bytes - reserve_bytes) if available_bytes is not None else 0
    )
    admitted = available_bytes is not None and required_bytes <= min(max_bytes, usable)
    if available_bytes is None:
        reason = "available_memory_unknown"
    elif required_bytes > max_bytes:
        reason = "cache_budget_exceeded"
    elif required_bytes > usable:
        reason = "memory_reserve_would_be_used"
    else:
        reason = "within_budget"
    return {
        "admitted": admitted,
        "reason": reason,
        "required_bytes": required_bytes,
        "available_bytes": available_bytes,
        "usable_bytes": usable,
        "max_bytes": max_bytes,
        "reserve_bytes": reserve_bytes,
    }


def _available_cpu_memory_bytes() -> int | None:
    available = None
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_pagefile", ctypes.c_ulonglong),
                ("available_pagefile", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available = int(status.available_physical)
    else:
        try:
            values = Path("/proc/meminfo").read_text().splitlines()
            available = next(
                int(line.split()[1]) * 1024
                for line in values
                if line.startswith("MemAvailable:")
            )
        except (OSError, StopIteration, ValueError):
            with suppress(AttributeError, OSError, ValueError):
                available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            limit = int(Path(limit_path).read_text().strip())
            used = int(Path(usage_path).read_text().strip())
        except (OSError, ValueError):
            continue
        cgroup_available = max(0, limit - used)
        available = (
            cgroup_available if available is None else min(available, cgroup_available)
        )
    return available


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def _source_preprocessor(dataset, image_size: int | None) -> EvalPreprocessor:
    source = dataset
    while isinstance(source, Subset):
        source = source.dataset
    transform = getattr(source, "transform", None)
    if type(transform) is not EvalPreprocessor:
        raise ValueError(
            "training caches require deterministic EvalPreprocessor inputs"
        )
    rows = getattr(source, "rows", None)
    if rows is None or any(
        row.get("split") != "train"
        or row.get("official_partition", "train") not in {"", "train"}
        for row in rows
    ):
        raise ValueError("training caches may contain only fit/train rows")
    if len(dataset) < 1:
        raise ValueError("training cache source must not be empty")
    if image_size is not None and image_size != transform.image_size:
        raise ValueError("cache image_size disagrees with source preprocessing")
    if transform.image_size < 1:
        raise ValueError("cache image_size must be positive")
    return transform


class PreparedTrainingCache:
    """An epoch-shuffled uint8 store that yields normalized training batches."""

    def __init__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        seed: int,
        cache_info: dict[str, Any],
    ) -> None:
        self.images = images
        self.targets = targets
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.epoch = 0
        self.cache_info = cache_info

    def __len__(self) -> int:
        return math.ceil(len(self.targets) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("cache epoch must be a nonnegative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[str]]]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.targets), generator=generator)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size].to(self.images.device)
            images = self.images[indices].to(self.device).float().mul_(1.0 / 255)
            targets = self.targets[indices].to(self.device)
            yield images, targets, [""] * len(targets)


def _fill_cache(
    dataset,
    batch_size: int,
    image_size: int,
    cache_device: torch.device,
    training_device: torch.device,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.empty(
        (len(dataset), 1, image_size, image_size),
        dtype=torch.uint8,
        device=cache_device,
    )
    targets = torch.empty(len(dataset), dtype=torch.int64, device=cache_device)
    options = {}
    if num_workers:
        options.update(
            multiprocessing_context="spawn",
            prefetch_factor=1,
            persistent_workers=False,
            worker_init_fn=_worker_init,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=training_device.type == "cuda",
        generator=torch.Generator(device="cpu").manual_seed(0),
        **options,
    )
    offset = 0
    for batch, labels, _metadata in loader:
        expected = (len(labels), 1, image_size, image_size)
        if batch.dtype != torch.float32 or tuple(batch.shape) != expected:
            raise ValueError(
                "cache source must yield prepared BCHW grayscale FP32 inputs"
            )
        end = offset + len(labels)
        images[offset:end].copy_(batch.mul(255).round().clamp_(0, 255).to(torch.uint8))
        targets[offset:end].copy_(labels.to(torch.int64))
        offset = end
    if offset != len(dataset):
        raise ValueError("cache source size changed during preparation")
    return images, targets


def build_training_cache(
    dataset,
    batch_size: int,
    device: str | torch.device,
    *,
    placement: str = "auto",
    max_gib: float = 32,
    reserve_gib: float = 16,
    seed: int = 7,
    num_workers: int = 0,
    image_size: int | None = None,
    model: nn.Module | None = None,
) -> tuple[PreparedTrainingCache | None, dict[str, Any]]:
    """Admit and prepare a fit-only cache, or explicitly report streaming fallback.

    Auto tries GPU storage for CUDA training, then host RAM, then streaming.
    Explicit CPU/CUDA requests fail before decoding if memory admission fails.
    The reserve is a conservative guard, not a measured activation-memory bound.
    """
    if placement not in {"auto", "cpu", "cuda"}:
        raise ValueError("cache placement must be auto, cpu, or cuda")
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(num_workers) is not int
        or num_workers < 0
    ):
        raise ValueError("cache batch size must be positive and workers nonnegative")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("cache seed must be an integer in [0, 2**63)")
    if not math.isfinite(max_gib) or max_gib <= 0:
        raise ValueError("max_gib must be finite and positive")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    transform = _source_preprocessor(dataset, image_size)
    training_device = torch.device(device)
    if training_device.type not in {"cpu", "cuda"}:
        raise ValueError("prepared training caches support CPU or CUDA training")
    if placement == "cuda" and training_device.type != "cuda":
        raise ValueError("CUDA cache placement requires a CUDA training device")
    count, size = len(dataset), transform.image_size
    workers = min(num_workers, 4)
    pixels = min(batch_size, count) * size * size
    payload_bytes = count * (size * size + 8)
    order_bytes = count * 8
    scratch_bytes = pixels * (8 * 4 + 2) + min(batch_size, count) * 32
    staging_bytes = pixels * 4 * (workers + 2)
    future_state_bytes = estimate_trainable_state_bytes(model)
    info = {
        "schema_version": 1,
        "requested_placement": placement,
        "placement": "stream",
        "status": "stream_fallback",
        "sample_count": count,
        "image_size": size,
        "dtype": "uint8",
        "target_dtype": "int64",
        "payload_bytes": payload_bytes,
        "order_bytes": order_bytes,
        "batch_scratch_bytes": scratch_bytes,
        "staging_bytes": staging_bytes,
        "future_trainable_state_bytes": future_state_bytes,
        "max_bytes": int(max_gib * _GIB),
        "reserve_bytes": int(reserve_gib * _GIB),
        "preprocessing": asdict(transform),
        "seed": seed,
        "build_workers": workers,
        "attempts": [],
        "build_seconds": 0.0,
    }
    candidates = (
        (["cuda", "cpu"] if training_device.type == "cuda" else ["cpu"])
        if placement == "auto"
        else [placement]
    )
    started = time.perf_counter()
    for candidate in candidates:
        cache_device = training_device if candidate == "cuda" else torch.device("cpu")
        if candidate == "cuda":
            available = int(torch.cuda.mem_get_info(training_device)[0])
        else:
            available = _available_cpu_memory_bytes()
        required = payload_bytes + order_bytes + scratch_bytes
        if candidate == "cpu":
            required += staging_bytes
        if candidate == "cuda" or training_device.type == "cpu":
            required += future_state_bytes
        admission = plan_cache_admission(
            required, available, info["max_bytes"], info["reserve_bytes"]
        )
        attempt = {"placement": candidate, **admission}
        info["attempts"].append(attempt)
        if not admission["admitted"]:
            if placement != "auto":
                raise MemoryError(
                    f"{candidate} training cache refused: {admission['reason']}"
                )
            continue
        allocation_error = None
        try:
            images, targets = _fill_cache(
                dataset, batch_size, size, cache_device, training_device, workers
            )
        except (MemoryError, RuntimeError) as error:
            if not isinstance(error, (MemoryError, torch.OutOfMemoryError)) and not any(
                text in str(error).lower()
                for text in (
                    "out of memory",
                    "cannot allocate memory",
                    "can't allocate memory",
                )
            ):
                raise
            attempt.update(admitted=False, reason="allocation_failed", error=str(error))
            allocation_error = str(error)
        if allocation_error is not None:
            if placement != "auto":
                raise MemoryError(
                    f"{candidate} training cache allocation failed: {allocation_error}"
                )
            gc.collect()
            if candidate == "cuda":
                torch.cuda.empty_cache()
            continue
        info.update(
            placement=candidate,
            status="cached",
            build_seconds=time.perf_counter() - started,
        )
        cache = PreparedTrainingCache(
            images,
            targets,
            batch_size=batch_size,
            device=training_device,
            seed=seed,
            cache_info=info,
        )
        return cache, info
    info["build_seconds"] = time.perf_counter() - started
    return None, info
