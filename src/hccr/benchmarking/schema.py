"""Canonical schema helpers shared by benchmark entrypoints."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

BENCHMARK_SCHEMA_VERSION = 1


def summarize_timings(
    timings_ms: Sequence[float], batch_size: int = 1
) -> dict[str, float]:
    """Summarize one sequence of latency measurements using canonical keys."""
    if not timings_ms:
        raise ValueError("timings_ms must contain at least one measurement")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mean_latency = statistics.mean(timings_ms)
    if mean_latency <= 0:
        raise ValueError("latency measurements must have a positive mean")
    return {
        "latency_mean_ms": mean_latency,
        "latency_p50_ms": _percentile(timings_ms, 50),
        "latency_p95_ms": _percentile(timings_ms, 95),
        "latency_p99_ms": _percentile(timings_ms, 99),
        "samples_per_second": batch_size * 1000 / mean_latency,
    }


def aggregate_timing_summaries(
    repeat_summaries: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Aggregate repeated benchmarks in the same form as resource profiles."""
    if not repeat_summaries:
        raise ValueError("repeat_summaries must contain at least one result")
    metric_names = (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "samples_per_second",
    )
    return {
        **{
            name: statistics.median(summary[name] for summary in repeat_summaries)
            for name in metric_names
        },
        "repeat_summaries": [dict(summary) for summary in repeat_summaries],
    }


def build_benchmark_protocol(
    *,
    device: str,
    batch_size: int,
    warmup_iterations: int,
    timed_iterations: int,
    repetitions: int,
    scope: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared protocol envelope used by benchmark reports."""
    if min(batch_size, warmup_iterations, timed_iterations, repetitions) < 1:
        raise ValueError("benchmark protocol counts must be positive")
    protocol = {
        "device": device,
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "repetitions": repetitions,
        "aggregation": (
            "single_run" if repetitions == 1 else "median_of_repetition_summaries"
        ),
        "execution_context": "torch.inference_mode",
        "model_mode": "eval",
        "scope": scope,
    }
    if metadata is not None:
        overlap = protocol.keys() & metadata.keys()
        if overlap:
            raise ValueError(
                "protocol metadata cannot override canonical fields: "
                + ", ".join(sorted(overlap))
            )
        protocol.update(metadata)
    return protocol


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]
