"""Shared benchmark result schema and timing summaries."""

from hccr.benchmarking.schema import (
    BENCHMARK_SCHEMA_VERSION,
    aggregate_timing_summaries,
    build_benchmark_protocol,
    summarize_timings,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "aggregate_timing_summaries",
    "build_benchmark_protocol",
    "summarize_timings",
]
