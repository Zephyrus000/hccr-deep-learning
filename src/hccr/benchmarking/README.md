# Benchmarking

This package defines the common timing and report contract shared by benchmark
entrypoints. It contains no model, dataset, or CLI orchestration.

Canonical latency summaries use these keys:

- `latency_mean_ms`
- `latency_p50_ms`
- `latency_p95_ms`
- `latency_p99_ms`
- `samples_per_second`

Repeated measurements additionally contain `repeat_summaries` and aggregate
each metric with the median. Reports declare `schema_version` and a
`benchmark_protocol` describing the device, iteration counts, aggregation, and
measurement scope.

[`hccr.commands.benchmark`](../commands/benchmark.py) provides the general
single-checkpoint CPU/CUDA command. The repository-level
`scripts/benchmark_cpu_lmdb.py` remains a specialized multi-run CPU benchmark
that can be copied to or executed on different CPU hosts while retaining this
schema.
