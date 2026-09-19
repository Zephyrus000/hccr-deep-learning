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

## Reviewer protocol

The JSON report is canonical. A comparable row must preserve all of these
fields: schema version, device and hardware identity, dtype, execution mode,
batch/input shape, warm-up iterations, timed iterations, repetitions, timing
scope, thread counts, and aggregation convention. CSV exports are derived
views and are not sufficient to audit a measurement by themselves.

Use `mode=eager` for the main architecture comparison. It permits only
`model.eval()`, `torch.inference_mode()`, and the angular classifier's shared
normalized-weight cache. Use `mode=optimized` only for a separately labelled
deployment study because it can fold Conv--BatchNorm and reparameterized
branches. Never compare CPU and CUDA, different CPU thread counts, different
hosts, or eager and optimized measurements in one latency ranking.

The specialized CPU script selects its fixed five-model × three-seed matrix
from `experiment_summary.csv`, records that manifest and a randomized execution
order, and measures the same LMDB sample for every checkpoint. It should be run
once per target CPU host and execution mode. The general CLI remains the tool
for an arbitrary single checkpoint.

Passing `--runs <run-dir> ...` switches the specialized script to an explicit
run manifest and bypasses matrix selection. Every explicit run is still loaded
from checkpoint metadata, checked for a checkpoint, and recorded in the output
manifest. A single timed iteration is valid and reports zero latency sample
standard deviation rather than failing.
