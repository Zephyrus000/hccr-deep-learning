# Benchmarking

This package owns model profiling, operator complexity accounting, and the timing
schemas shared by explicit benchmark entrypoints.

`profiling.ModelProfiler` measures the selected training checkpoint using native
FP32 inference at batch size 1. `resource_profile.json` schema 2 retains parameter
counts and size, MACs/FLOPs with operator coverage, mean/p50/p95 latency,
throughput, peak CUDA allocation, warm-up/timing counts, device/runtime metadata,
and repetition summaries. Training attaches the selected checkpoint name and
SHA-256. `complexity.py` owns the operator FLOP formulas.
CUDA memory reports peak live process allocation during forward timing.
Training clears optimizer state and gradients before collecting the profile.

Automatic profiles no longer run batch sizes 8 and 32, optimized model copies,
synthetic PIL-to-logits timing, or a second randomly initialized full-class head.
They also omit p99. Analytical full-class parameter/operation projections remain
clearly separate from measurements. Use the explicit CLI below for deployment
comparisons, different batch sizes, and optimized execution.

Explicit benchmark CLI summaries use these keys:

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
