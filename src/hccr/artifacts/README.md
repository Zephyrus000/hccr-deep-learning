# Run artifacts

`RunArtifacts` owns the output directory of one training run. It writes resolved
configuration, provenance, curves, checkpoints, and final holdout metadata.
`SummaryTable` appends directly measured results to a CSV. Serialization,
provenance, checkpoint persistence, and table construction live in separate modules.

Training writes `experiments/<run-id>/`; sweeps write their plans, status, and
aggregates under `experiments/sweeps/<experiment-id>/`. Explicit analysis scripts
choose their own output directory, often under `artifacts/`. No package scans
both trees to merge or normalize their contents. The former `hccr.evidence`
package and `scripts/build_evidence_bundle.py` have been removed.

Historical JSON and CSV files are not rewritten. When an existing
`experiment_summary.csv` has a different header, new runs append to
`experiment_summary.v2.csv`. Pass that file explicitly to comparison tools.
A versioned table with an incompatible header raises an error.

The primary metrics are Top-1, Top-5, macro recall, parameter count and size,
MACs, FLOPs, batch-one mean/p50/p95 latency, throughput, and CUDA memory usage.
Keep validation, test, and ICDAR results separate; absent test values stay empty.
Head/mid/tail recall and calibration error remain available for class-frequency
and confidence analysis. Per-component complexity supports classifier ablations.
They use the existing evaluation pass and do not trigger another benchmark.

The paper in `docs/repo_articles/main.tex` uses Top-1, Top-5, macro recall,
parameters, MACs/FLOPs, and mean/p50/p95 latency. Compact HCCR research also reports
[accuracy, model size, and CPU time](https://arxiv.org/abs/1702.07975).
[ShuffleNet V2](https://arxiv.org/abs/1807.11164) motivates measuring speed on the
target hardware rather than inferring it from operation counts.

See [profiling](../benchmarking/README.md) for removed automatic measurements and
the protocol retained in each profile. The raw configuration, checkpoint hashes,
device metadata, and repeated timing summaries remain the audit record.
