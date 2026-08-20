# `hccr.training`

This package orchestrates reproducible training and creates all run artifacts.

| File | Responsibility |
| --- | --- |
| `workflow.py` | `TrainingConfig`, data/model setup, AdamW, scheduler, early stopping, validation and run summary. |
| `trainer.py` | One tqdm-backed train epoch and loss/gradient/throughput statistics. |
| `callbacks.py` | `EarlyStopping`, monitored on validation top-1. |
| `diagnostics.py` | Parameters, MAC/FLOP estimate, repeated inference benchmarks, BatchNorm health/recalibration and activation/gradient diagnostics. |
| `artifacts.py` | Checkpoint state, checkpoint metadata and manifest digest. |

`scheduler` supports `none`, `cosine`, and `plateau`. Early stopping uses
`early_stopping_patience` and `early_stopping_min_delta`; CLI value `0` disables
it. A successful run is written to `experiments/<run-id>/`, never directly into
the experiment root.

Inference benchmarks default to 20 warm-up iterations, 200 timed iterations
and 5 repetitions for each batch size. Aggregate latency is the median of the
per-repetition summaries; `resource_profile.json` retains every repetition and
the CPU/CUDA runtime metadata. The three counts are configurable from the CLI.

Every epoch records BatchNorm running-stat health and writes
`validation_stability.json`. `--bn-recalibration-batches N` additionally
recomputes BatchNorm statistics from a fixed deterministic training subset and
records the recalibrated validation delta without changing the trained model or
checkpoint. The option is disabled by default because it adds another complete
validation pass per epoch.
