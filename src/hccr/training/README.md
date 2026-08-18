# `hccr.training`

This package orchestrates reproducible training and creates all run artifacts.

| File | Responsibility |
| --- | --- |
| `workflow.py` | `TrainingConfig`, data/model setup, AdamW, scheduler, early stopping, validation and run summary. |
| `trainer.py` | One tqdm-backed train epoch and loss/gradient/throughput statistics. |
| `callbacks.py` | `EarlyStopping`, monitored on validation top-1. |
| `diagnostics.py` | Parameters, MAC/FLOP estimate, inference percentiles and top-level activation/gradient diagnostics. |
| `artifacts.py` | Checkpoint state, checkpoint metadata and manifest digest. |

`scheduler` supports `none`, `cosine`, and `plateau`. Early stopping uses
`early_stopping_patience` and `early_stopping_min_delta`; CLI value `0` disables
it. A successful run is written to `experiments/<run-id>/`, never directly into
the experiment root.
