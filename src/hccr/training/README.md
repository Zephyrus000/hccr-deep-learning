# `hccr.training`

`hccr.training` owns the reproducible end-to-end train workflow. It resolves a
`TrainingConfig`, creates run-scoped data/model state, trains and evaluates the
best checkpoint, and persists artifacts needed for analysis and deployment.

## Public API

| Symbol | Purpose |
| --- | --- |
| `TrainingConfig` | Frozen, complete training configuration used by the CLI. |
| `run_training(config)` | Execute one training run and return best validation metrics. |
| `train_epoch(...)` | Run one optimizer epoch and collect loss, timing, gradient, and throughput metrics. |
| `EarlyStopping` | Track validation top-1 with patience and minimum delta. |
| `profile_model(...)` | Measure parameters, MACs, eager/optimized latency, and device metadata. |
| `write_training_diagnostics(...)` | Persist epoch diagnostic history. |

## Module map

| File | Responsibility |
| --- | --- |
| `workflow.py` | Validation, reproducibility setup, loaders, optimizer/scheduler, epoch loop, best-checkpoint evaluation, and summary. |
| `trainer.py` | Batch-level training and statistics. |
| `losses.py` | Cross-entropy construction with label smoothing. |
| `callbacks.py` | Early-stopping state. |
| `artifacts.py` | Checkpoints, metadata, digests, and recalibrated variants. |
| `diagnostics.py` | Complexity, latency, BatchNorm, stability, activation, and gradient diagnostics. |

## Minimal Python usage

```python
from pathlib import Path

from hccr.training import TrainingConfig, run_training

metrics = run_training(
    TrainingConfig(
        manifest_path=Path("data/processed/casia_hwdb/manifest.csv"),
        output_dir=Path("experiments"),
        num_classes=7186,
        device="auto",
    )
)
```

The CLI is normally preferred because it records the same resolved config while
providing argument validation and discoverable help.

## Run lifecycle

1. Validate configuration, resolve the device, and seed Python/Torch.
2. Create `experiments/<run-id>/` and write config/environment metadata.
3. Select an optional deterministic class subset and create train/validation
   loaders.
4. Build `EfficientHCCRNet`, profile eager/optimized inference, and initialize
   AdamW, scheduler, loss, and early stopping.
5. Train/evaluate each epoch, persist curves/diagnostics, and replace the best
   checkpoint on non-decreasing top-1.
6. Reload the best checkpoint for full validation diagnostics.
7. Optionally recalibrate BatchNorm on a deterministic training subset and
   evaluate the recalibrated copy.
8. Append one row to `experiment_summary.csv` and return best metrics.

Schedulers are `none`, `cosine`, and validation-based `plateau`. CosFace and
ArcFace margins can be warmed from zero to their configured value independently
of label smoothing.

## Artifacts

Important run files include:

- `config.json`, `metadata.json`, `run.log`
- `checkpoint.pt`, `checkpoint_metadata.json`, `labels.json`
- `metrics.json`, `curves.json`, `training_diagnostics.json`
- `resource_profile.json`, `validation_stability.json`
- preprocessing/augmentation galleries and evaluation reports

With BN recalibration enabled, the original checkpoint is preserved and the run
also writes `checkpoint_recalibrated.pt`, recalibration metadata, and reports
under `bn_recalibrated/`.

When workers are enabled, CUDA uses the `spawn` start method to avoid inherited
CUDA state. Persistent workers, prefetching, and a timeout are configurable;
with zero workers, multiprocessing-only options are omitted from the loader.
