# `hccr.training`

`hccr.training` owns the reproducible end-to-end train workflow. It resolves a
`TrainingConfig`, creates run-scoped data/model state, trains and evaluates the
best checkpoint, and persists artifacts needed for analysis and deployment.

## Public API

| Symbol | Purpose |
| --- | --- |
| `TrainingConfig` | Frozen, complete training configuration used by the CLI. |
| `run_training(config)` | Execute one training run and return selected validation or final held-out test metrics according to policy. |
| `train_epoch(...)` | Run one optimizer epoch and collect loss, timing, gradient, and throughput metrics. |
| `EarlyStopping` | Track validation top-1 with patience and minimum delta. |
| `profile_model(...)` | Measure parameters, MACs, eager/optimized latency, and device metadata. |
| `write_training_diagnostics(...)` | Persist epoch diagnostic history. |
| `resolve_precision(...)` | Resolve FP32, FP16+GradScaler, or BF16 autocast safely per device. |

## Module map

| File | Responsibility |
| --- | --- |
| `workflow.py` | Validation, reproducibility setup, loaders, optimizer/scheduler, epoch loop, best-checkpoint evaluation, and summary. |
| `trainer.py` | Batch-level training and statistics. |
| `losses.py` | Cross-entropy construction with label smoothing. |
| `callbacks.py` | Early-stopping state. |
| `artifacts.py` | Checkpoints, metadata, digests, and recalibrated variants. |
| `diagnostics.py` | Complexity, latency, BatchNorm, stability, activation, and gradient diagnostics. |
| `precision.py` | AMP policy, autocast context, FP16 gradient scaling, and run metadata. |

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
   loaders from the frozen manifest. A test loader is created only for
   `evaluation_policy="final_test"`.
4. Build `EfficientHCCRNet`, profile eager/optimized inference, and initialize
   AdamW, scheduler, loss, and early stopping.
5. Train and evaluate the `validation` split every epoch; use its top-1 for
   checkpoint selection, plateau scheduling, early stopping, curves, and
   diagnostics.
6. Reload the selected checkpoint. In `validation_only`, stop without opening
   test data; in `final_test`, evaluate the held-out test split once.
7. Optionally recalibrate BatchNorm on a deterministic training subset and
   choose the recalibrated copy only when validation top-1 improves.
8. Append one row to `experiment_summary.csv` and return best metrics.

When `distributed=True`, launch the CLI with `torchrun` and one process per
CUDA GPU. Each process receives a deterministic `DistributedSampler` shard of
the training data, while rank 0 alone evaluates validation/final test and
writes artifacts. `batch_size` remains per GPU; the persisted plan and summary
also record `world_size` and `effective_global_batch_size` for comparable runs.
The sequential experiment runner exposes the same launcher with
`torchrun_nproc_per_node: 2` plus `base_args.distributed: true` in its YAML.

`precision="float32"` remains the default for reproducibility. On a supported
CUDA GPU, set `precision="bfloat16"` (recommended for RTX 4090) or
`precision="auto"`; the latter selects BF16 when supported and otherwise FP16
with `GradScaler`. The convolutional backbone uses autocast, whereas angular
logits, loss, validation/test metrics, and gradient diagnostics use FP32. Each
run records requested and resolved precision in its metadata and summary.

Schedulers are `none`, per-optimizer-step `cosine`, and validation-based
`plateau`. The default batch-aware plan treats batch 64 as the reference:
larger batches receive enough physical epochs to retain the reference optimizer
update budget, scale AdamW learning rate by square root, and use a cosine-only
linear warm-up over the first 5% of optimizer steps. Persisted `batch_training_plan.json`,
checkpoint metadata, curves, and `experiment_summary.csv` record both requested
and resolved values. Set `optimizer_step_policy="configured_epochs"` only to
reproduce legacy epoch semantics. CosFace and ArcFace margins can be warmed from
zero to their configured value independently of label smoothing.

## Artifacts

Important run files include:

- `config.json`, `metadata.json`, `run.log`
- `batch_training_plan.json` (requested/resolved update and learning-rate plan)
- `checkpoint.pt`, `checkpoint_metadata.json`, `labels.json`
- `metrics.json`, `curves.json`, `training_diagnostics.json`
- `resource_profile.json`, `validation_stability.json`, and final test reports
- preprocessing/augmentation galleries and evaluation reports

With BN recalibration enabled, the original checkpoint is preserved. The
recalibrated variant is compared against it on validation and is saved as
`checkpoint_recalibrated.pt` only when it improves validation top-1. The chosen
variant is then evaluated once on the held-out test split.

When workers are enabled, CUDA uses the `spawn` start method to avoid inherited
CUDA state. Persistent workers, prefetching, and a timeout are configurable;
with zero workers, multiprocessing-only options are omitted from the loader.

The run log prints validation top-1/top-5/macro/tail recall every epoch. Test
metrics are emitted only in `final_test` after training for the selected
checkpoint. Writer IDs are unavailable in the current Kaggle-derived image
export, so artifacts record writer separation as `not_verifiable`, not as a
verified writer-disjoint split.

Paper screening runs should use `validation_only` and `reproducibility_mode="strict"`.
Run `final_test` only after architecture and hyperparameters are frozen. Run
metadata records deterministic flags, key package/runtime versions, the full
Git commit, dirty state, and a working-tree content digest.
