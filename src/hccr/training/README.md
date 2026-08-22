# `hccr.training`

This package orchestrates reproducible training and creates all run artifacts.

| File | Responsibility |
| --- | --- |
| `workflow.py` | `TrainingConfig`, data/model setup, AdamW, scheduler, early stopping, validation and run summary. |
| `trainer.py` | One tqdm-backed train epoch and loss/gradient/throughput statistics. |
| `losses.py` | Cross-entropy construction with explicit label smoothing. |
| `callbacks.py` | `EarlyStopping`, monitored on validation top-1. |
| `diagnostics.py` | Parameters, MAC/FLOP estimate, repeated inference benchmarks, BatchNorm health/recalibration and activation/gradient diagnostics. |
| `artifacts.py` | Checkpoint state, checkpoint metadata and manifest digest. |

Preprocessing is fixed to black-on-white grayscale after the ablation sweep.

`scheduler` supports `none`, `cosine`, and `plateau`. Early stopping uses
`early_stopping_patience` and `early_stopping_min_delta`; CLI value `0` disables
it. A successful run is written to `experiments/<run-id>/`, never directly into
the experiment root.

Inference benchmarks default to 20 warm-up iterations, 200 timed iterations
and 5 repetitions for each batch size. Aggregate latency is the median of the
per-repetition summaries; `resource_profile.json` retains every repetition and
the CPU/CUDA runtime metadata. The three counts are configurable from the CLI.
The same profile separates total, backbone, and classifier-head parameters and
MACs so subset and full-class architectures remain directly comparable. It
declares unsupported MAC operator types, projects the classifier to 7,186
classes, and records synthetic raw-PIL-to-logits batch-1 latency in addition to
model-only latency.
Resolved width and `stage_depths` are stored in checkpoint metadata and the
experiment summary. Width and all three stage depths must be positive.

`classification_head` supports `cosface` and `arcface`. Both angular heads
normalize embeddings and weights, use target-free scaled cosine logits
for validation/inference, and apply the configured target margin only during
training. `margin_warmup_epochs` ramps the margin from zero to full strength;
`label_smoothing` remains an independent cross-entropy ablation.

Every epoch records BatchNorm running-stat health and writes
`validation_stability.json`. `--bn-recalibration-batches N` additionally
recomputes BatchNorm statistics once from the best raw checkpoint after
training, using a fixed deterministic training subset. The raw checkpoint is
preserved, while `checkpoint_recalibrated.pt`, its metadata,
`bn_recalibration.json`, and full reports under `bn_recalibrated/` capture the
deployable recalibrated variant. The option is disabled by default because it
adds one additional complete validation pass per run.

When multiprocessing is enabled, CUDA training resolves the DataLoader start
method to `spawn`. This prevents workers from inheriting CUDA/profiler state
from the parent process. Persistent workers avoid repeated startup per epoch,
while a configurable prefetch factor and timeout bound Docker shared-memory use
and convert worker stalls into actionable errors.
