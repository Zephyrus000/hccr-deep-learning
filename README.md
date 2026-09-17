# HCCR Deep Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An offline training and evaluation toolkit for isolated handwritten Chinese
character recognition (HCCR). The repository contains a compact PyTorch CNN,
manifest-based CASIA-HWDB data handling, reproducible experiment artifacts,
and latency-aware model evaluation.

> [!IMPORTANT]
> CASIA-HWDB is not distributed with this repository. Obtain the dataset under
> its own terms and keep it under the ignored `data/` directory.

## Highlights

- Compact depthwise-separable CNN with CosFace or ArcFace classification heads.
- Deterministic manifest generation and validation for folder-exported data.
- Reproducible training runs with checkpoints, metrics, plots, logs, and
  environment metadata.
- Class-subset experiments for faster architecture screening.
- Multi-seed and variant sweeps with resumable jobs and aggregate reports.
- Eager and optimized inference benchmarks for CPU and CUDA deployments.
- Batch-normalization recalibration, calibration metrics, and validation error
  diagnostics.

The executable CLI implements `train`, `benchmark`, and `compare-runs` through
separate command modules. `validate`, `deploy`, `prepare-data`, `evaluate`,
`predict`, and `tune` are explicit workflow scaffolds; use the Python APIs
documented below until those commands are implemented.

## Architecture

```text
folder-exported images
        │
        ▼
dataset audit + manifest ──► deterministic preprocessing
        │                              │
        └──────────────────────────────┤
                                       ▼
                            EfficientHCCRNet training
                                       │
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
              experiment artifacts              evaluation reports
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                           optimized inference benchmark
```

`EfficientHCCRNet` uses a convolutional stem, three configurable
depthwise-separable residual stages, global average pooling, an embedding
projection, and an angular classifier. Inference optimization folds
Conv-BatchNorm pairs, collapses optional training-only depthwise branches,
removes eval-only module hops, and caches normalized classifier weights without
modifying the training checkpoint.

## Installation

Requirements:

- Python 3.12 or newer
- A CUDA-capable PyTorch installation for GPU training (optional)
- Git for run metadata

Create an isolated environment and install the project:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The requirements file points pip at the CUDA 13.0 PyTorch wheel index. Install
the appropriate PyTorch build first if your platform needs a different CUDA or
CPU-only wheel.

Verify the installation:

```bash
hccr --version
hccr --help
```

## Dataset preparation

The manifest builder expects one directory per Unicode label:

```text
data/raw/
├── CASIA-HWDB_Train/Train/
│   ├── 一/*.png
│   ├── 丁/*.png
│   └── ...
└── CASIA-HWDB_Test/Test/
    ├── 一/*.png
    ├── 丁/*.png
    └── ...
```

Audit the PNG files and generate a frozen manifest:

```bash
python scripts/build_dataset_manifest.py \
  --data-root data/raw \
  --train-dir data/raw/CASIA-HWDB_Train/Train \
  --test-dir data/raw/CASIA-HWDB_Test/Test \
  --output-dir data/processed/casia_hwdb
```

If only the raw ZIP is available, generate the same artifacts without
extracting it:

```bash
python scripts/build_dataset_manifest.py \
  --source-zip raw.zip \
  --zip-prefix raw \
  --zip-train-dir CASIA-HWDB_Train/Train \
  --zip-test-dir CASIA-HWDB_Test/Test \
  --output-dir data/processed/casia_hwdb
```

The ZIP prefix and both partition directories are configurable. Generated
`source_file` values are relative to `--zip-prefix`, so the resulting manifest
can be passed directly to `build_lmdb_dataset.py --source-zip ...` with the same
prefix.

Generated files include `manifest.csv`, `labels.json`, `audit_report.json`, and
`invalid_images.json`. The training pipeline requires these manifest columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable unique sample identifier. |
| `source_file` | Image path relative to the detected data root. |
| `writer_id` | Writer identity when available; may be empty. |
| `unicode_label` | Character represented by the sample. |
| `class_id` | Integer class identifier. |
| `split` | `train`, `validation`, or `test`. |

Pack the frozen dataset into a single LMDB file before large training runs:

```bash
python scripts/build_lmdb_dataset.py \
  --manifest data/processed/casia_hwdb/manifest.csv
```

If the raw dataset is a ZIP, avoid extracting millions of files by adding
`--source-zip raw.zip`. Use `--zip-prefix raw` (or another directory name) when
archive members have a path prefix not present in manifest `source_file` values.

This creates `data/processed/casia_hwdb/images.lmdb`. Training discovers it
automatically; pass `--dataset-backend lmdb` for full-class runs so a missing or
stale store fails fast. Use `--lmdb-path` only when the store is elsewhere.

Do not commit the raw dataset, generated manifests, checkpoints, or local
experiment output.

## Training

Start with a small deterministic class subset to validate the pipeline:

```bash
hccr train \
  --manifest data/processed/casia_hwdb/manifest.csv \
  --output-dir experiments \
  --max-classes 200 \
  --class-subset-seed 7 \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --device auto
```

For a quick overfitting check:

```bash
hccr train \
  --manifest data/processed/casia_hwdb/manifest.csv \
  --max-classes 20 \
  --overfit-samples 64 \
  --overfit-check \
  --epochs 30
```

Use `hccr train --help` for the complete option list. Important controls are:

| Option | Purpose |
| --- | --- |
| `--width`, `--stage-depths` | Backbone capacity. |
| `--stem-stride` | Compare early stride-2 downsampling with stroke-preserving stride 1. |
| `--reparameterize-depthwise` | Train 3×3/1×3/3×1 depthwise branches and fuse them for deploy; enabled by default. Use `--no-reparameterize-depthwise` for a control run. |
| `--classification-head` | `cosface` or `arcface`. |
| `--margin-warmup-ratio` | Fraction of epochs used to ramp the angular margin multiplier from 0 to 1; promoted default is `0.2`. |
| `--reference-batch-size` | Reference batch size for equalizing training update budget; default `64`. |
| `--optimizer-step-policy` | `reference_batch` (default) extends larger-batch runs until they receive at least the reference number of AdamW updates. `configured_epochs` preserves legacy physical-epoch behavior. |
| `--learning-rate-scaling` | AdamW LR scaling from the reference batch: `sqrt` (default), `linear`, or `none`. |
| `--lr-warmup-ratio` | Initial cosine-scheduler fraction of optimizer steps used for linear LR warm-up; default `0.05`. |
| `--image-size` | Square model input resolution. |
| `--max-classes` | Deterministic fast-benchmark class subset. |
| `--evaluation-policy` | `validation_only` for selection/ablation, or `final_test` for one held-out evaluation after selection. |
| `--reproducibility-mode` | `seeded` or strict deterministic-algorithm execution. |
| `--scheduler` | `none`, `cosine`, or validation-based `plateau`. |
| `--num-workers` | DataLoader concurrency. |
| `--dataset-backend` | `auto`, direct `filesystem`, or required `lmdb` reads. |
| `--lmdb-path` | Optional non-default LMDB file path. |
| `--bn-recalibration-batches` | Post-training BN-statistics recalibration. |
| `--device` | `auto`, `cpu`, or `cuda`. |
| `--precision` | `float32` (default), `float16`, `bfloat16`, or `auto`. Use BF16 on RTX 4090/Ampere-or-newer CUDA GPUs; FP16 uses gradient scaling. |

Each run is written to `experiments/<run-id>/`. Core artifacts include:

- `checkpoint.pt`, `checkpoint_metadata.json`, and `labels.json`
- `config.json`, `metadata.json`, `metrics.json`, and `curves.json`
- `resource_profile.json` and `training_diagnostics.json`
- learning curves, reliability diagrams, per-class metrics, and error galleries
- a run-scoped log and a cross-run `experiment_summary.csv`

The manifest builder moves a deterministic 10% of each uploader-provided
training class into `validation`; the remaining 90% is `train`. This is an
image-level split because writer identity is unavailable in the current Kaggle
image export. The pipeline therefore records writer-disjointness as
`not_verifiable`; it does not claim that train, validation, and test writers are
distinct. Checkpoint selection, plateau scheduling, early stopping, and
learning curves use only validation.

Use `--evaluation-policy validation_only` for ablations and hyperparameter
selection. In that mode the test dataset and loader are not constructed. After
all choices are frozen, use `--evaluation-policy final_test` to evaluate the
selected checkpoint once and write the result separately in `metrics.json` and
checkpoint metadata.

If BN recalibration is enabled, its variant is compared on validation first;
only the selected variant receives the single final test evaluation.

The default batch-aware policy makes `--epochs` an update-budget request rather
than only a physical-epoch count. For example, with a 64-sample reference,
`--batch-size 256 --epochs 20` trains roughly four physical epochs for each
configured epoch so it receives the same number of optimizer updates as batch
64. The resolved plan is saved to `batch_training_plan.json`; use
`--optimizer-step-policy configured_epochs` only when reproducing an older run.

Mixed precision keeps model weights and persisted checkpoints in FP32. When
selected, autocast accelerates the convolutional forward pass; CosFace/ArcFace
normalization, cross-entropy, evaluation metrics, and diagnostics stay in FP32.
`metadata.json`, checkpoint metadata, `metrics.json`, and the experiment summary
record both the requested and resolved precision so CPU/FP32/BF16 runs are not
mistakenly compared as identical conditions.

### Single-node multi-GPU training

The training command supports standard single-node PyTorch Distributed Data
Parallel (DDP). Launch exactly one process per visible GPU with `torchrun`; do
not start two ordinary `hccr train` processes. For two GPUs:

```bash
torchrun --standalone --nproc-per-node=2 -m hccr train \
  --distributed \
  --device cuda \
  --manifest data/processed/casia_hwdb/manifest.csv \
  --dataset-backend lmdb \
  --output-dir experiments \
  --image-size 96 \
  --batch-size 128 \
  --reference-batch-size 256 \
  --num-workers 16
```

`--batch-size` is **per GPU**: the example has an effective global batch of
256. This value, the DDP world size, rank-zero device, and resolved learning
rate are recorded in `batch_training_plan.json`, `metadata.json`, and checkpoint
metadata. If you retain `--batch-size 256` on two GPUs, the global batch becomes
512 and the configured LR scaling policy is applied from that value.

DDP shards only the training dataset. Rank 0 runs validation and the one-time
final test on the unwrapped checkpoint model, then writes the sole run folder,
logs, profiles, checkpoints, and reports. This keeps evaluation artifacts
identical to a single-GPU run and prevents workers from duplicating files.

The sequential experiment runner can launch the same command for every YAML
job. Add the following top-level field and base argument to a sweep config:

```yaml
torchrun_nproc_per_node: 2
base_args:
  distributed: true
```

It emits `python -m torch.distributed.run --standalone --nproc-per-node=2 ...`
for each job. Omit both fields for the existing one-GPU sequential behavior.

## Experiment sweeps

Validate an experiment matrix without starting training:

```bash
python scripts/run_experiments.py \
  --config configs/experiment/ablations.yaml \
  --dry-run
```

Run a matched three-seed experiment:

```bash
python scripts/run_experiments.py \
  --experiment-id baseline-1k-three-seeds \
  --seeds 7 17 29 \
  --set max_classes=1000 \
  --set image_size=64 \
  --set width=64 \
  --set epochs=20 \
  --profile-devices cuda cpu
```

Every `base_args` or variant `args` key maps to a `hccr train` option. Sweeps
default to `evaluation_policy: validation_only`; the checked-in ablation config
also states this explicitly. Keep the class subset, seeds, input size, hardware,
and benchmark settings fixed when comparing model variants.

Apply an accuracy/latency quality gate to two completed runs:

```bash
hccr compare-runs \
  --summary experiments/experiment_summary.csv \
  --baseline <baseline-run-id> \
  --candidate <candidate-run-id> \
  --min-top1-gain 0.01 \
  --max-p95-latency-ratio 1.10
```

## Inference benchmarking

Use eager mode for the fair main-table comparison after loading a checkpoint.
It is the default and runs only `model.eval()` under `torch.inference_mode()`;
it does not fold Conv--BN, reparameterize branches, compile, export, change
memory format, or capture a CUDA graph:

```bash
hccr benchmark \
  --mode eager \
  --checkpoint experiments/<run-id>/checkpoint.pt \
  --num-classes 1000 \
  --image-size 64 \
  --width 64 \
  --stage-depths 1 2 2 \
  --no-reparameterize-depthwise \
  --classification-head cosface \
  --device cuda \
  --precisions float32 float16
```

Use optimized mode only for a separately reported deployment study:

```bash
hccr benchmark \
  --mode optimized \
  --checkpoint experiments/<run-id>/checkpoint.pt \
  --num-classes 1000 \
  --image-size 64 \
  --width 64 \
  --stage-depths 1 2 2 \
  --no-reparameterize-depthwise \
  --classification-head cosface \
  --device cuda \
  --precisions float32 float16 \
  --cuda-graph
```

For CPU latency, benchmark explicit thread counts independently, for example
`--cpu-threads 1`, `2`, `4`, and `8`. FP16 requires CUDA; CUDA graph capture
requires CUDA together with `--mode optimized`. The shared CosFace/ArcFace
classifier may cache its normalized weight after `eval()` in both modes.
`python scripts/benchmark_inference.py` remains available as a compatibility
wrapper around the same command implementation.

To compare multiple completed runs on a specific CPU host using a fixed sample
from the LMDB dataset:

```bash
python scripts/benchmark_cpu_lmdb.py \
  --mode eager \
  --runs experiments/<run-a> experiments/<run-b> \
  --threads 1 \
  --warmup 100 \
  --iterations 3000 \
  --repetitions 5
```

Run this specialized script independently on each target CPU. Its JSON output
uses the same versioned timing keys and benchmark protocol as the general
benchmark command; its CSV output is a scalar summary suitable for comparison.

## Repository layout

```text
configs/              Data, model, baseline, and ablation configuration
scripts/              Dataset, sweep, and inference benchmark entry points
src/hccr/             Installable Python package
├── benchmarking/     Shared benchmark schemas and timing summaries
├── commands/         Isolated CLI parsers and command handlers
├── config/           YAML loading and lightweight schemas
├── data/             Manifest validation and dataset adapters
├── evaluation/       Metrics, diagnostics, and plots
├── inference/        Artifact-aware prediction primitive
├── models/           EfficientHCCRNet and deploy optimization
├── preprocessing/    Train/evaluation image transforms
├── training/         Training workflow, callbacks, and artifacts
└── utils/            Device, logging, and experiment utilities
tests/                Unit and end-to-end smoke tests
```

## Development

Install the development tools from `requirements.txt`, then run:

```bash
python -m ruff check src tests scripts
python -m black --check src tests scripts
python -m pytest
```

Or use the equivalent Make targets:

```bash
make lint
make format-check
make test
```

Before submitting a change:

1. Keep training and validation preprocessing isolated.
2. Add or update tests for behavior changes.
3. Compare architecture candidates on identical data and hardware.
4. Report both accuracy and batch-1 p95 latency.
5. Avoid committing datasets, credentials, checkpoints, or generated runs.

## Limitations

- The project targets isolated characters, not text-line recognition.
- Dataset download and original CASIA container decoding are out of scope; the
  manifest builder consumes folder-exported PNG images.
- The high-level `validate`, `deploy`, `evaluate`, `predict`, and `tune` CLI
  commands are not yet implemented.
- Subset results are screening signals and must not be presented as full
  7,186-class results.

## License

Source code is available under the [MIT License](LICENSE). Dataset rights and
restrictions remain with the dataset provider.
