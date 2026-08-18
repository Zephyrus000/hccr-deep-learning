# HCCR Deep Learning

Offline handwritten Chinese character recognition (HCCR) for isolated,
grayscale character images. The project builds and evaluates a compact custom
CNN, `EfficientHCCRNet`, with accuracy and inference latency treated as joint
requirements.

The current implementation supports dataset-manifest training, validation,
experiment artifacts, architecture diagnostics, class-subset benchmarks and
artifact-backed prediction primitives. It is designed for the CASIA-HWDB
manifest produced under `data/processed/casia_hwdb/`.

## Install

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
pre-commit install
```

Run the local gate before committing:

```powershell
python -m black --check src tests
python -m ruff check src tests
python -m unittest discover -s tests -v
```

## First training run

Start with a fixed, small class subset. It is much faster for testing the data
pipeline, augmentation, optimizer and architecture; its accuracy is not a
replacement for a full 7,186-class result.

```powershell
hccr train `
  --manifest data/processed/casia_hwdb/manifest.csv `
  --output-dir experiments `
  --max-classes 200 `
  --class-subset-seed 7 `
  --epochs 10 `
  --batch-size 64 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --scheduler cosine `
  --early-stopping-patience 8 `
  --device auto
```

For a pipeline sanity check, verify the model can memorize a tiny sample:

```powershell
hccr train --manifest data/processed/casia_hwdb/manifest.csv `
  --max-classes 20 --overfit-samples 64 --overfit-check --epochs 30
```

Every invocation creates `experiments/<run-id>/`. Important artifacts include:

- `config.json`, `metadata.json`: effective arguments, run ID, Git revision,
  environment and resolved device.
- `checkpoint.pt`, `checkpoint_metadata.json`, `labels.json`: model state and
  the class-index-to-Unicode mapping required by inference.
- `curves.json`, `training_diagnostics.json`, `run.log`: loss, learning rate,
  gradients, activation statistics, throughput and logs.
- `resource_profile.json`: parameter count, estimated MACs/FLOPs and batch
  1/8/32 latency mean, p50, p95 and p99.
- `per_class_metrics.csv`, `validation_errors.csv`, calibration artifacts,
  error gallery and preprocessing gallery.

`experiments/experiment_summary.csv` is the cross-run comparison table. Use a
quality gate to compare one candidate with a baseline:

```powershell
hccr compare-runs `
  --summary experiments/experiment_summary.csv `
  --baseline <baseline-run-id> `
  --candidate <candidate-run-id> `
  --min-top1-gain 0.01 `
  --max-p95-latency-ratio 1.10
```

## Training configuration

The documented baseline is [configs/experiment/baseline.yaml](configs/experiment/baseline.yaml).
The CLI is currently the executable source of training configuration; its
effective values are always captured in a run's `config.json`.

| Option | Purpose |
| --- | --- |
| `--batch-size`, `--num-workers` | DataLoader throughput and memory use. |
| `--learning-rate`, `--weight-decay` | AdamW optimization. |
| `--scheduler` | `none`, `cosine`, or validation-top-1 `plateau`. |
| `--early-stopping-patience` | Stop after this many non-improving validation epochs; `0` disables it. |
| `--width` | Base channel width of `EfficientHCCRNet`. |
| `--image-size` | Square model input size. |
| `--max-classes`, `--class-subset-seed` | Deterministic fast benchmark subset. |
| `--center-by-centroid` | Optional foreground-centroid normalization. |
| `--otsu-binarize`, `--median-filter-size` | Optional denoising/binarization ablations. |
| `--device` | `auto`, `cpu`, or `cuda`. |
| `--seed` | Reproducible model/data-order/augmentation setup. |

## Source module map

All application code lives in `src/hccr`.

| Module | Main responsibility |
| --- | --- |
| `hccr.cli` | CLI parser and dispatch for `train` and `compare-runs`; other command names remain placeholders. |
| `hccr.config` | YAML loader plus small data/model/experiment schema contracts. |
| `hccr.data` | Folder adapters, CSV-manifest invariants/audits, writer split support, image dataset and deterministic class-subset mapping. |
| `hccr.preprocessing` | Deterministic eval normalization; random affine train augmentation; optional centroid, Otsu and median filtering; visual gallery. |
| `hccr.models` | `EfficientHCCRNet`, its depthwise-separable residual blocks and model factory. |
| `hccr.training` | End-to-end workflow, train epoch, AdamW artifacts, callbacks, resource profiling and architecture diagnostics. |
| `hccr.evaluation` | Top-k metrics, validation loop, calibration/per-class/error reports and static plots. |
| `hccr.inference` | `Predictor`, which returns artifact-label-aware top-k scores from a prepared tensor. |
| `hccr.experiments` | Baseline/candidate quality gate using `experiment_summary.csv`. |
| `hccr.utils` | Device resolution, structured JSON experiment metadata and run-scoped logging. |

### `hccr.data`

- `folder_adapter.py`: enumerates folder-labeled source images.
- `manifest.py`: reads the frozen manifest and validates required columns,
  split membership, label consistency and writer overlap.
- `splitter.py`: deterministic writer-disjoint split policy when writer IDs
  exist.
- `dataset.py`: opens images from manifest-relative paths, applies a transform,
  returns `(tensor, target, metadata)` and can remap a class subset to compact
  output indices.

### Entry point, configuration and utilities

- `__init__.py`: package version.
- `__main__.py`: enables `python -m hccr` by delegating to the CLI.
- `cli.py`: command parsing and conversion of train options into
  `TrainingConfig`.
- `config/loader.py`: explicit YAML mapping reader.
- `config/schema.py`: lightweight `DataConfig`, `ModelConfig` and
  `ExperimentConfig` dataclasses for tools that consume YAML.
- `utils/device.py`: validates and resolves `auto`/CPU/CUDA device requests.
- `utils/experiment.py`: run IDs, environment/Git metadata and structured JSON
  persistence.
- `utils/logging.py`: console/file logger setup and handler cleanup for Windows.

### `hccr.preprocessing`

- `EvalPreprocessor`: grayscale conversion, optional invert/median/Otsu,
  foreground crop, aspect-ratio-preserving resize, padding and optional
  centroid centering. It must remain deterministic.
- `TrainPreprocessor`: inherits eval normalization, then applies random
  rotation, translation, scale and optional blur.
- `gallery.py`: generates raw-versus-preprocessed contact sheets for visual QA.

Keep Otsu, median filtering and centroid centering as ablations rather than
unconditional defaults: they can improve noisy scans but may erase or distort
fine HCCR strokes.

### `hccr.models`

`efficient_hccr.py` contains the target architecture:

1. `ConvNormAct` stem downsamples grayscale input.
2. Configurable depthwise-separable residual stages learn spatial features.
3. Global average pooling and a linear classifier produce class logits.

The intended architecture experiments are width, stage depth, resolution and
preprocessing variants. Compare candidates using both validation accuracy and
batch-1 p95 latency, not parameter count alone.

### `hccr.training`

- `workflow.py`: reproducible run setup; dataset/loaders; optimizer, scheduler
  and early stopping; checkpoints; all report generation.
- `trainer.py`: tqdm train loop and per-epoch loss, gradient, data-load and
  compute timing statistics.
- `callbacks.py`: validation-top-1 early stopping with `patience` and
  `min_delta`.
- `diagnostics.py`: parameter size, MAC/FLOP estimate, latency percentiles and
  top-level activation/gradient diagnostics.
- `artifacts.py`: model checkpoint and checkpoint metadata persistence.

### `hccr.evaluation`

- `metrics.py`: top-1 and top-5 classification metrics.
- `evaluator.py`: inference-mode validation aggregation.
- `diagnostics.py`: per-class accuracy, top-k error CSV, ECE/reliability plot,
  calibration bins, validation health and image error gallery.
- `reports.py`: learning curve, selected-class confusion matrix and confidence
  distribution figures.
- `analysis.py`: lightweight error/confusion analysis helpers.

### `hccr.inference` and `hccr.experiments`

`Predictor` accepts a loaded model, ordered labels and a preprocessed tensor;
checkpoint loading/serving is intentionally kept outside the predictor itself.
`compare_runs` accepts summary path and run IDs, then enforces a minimum top-1
gain and maximum p95-latency ratio.

## Repository layout

```text
configs/       Baseline, data, model and ablation presets
data/          Ignored local CASIA data and generated manifest
docs/          Ignored Obsidian project vault
scripts/       Dataset manifest/audit utilities
src/hccr/      Application packages documented above
tests/         Unit and end-to-end smoke tests
```

## Development notes

- Do not compare different architectures using different class subsets, seeds,
  input sizes or hardware.
- A low-class benchmark identifies bad candidates quickly; run finalists on the
  full class set before making accuracy claims.
- Train/eval preprocessing must not leak augmentation into validation or
  inference.
- `data/` and `docs/` are intentionally excluded from Git; commit source,
  configs, tests and this README instead.
