# `hccr`

`hccr` is the installable application package for offline isolated handwritten
Chinese character recognition. It connects a frozen CSV dataset manifest to
preprocessing, model training, evaluation artifacts, and top-k inference.

Both supported entry points delegate to the thin router in [`cli.py`](cli.py):

```bash
hccr --help
python -m hccr --help
```

The router delegates parser setup and execution to [`commands`](commands/).
`train`, `benchmark`, and `compare-runs` are executable. `validate`, `deploy`,
and the legacy reserved commands are placeholders, not complete workflows.

## Package map

| Package | Responsibility |
| --- | --- |
| [`benchmarking`](benchmarking/README.md) | Shared benchmark schemas, protocols, and timing summaries. |
| [`commands`](commands/) | Isolated CLI parsers and handlers for each workflow. |
| [`config`](config/README.md) | YAML loading and small cross-layer configuration records. |
| [`data`](data/README.md) | Manifest validation, folder adapters, class subsets, and PyTorch datasets. |
| [`preprocessing`](preprocessing/README.md) | Deterministic evaluation normalization and train-only augmentation. |
| [`models`](models/README.md) | `EfficientHCCRNet`, angular classifiers, and inference optimization. |
| [`training`](training/README.md) | End-to-end training orchestration, callbacks, diagnostics, and checkpoints. |
| [`evaluation`](evaluation/README.md) | Classification metrics, validation diagnostics, and plots. |
| [`inference`](inference/README.md) | A small artifact-label-aware top-k predictor. |
| [`utils`](utils/README.md) | Device selection, run logging, and structured experiment metadata. |

Two orchestration modules live at package root:

- `experiments.py` applies an accuracy/latency quality gate to completed runs.
- `experiment_runner.py` expands YAML/CLI variants and seeds into sequential,
  resumable training jobs.

## Data flow

```text
CSV manifest
    │
    ▼
HCCRDataset ──► Eval/TrainPreprocessor ──► EfficientHCCRNet
                                                    │
                     ┌──────────────────────────────┤
                     ▼                              ▼
              training artifacts             evaluation reports
                     │
                     ▼
             optimized inference
```

## Boundaries

The package does not download or license CASIA-HWDB, decode the original CASIA
container format, or mutate source images. Raw data, generated manifests,
checkpoints, and experiment output remain outside `src/`.

Subpackages expose a deliberately small API through their `__init__.py` files.
Modules prefixed with `_` and unexported helpers are implementation details and
may change without a compatibility guarantee.

See the [repository README](../../README.md) for installation and complete CLI
examples.
