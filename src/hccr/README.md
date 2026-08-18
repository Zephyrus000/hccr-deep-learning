# `hccr` package

This is the application package for isolated offline handwritten Chinese
character recognition. `python -m hccr` and the installed `hccr` command both
delegate to `cli.py`.

## Package map

- [`config/`](config/README.md): YAML loading and small configuration contracts.
- [`data/`](data/README.md): manifest contracts, folder adapters and datasets.
- [`preprocessing/`](preprocessing/README.md): deterministic normalization and
  train-only augmentation.
- [`models/`](models/README.md): the EfficientHCCRNet model factory.
- [`training/`](training/README.md): complete train workflow and artifacts.
- [`evaluation/`](evaluation/README.md): metrics and validation diagnostics.
- [`inference/`](inference/README.md): top-k prediction primitive.
- [`utils/`](utils/README.md): device, logging and experiment helpers.
- `experiments.py`: compares completed runs against an accuracy/latency gate.

## Boundary

The package accepts a frozen CSV manifest and produces run-scoped artifacts.
It does not download data, mutate source data, or train a reference ResNet.

See the repository-level README for installation and CLI examples.
