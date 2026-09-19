# `hccr.config`

This package provides configuration primitives that are independent of the
command-line parser. It intentionally performs loading and structural typing,
not full training-option validation.

## Public API

| Symbol | Defined in | Contract |
| --- | --- | --- |
| `load_yaml(path)` | `loader.py` | Loads a UTF-8 YAML document and requires a mapping at its root. |
| `DataConfig` | `schema.py` | Manifest path, image size, and data seed. |
| `ModelConfig` | `schema.py` | Generic model name, class count, and input-channel count. |
| `ExperimentConfig` | `schema.py` | Experiment name, output root, and preferred device. |

All three schemas are frozen dataclasses. They are useful for small tools and
configuration boundaries; the executable training contract is
`hccr.training.TrainingConfig`, which contains the complete set of resolved CLI
options.

## Example

```python
from pathlib import Path

from hccr.config import DataConfig, load_yaml

raw = load_yaml(Path("configs/data/default.yaml"))
data = DataConfig(**raw)
print(data.manifest_path, data.image_size)
```

`load_yaml` raises:

- `RuntimeError` when PyYAML is unavailable.
- `ValueError` when the YAML root is not a mapping.
- The normal filesystem or YAML parser exception for unreadable/invalid input.

Experiment sweep YAML is interpreted by `hccr.experiment_runner`; see
[`configs/experiment/thesis_model_comparison.yaml`](../../../configs/experiment/thesis_model_comparison.yaml)
for the supported `base_args`, `variants`, hardware constraints, and seed
structure.

[`configs/model/baseline.yaml`](../../../configs/model/baseline.yaml) is only an
architecture mirror of the paper's single-branch EfficientHCCR control. It is
not an independent training preset. The experiment YAML and generated
`plan.json` remain the executable protocol.

## Reproducibility rule

For paper experiments, the YAML plus its generated `plan.json` is the immutable
configuration record. Reviewers should compare the expanded command matrix,
not rely on prose or current defaults. Unknown/misspelled training keys fail
during command validation, while hardware requirements are checked before a
non-dry run. Preserve the YAML, plan, Git commit, and every run's resolved
`config.json` together.

Changing a schema or loader is paper-neutral only when the same YAML expands to
the same jobs and resolved training values. If a new default changes an omitted
field, pin that field in the paper YAML before claiming protocol equivalence.
