# `hccr.data`

This package owns dataset contracts. It converts folder-labeled images into a
frozen manifest, validates that manifest, and presents it to PyTorch.

| File | Responsibility |
| --- | --- |
| `folder_adapter.py` | Iterates deterministic `<unicode-label>/<image>` folders as `FolderSample`. |
| `manifest.py` | Reads CSV manifests and validates IDs, splits, labels and writer overlap. |
| `splitter.py` | Selects validation writers without writer-image leakage when writer IDs exist. |
| `dataset.py` | `HCCRDataset` loads an image, applies a transform and returns `(tensor, target, row)`. |
| `dataset.py` | `select_class_subset` samples fixed original class IDs and remaps them to compact model indices. |

## Manifest contract

Required columns are `sample_id`, `source_file`, `writer_id`, `unicode_label`,
`class_id`, and `split`. `source_file` is relative to the data root inferred
from the manifest location. Valid splits are `train`, `validation`, and `test`.

For low-class architecture benchmarks, use the same `max_classes` and
`class_subset_seed` for every candidate. The workflow writes that mapping as
`class_subset.json`.
