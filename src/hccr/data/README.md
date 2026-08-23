# `hccr.data`

`hccr.data` owns the boundary between source images and model-ready samples. It
defines the CSV manifest contract, audits dataset invariants, supports
deterministic class/writer selection, and exposes the PyTorch dataset used by
training.

## Public API

| Symbol | Purpose |
| --- | --- |
| `read_manifest` | Read a UTF-8 CSV manifest and verify its required columns. |
| `audit_manifest` / `ManifestAudit` | Validate unique samples/files, splits, class-label consistency, and writer overlap. |
| `HCCRDataset` | Load one manifest split and return normalized grayscale tensors. |
| `select_class_subset` | Select original class IDs deterministically and remap them to a compact range. |
| `iter_folder_samples` / `FolderSample` | Enumerate sorted `<unicode-label>/<image>` exports with stable sample IDs. |
| `WriterDisjointSplitter` | Select complete writer groups for validation without image-level leakage. |

## Manifest contract

The required columns are:

| Column | Meaning |
| --- | --- |
| `sample_id` | Non-empty identifier unique across the manifest. |
| `source_file` | Image path relative to the discovered data root. |
| `writer_id` | Writer identity when known; an empty value is allowed. |
| `unicode_label` | Character label associated with the class ID. |
| `class_id` | Original integer class identifier. |
| `split` | `train`, `validation`, or `test`. |

`HCCRDataset` filters rows by split and resolves `source_file` under `data/`,
`data/raw/`, or the corresponding manifest-relative fallback. A missing first
sample produces a clear `FileNotFoundError` instead of silently changing roots.

## Dataset output

```python
from pathlib import Path

from hccr.data import HCCRDataset
from hccr.preprocessing import EvalPreprocessor

dataset = HCCRDataset(
    Path("data/processed/casia_hwdb/manifest.csv"),
    split="validation",
    transform=EvalPreprocessor(image_size=64),
)
image, target, metadata = dataset[0]
```

The returned tuple is:

- `image`: float tensor `[1, H, W]` scaled to `[0, 1]`.
- `target`: original class ID, or its compact subset index.
- `metadata`: a copy of the manifest row plus `applied_augmentations`.

For comparable subset experiments, keep `max_classes` and the subset seed
identical across every candidate. Persist the returned mapping with the run so
model output indices can always be mapped back to original classes.
