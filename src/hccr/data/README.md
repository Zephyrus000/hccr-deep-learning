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
| `build_lmdb_image_store` | Pack encoded images into one random-read-optimized LMDB file. |
| `LMDBImageStore` | Lazily open a read-only LMDB reader in each DataLoader process. |
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

When no manifest exists and the source is archived, audit and generate it
directly from ZIP:

```bash
python scripts/build_dataset_manifest.py \
  --source-zip raw.zip \
  --zip-prefix raw \
  --zip-train-dir CASIA-HWDB_Train/Train \
  --zip-test-dir CASIA-HWDB_Test/Test \
  --output-dir data/processed/casia_hwdb
```

`HCCRDataset` filters rows by split. With the default `storage_backend="auto"`,
it uses `images.lmdb` next to the manifest when available and otherwise resolves
`source_file` under `data/`, `data/raw/`, or the manifest-relative fallback.
Use `storage_backend="lmdb"` in long runs to fail fast instead of silently
falling back to small-file reads.

Build the store once after freezing the manifest:

```bash
python scripts/build_lmdb_dataset.py \
  --manifest data/processed/casia_hwdb/manifest.csv
```

When the raw dataset is already archived, stream it directly into LMDB without
extracting millions of small files:

```bash
python scripts/build_lmdb_dataset.py \
  --manifest data/processed/casia_hwdb/manifest.csv \
  --source-zip raw.zip \
  --zip-prefix raw
```

`--zip-prefix` is optional and accepts any archive directory prefix; use an
empty prefix when ZIP members already match each manifest `source_file`.

The store preserves the original encoded image bytes and validates the manifest
SHA-256 before reading. Each worker opens its own lazy read-only transaction;
the environment object is never pickled into spawned workers.

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
- `metadata`: controlled by `metadata_mode`: the full manifest row (default), a
  compact augmentation string for training, or an integer row index for
  validation diagnostics.

The training workflow selects compact modes automatically, avoiding repeated
IPC serialization of string dictionaries. Tensor conversion uses torchvision
v2's optimized `to_image`/`to_dtype` path.

For comparable subset experiments, keep `max_classes` and the subset seed
identical across every candidate. Persist the returned mapping with the run so
model output indices can always be mapped back to original classes.

## Reviewer checks

The manifest is part of the experimental evidence, not disposable cache. Keep
the generated `manifest.csv`, `labels.json`, `audit_report.json`, and
`invalid_images.json` with the reproduction record. Before comparing runs,
verify that `metadata.json` contains the same manifest SHA-256 and that each
run's `labels.json` has the same class ordering.

The current folder export lacks usable writer IDs. Its deterministic 10%
validation partition is therefore image-level and artifacts correctly record
writer separation as `not_verifiable`. Do not describe this export as a
writer-disjoint protocol. LMDB changes storage only: it must preserve the
manifest digest and original encoded image bytes, so filesystem and LMDB
backends should produce the same sample tensors and targets.
