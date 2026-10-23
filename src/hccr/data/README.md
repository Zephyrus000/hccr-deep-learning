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
| `source_file` | Image or GNT path relative to the discovered data root. |
| `writer_id` | Writer identity when known; an empty value is allowed. |
| `unicode_label` | Character label associated with the class ID. |
| `class_id` | Original integer class identifier. |
| `split` | `train`, `calibration`, `validation`, `test`, or `icdar`. |

For GNT sources, `record_offset` is also required. The pair `source_file` and
`record_offset` identifies a record; many images share one GNT file.

## Proposal protocols

Use the official proposal presets for the difficulty-aware classifier study.
The older presets below keep their original split policy for reproducibility.
The primary protocol uses HWDB1.0 and HWDB1.1 official training/test sources,
filters to the 3,755 GB2312 level-1 characters, and retains ICDAR 2013 as a
separate endpoint. All sources are original GNT records. Kaggle JPEG exports
and their checkpoints are excluded from this study.

```powershell
python scripts/build_casia_manifest.py --protocol hwdb1.0-1.1-official
```

The expected source directories are `HWDB1.0_Train`, `HWDB1.1_Train`,
`HWDB1.0_Test`, `HWDB1.1_Test`, and `ICDAR2013_Test`, under `data/raw`.
The output is `data/processed/hwdb1_0-1_1_official_five_splits`.
Use `--without-icdar` to prepare only the HWDB endpoints; this writes
`data/processed/hwdb1_0-1_1_official_four_splits`.

Official training writers are assigned approximately 80% to `train`, 10% to
`calibration`, and 10% to `validation`, using seed `20261022`. Splitting is
stratified by source-version membership. A writer occurring in several files
or versions stays in one split. Each class must have fit and calibration
support; a build fails rather than adjusting splits using model predictions.
The report freezes the exact writer lists, class support, and per-version
sample counts. Official HWDB test records never enter development splits.

The optional larger-vocabulary protocol needs an audited JSON list containing
the canonical 7,185 Unicode labels, or a `class_mapping_full.json` with those
labels. A class count alone cannot establish which 7,185 labels are canonical.
The builder checks vocabulary membership and complete fit/calibration support.
It never removes an unexplained 7,186th class automatically.

```powershell
python scripts/build_casia_manifest.py --protocol hwdb1.0-1.2-official --vocabulary docs/canonical_hwdb_7185.json
```

This reads separate `HWDB1.0`, `HWDB1.1`, and `HWDB1.2` train/test directories
and writes `data/processed/hwdb1_0-1_2_official_four_splits`. ICDAR is optional
via `--icdar-dir` and remains a 3,755-class subset endpoint. Its addition changes
the output suffix to `_five_splits`.

For merged source directories, repeat `--train-dir` and `--test-dir` and pass
`--source-version-map versions.json`. The JSON object maps source files or
directory prefixes relative to `--data-root` to `HWDB1.0`, `HWDB1.1`, or
`HWDB1.2`. A more specific file entry overrides a directory entry. For example:

```json
{
  "HWDB1.0-1.1_Train/1001-f.gnt": "HWDB1.1",
  "HWDB1.0-1.1_Test/1241-f.gnt": "HWDB1.1"
}
```

The map must cover every file whose version is ambiguous. Fill it from the
original archive inventory or source metadata. The builder does not infer a
version from numerical writer ranges. Every requested version must occur in
both official train and official test sources.

For the merged directories in this checkout, use their paths explicitly after
preparing the source-version map:

```powershell
python scripts/build_casia_manifest.py --protocol hwdb1.0-1.1-official --train-dir data/raw/HWDB1.0-1.1_Train --test-dir data/raw/HWDB1.0-1.1_Test --source-version-map versions.json
```

Schema 4 adds `source_version`, `official_partition`, `source_writer_id`, and
`dataset_protocol` to every row. `writer_id` is the global identity used for
splitting; `source_writer_id` preserves the filename identity. A source metadata
crosswalk may be supplied with `--writer-identity-map identities.json`:

```json
{
  "icdar2013:C001": "casia:verified-source-writer-id"
}
```

Only populate that map with identities supported by source evidence. Without
a complete competition crosswalk, the native `icdar2013:C001` namespace stays
unchanged and the report records
`icdar_identity_reconciliation=unverified_source_namespace_ids`. It verifies
HWDB writer separation and native pixel separation but does not claim that
different CASIA/ICDAR namespaces prove different people. The overall
`writer_disjoint_verified` field is `null` in that case; the HWDB-only field is
`hwdb_writer_disjoint_verified=true`. Competition writer-cluster uncertainty
must use only identities that can be verified. A supplied crosswalk that
reveals overlap causes the build to fail.

Before official-protocol training, `validate_training_manifest` checks the
manifest and class-map SHA-256 hashes, schema, split and source provenance,
class coverage, exact class support and version counts, and pixel audit coverage.
Validation, HWDB test, and ICDAR may contain fewer classes than the vocabulary;
every vocabulary class requires support in fit and calibration.
`audit_report.json`
contains `manifest_digest` and `label_mapping_digest`, source SHA-256 values,
the frozen writer lists, and support counts. Preserve the report alongside the
manifest and class map. A legacy or modified manifest cannot claim a new
official protocol without this audit.

The rebuild helper defaults to the primary protocol; inspect its plan first:

```powershell
python scripts/run_hwdb_manifest_rebuild.py --plan-only
python scripts/run_hwdb_manifest_rebuild.py --include-secondary --secondary-vocabulary docs/canonical_hwdb_7185.json --plan-only
```

Remove `--plan-only` only when ready to scan the source data. Preparation does
not train models. The helper refuses existing output directories and publishes
job status under `artifacts/hwdb-mapping-rebuild-20261022` by default.

## Prepare official CASIA GNT datasets

Use a separate manifest and mapping for each protocol. Build the HWDB1.1
GB2312 level-1 dataset from the downloaded writer files:

```bash
python scripts/build_casia_manifest.py --protocol hwdb1.1
```

This reads `data/raw/HWDB1.1_Train` and `data/raw/HWDB1.1_Test` and publishes
`data/processed/hwdb1_1`. It excludes symbols and checks for exactly 3,755
training classes. Class IDs follow Unicode-sorted training labels; the same
IDs apply to validation and test.

Build the combined HWDB1.0 through HWDB1.2 dataset separately:

```bash
python scripts/build_casia_manifest.py --protocol hwdb1.0-1.2
```

This reads `data/raw/HWDB1.0-1.2_Train` and `data/raw/HWDB1.0-1.2_Test` and
publishes `data/processed/hwdb1_0-1_2`. The `all-hanzi` filter retains Unicode
Han characters and excludes symbols. Use `audit_report.json.class_count` for
the classifier size. Do not reuse the old folder export's 7,186-class setting
without checking the new mapping.

Both commands preserve the supplied test partition. They hold out 10% of
training writers for validation, using seed `20260817` by default. Writer IDs
come from CASIA filenames such as `1001-f.gnt`. A build fails if writers overlap
between train and test, if train loses a class, if test contains
an unknown class, or if a GNT record is malformed. Existing output directories
are never overwritten.

The report lists classes absent from validation or test. Writer-level holdout
does not guarantee that every rare class appears in validation.

Each output directory contains:

- `manifest.csv`, including writer IDs, raw tag codes, and record offsets.
- `class_mapping_full.json`, with the existing list-based mapping schema.
- `labels.json`, mapping model output indices to Unicode labels.
- `audit_report.json`, with protocol, filtering, split counts, source inventory,
  validation writers, class count, and manifest SHA-256.

The scanner checks GNT headers and bitmap bounds without writing image files.
The CLI also hashes native bitmaps and rejects exact images shared across
splits. Same-split duplicates are counted and retained. Well-formed empty
records with dimensions `0x0` are counted in `empty_records` and excluded;
truncated records and inconsistent sizes still fail the build.
To use different directory names, pass `--data-root`, `--train-dir`,
`--test-dir`, and `--output-dir`. The source directories must be under
`--data-root`. For another partition, use `--protocol custom` with both source
directories explicitly set. `--label-set` selects `gb2312-level1` or
`all-hanzi`; `--expected-class-count` adds a class-count check.

HWDB1.1's official test split is distinct from the ICDAR competition test set.
Use the dedicated ICDAR protocols below for that benchmark. Consult the
[CASIA download page](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)
for the distinction between those downloads and the
[GNT format specification](https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)
for the record layout.

## Prepare the ICDAR 2013 benchmark

These protocols preserve four separate splits. Validation holds out writers
from official CASIA training data only; CASIA test never enters training.
ICDAR2013 remains an additional external holdout, not a replacement for CASIA test.
For the `icdar2013-*` protocols, all four splits share one mapping of 3,755
GB2312 level-1 classes.

| Protocol | Train writers | Validation writers | CASIA test writers | ICDAR writers |
| --- | ---: | ---: | ---: | ---: |
| HWDB1.1 + ICDAR | 216 | 24 | 60 | 60 |
| HWDB1.0+1.1 + ICDAR | 519 | 57 | 144 | 60 |
| HWDB1.0+1.2 + ICDAR | 735 | 81 | 204 | 60 |

These counts use the default 10% writer holdout and fixed manifest seed. They
are not the 720-development-writer protocol used in some publications; report
the actual training budget when comparing results.

```bash
python scripts/build_casia_manifest.py --protocol icdar2013-hwdb1.0-1.1
```

This reads the combined official training data from `HWDB1.0-1.1_Train`,
official CASIA test data from `HWDB1.0-1.1_Test`, and external holdout data
from `ICDAR2013_Test`, all under `data/raw`. It writes
`data/processed/icdar2013-hwdb1_0-1_1_four_splits`. HWDB1.2 is not included.

For the expanded official HWDB1.0–1.2 data with the same fixed 3,755 classes:

```bash
python scripts/build_casia_manifest.py --protocol icdar2013-hwdb1.0-1.2
```

This reads `HWDB1.0-1.2_Train` and `HWDB1.0-1.2_Test` and writes
`data/processed/icdar2013-hwdb1_0-1_2_four_splits`. The official CASIA test
and ICDAR benchmark stay separate. The larger training set changes the
experimental protocol, so report it when comparing results.

For an additional `all-hanzi` mapping from HWDB1.0–1.2 with the ICDAR holdout:

```bash
python scripts/build_casia_manifest.py --protocol hwdb1.0-1.2 \
  --icdar-dir data/raw/ICDAR2013_Test \
  --output-dir data/processed/hwdb1_0-1_2_four_splits
```

This keeps all Hanzi classes found in the official training files and reports
the resulting class count. Since ICDAR contains only its own label subset,
this is an expanded-class experiment rather than the standard 3,755-class
ICDAR comparison.

For training using only the official HWDB1.1 training partition:

```bash
python scripts/build_casia_manifest.py --protocol icdar2013-hwdb1.1
```

This writes `data/processed/icdar2013-hwdb1_1_four_splits`. Use `--icdar-dir`
to specify a different competition directory. For custom source layouts,
repeat `--train-dir` and `--test-dir` for multiple official source directories.
Do not pass official test directories as training directories. Previous
manifests that merged CASIA test into development are not overwritten; their
protocol is different and they must not be used for this four-split experiment.

ICDAR benchmark protocols require exactly 60 writers named `C001` through
`C060`, 224,419 usable test records, and 3,755 training classes. The expanded
`all-hanzi` variant retains the ICDAR writer and sample checks but derives its
class count from training data. The parser preserves the
literal competition prefix: `C001-f-f.gnt` becomes `icdar2013:C001`, while
`001-f.gnt` becomes `casia:1`. These identifiers document the source naming
scheme. Different prefixes alone do not prove that the people are different.
Writer separation across datasets relies on the published dataset provenance;
the software checks identifier consistency and exact image leakage, not human
identity or handwriting similarity.

Every retained image receives `pixel_sha256`, computed from its native
dimensions and grayscale bytes without the label. A disk-backed index checks
all six split pairs. A renamed file or changed label cannot hide an exact
bitmap copy. The builder fails on a cross-split match and leaves the source
data untouched. Review the source if this happens; do not remove competition
test images to make the check pass. This audit does not detect resized or
otherwise modified duplicates.

`audit_report.json` records the image audit result and coverage, source-file
SHA-256 values, manifest SHA-256, and class-mapping SHA-256. Training preflight
rejects known writer overlap in any split pair, GNT writer IDs inconsistent
with their filenames, and changes to an audited GNT manifest or mapping.
Competition manifests require a successful image audit covering every sample;
competition writers must appear only in `icdar`. Writer overlap is checked
across all six split pairs, including `test/icdar`.

Keep the split seed fixed across models and training seeds. Select architecture,
augmentation, hyperparameters, checkpoint, and any calibration using development
data only. Use `evaluation_policy: validation_only` during model selection.
After freezing those choices, use `evaluation_policy: final_test`. Training
evaluates the same selected checkpoint once on CASIA `test`, then once on
`icdar`. Neither holdout is evaluated during epochs or used for checkpoint or
BN-recalibration selection. `validation_only` evaluates neither holdout.

`metrics.json` and checkpoint metadata keep separate `test` and `icdar`
objects. CASIA diagnostics remain in the run directory; ICDAR diagnostics go
under `icdar/`, including `icdar/icdar_health.json`, per-class metrics,
confusion pairs, and calibration plots, so they cannot overwrite CASIA reports.
In `experiment_summary.csv`, `top1` remains CASIA test
accuracy and `icdar_top1` is external ICDAR accuracy. The training API's return
value remains CASIA test metrics; it does not replace or average those two results.

Report the actual held-out development budget. Do not use CASIA test or
ICDAR accuracy to choose the epoch or tune the model. The sweep summary's
selection metrics are validation metrics; obtain final accuracy from test
metrics. Compare models using the same manifest, mapping, data budget, and
evaluation policy, and report mean and sample standard deviation across all
predeclared training seeds rather than only the best seed.

Pack the frozen manifest into LMDB using the existing builder:

```bash
python scripts/build_lmdb_dataset.py --manifest data/processed/icdar2013-hwdb1_1_four_splits/manifest.csv
```

The builder reads GNT records directly and stores each bitmap as lossless PNG
inside one LMDB containing all four splits. It preserves the manifest's split
assignments, class mapping, native size, grayscale values, and polarity. Resize,
centering, inversion, and augmentation still run through the existing
preprocessing pipeline. Extract GNT archives before this step; the ZIP image
builder is intended for encoded image exports.

Train with the new manifest and its class count:

```bash
python -m hccr train --manifest data/processed/icdar2013-hwdb1_1_four_splits/manifest.csv \
  --num-classes 3755 --dataset-backend lmdb --evaluation-policy final_test
```

Add the model and training arguments for your experiment. For YAML sweeps,
set top-level `manifest` to the new manifest and `base_args.num_classes` to its
class count. If enabled, set `full_class_num_classes` to that count too.
Keep the same manifest digest and label ordering across compared models.
An old checkpoint's output indices refer to its old mapping and cannot be
reinterpreted using this mapping.

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

For image-file sources, the store preserves the original encoded bytes. For
GNT sources, it stores lossless PNG records. It validates the manifest
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

The legacy folder export lacks usable writer IDs. Its deterministic 10%
validation partition is therefore image-level and artifacts correctly record
writer separation as `not_verifiable`. Do not describe this export as a
writer-disjoint protocol. Official GNT manifests instead use filename writer
IDs and a writer-disjoint validation split. LMDB changes storage only: it must
preserve the manifest digest and pixel values, so filesystem and LMDB
backends should produce the same sample tensors and targets.
