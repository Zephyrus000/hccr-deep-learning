# Run the stage 2 cubical PH pilot

This package implements `stage2-ph-handoff.md` without changing the baseline
model or its checkpoint format. The auxiliary input is spatial topology of a
signed scalar projection of stage 2. It is not a point cloud in channel space,
and a feature-map hole is not necessarily an ink loop.

## Install and check the bounded path

GUDHI is optional. Install the extra if it is absent:

```powershell
python -m pip install -e ".[tda]"
python -m pytest tests/test_stage2_ph.py tests/test_stage2_selection.py tests/test_stage2_streaming.py -q
python -m hccr.tda smoke
python -m hccr.tda real-smoke --config configs/experiment/stage2_ph.yaml
```

`smoke` creates a temporary synthetic two-class checkpoint and 32 images. It
runs three projection epochs, two head epochs per trial, all eight controls,
and final synthetic test/ICDAR evaluation. Its report explicitly marks the
results as smoke evidence, not research accuracy. `real-smoke` reads four TRAIN
images from the configured local manifest and checks real stage shapes, PH and
unchanged baseline logits. It does not train or open held-out images. Use a new
`--output-dir` to retain either check's artifacts.

## Run one pilot seed

The supplied configs reuse `data/processed/casia_hwdb/manifest.csv` and the
existing encoded-image `data/processed/casia_hwdb/images.lmdb`. The backend is
explicitly `lmdb`, so a missing or incompatible store fails instead of falling
back to raw files. Do not rebuild the LMDB or substitute the new GNT manifest
for this experiment. The full mapping has 7,186 classes; the pilot selects 1,000.

Edit `run_dir`, `manifest`, and `output_dir` in
`configs/experiment/stage2_ph.yaml`. The model, original labels, image size,
and padding come from the checkpoint artifacts, not constructor defaults.
The manifest and checkpoint can have different numeric class IDs: the study
joins their Unicode labels and persists the original classifier columns and
pilot IDs in `pilot_mapping.json`. Candidate classes come from TRAIN labels
only. Both original and refitted classifiers solve that same restricted
candidate-class problem.

```powershell
python -m hccr.tda project --config configs/experiment/stage2_ph.yaml --seed 7
python -m hccr.tda extract --config configs/experiment/stage2_ph.yaml --seed 7 --splits train validation
python -m hccr.tda compare --config configs/experiment/stage2_ph.yaml --seed 7
```

Alternatively, `python -m hccr.tda run --config
configs/experiment/stage2_ph.yaml --seed 7` connects these three operations.
Seeds 7, 17, and 29 are configured, but no command launches them automatically.
Reuse the same checkpoint and class-subset seed across the comparisons.

The projection is a bias-free `Conv2d(C,1,1)` initialized to `1/C`. Stage 2
continues unchanged into stage 3. A temporary flatten-to-linear classifier
trains the projection with cross-entropy while backbone parameters and BN
statistics stay frozen. Validation loss selects the projection checkpoint.
The temporary classifier is then discarded; PH is not differentiable here.
Default projection settings are AdamW, 20 epochs, batch 256, learning rate
`3e-4`, and weight decay `1e-4`. These are pilot choices, not recovered paper
hyperparameters.

After selection, each projection gets its own positive global RMS scale,
fitted on a seeded uniform TRAIN-only calibration sample. Exact sample IDs
are stored. A zero RMS uses scale one and records the degeneracy. There is no
per-image scaling, clipping, sigmoid, or added ReLU in the projection path.

PH uses `gudhi.CubicalComplex(vertices=map_values)` on a finite 2D float64 grid,
with vertex lower-star sublevel filtration and ordinary H0/H1 over F2. A
cell enters at the maximum of its vertex values. Equal values enter together;
there are no diagonal edges or periodic boundaries. Each dimension has six
positive finite-lifetime summaries: count, total, mean, population standard
deviation, maximum, and natural-log entropy. Essential count is a separate
seventh feature. Infinite deaths remain infinite in stored raw diagrams and
never enter lifetime statistics. See the [GUDHI reference](https://gudhi.inria.fr/python/latest/cubical_complex_ref.html).

## Compare the controls

All refitted heads receive the frozen final embedding plus 14 auxiliary
dimensions. Both blocks use separate TRAIN-fitted population z-scores; scales
below `1e-8` become one. Changed held-out values in constant training columns
are not erased. Fusion is `[z_cnn, alpha * z_aux]`, followed directly by the
repository's CosFace classifier, with no MLP.

| Variant | Auxiliary block |
| --- | --- |
| `original` | Original checkpoint top-1 predictions, without refitting or z-scores |
| `cnn` | Fourteen zeros; common-head retraining control |
| `gap` | Stage 2 GAP, TRAIN-only PCA to 14 dimensions |
| `flat` | Flattened learned scalar map, TRAIN-only PCA to 14 dimensions |
| `statistics` | Fourteen nonspatial statistics of the same learned map |
| `h0` | Seven learned-map H0 summaries plus seven zeros |
| `h0_h1` | Fourteen learned-map H0/H1 summaries |
| `mean_h0_h1` | Fourteen PH summaries from the fixed channel-mean projection |

Rank-deficient PCA retains the actual numerical rank and pads with zeros.
The nonspatial statistics are mean, population standard deviation, mean
absolute value, skewness, excess kurtosis, and nine linearly interpolated
quantiles at 0, .05, .10, .25, .50, .75, .90, .95, and 1. Constant maps have
zero skewness and excess kurtosis.

Each nonzero block searches alpha `[0.25,0.5,1,2]`; ties retain the first
candidate. The zero block trains once. Each head checkpoint uses validation
top-1, not projection validation loss. Defaults are eight head epochs,
batch 2048, AdamW learning rate `.001`, and weight decay `.0001`.
`precision` uses the existing FP32/BF16/FP16 policy; the angular classifier
retains its shared FP32 numerical path. Every head trial restarts with the
same seed and data-order generator. These 14-dimensional descriptors preserve
the historical fusion mechanism, not the raw-image 340/63-feature protocol.

## Evaluate only after freezing the comparison

```powershell
python -m hccr.tda evaluate --config configs/experiment/stage2_ph.yaml --seed 7 --splits test
```

`selection.json` must contain every configured variant and match the current
training signature before any held-out dataset or cache can be constructed.
Evaluation applies all frozen variant choices, without selecting a new winner
or changing PCA, z-scores, calibration, alpha, or checkpoints.
The current image-export manifest has train, validation, and test only. It has
no ICDAR split. With a later audited four-split dataset and matching model
provenance, request `--splits test icdar` and keep those results separate.
Do not use either holdout to choose the method.

The current image export has no writer IDs; validation is its frozen image-level
split. The supplied historical CNN checkpoint's original pretraining writer
provenance is also unverified. The reports retain both provenance records.
Do not claim writer-disjoint or unseen-writer generalization for this experiment.

## Inspect and reuse artifacts

Each seed directory contains `projection.pt`, `projection.json`,
`pilot_mapping.json`, per-split `features/`, selected `heads/`, `selection.json`,
and, only after explicit evaluation, `heldout.json`.
Feature caches store embeddings, controls, restricted original top-1 predictions, targets,
and one inspection map with raw H0/H1 diagrams for each projection mode.

Cache identity includes checkpoint, metadata and label digests, manifest digest,
ordered sample IDs, class mapping, preprocessing, stage, both projection modes,
projection weights, both calibration states, filtration, field, feature schema,
precision/device policy, dependency versions, and the relevant source files.
Array hashes detect corrupted caches. Staging directories publish atomically;
an interrupted write is not a valid cache. Incompatibility errors name changed
identity fields. Use a new output directory after changing the method; existing
projection/selection artifacts are not silently overwritten.

Caches use feature schema 2. Extraction writes each batch into staged NumPy
memory maps instead of retaining batches for a final concatenation. Only top-1
predictions of the original classifier are stored, after restricting its columns
to the pilot candidates. This preserves the reported top-1 and class-error counts;
it does not preserve probabilities for later calibration or top-k analysis.

Z-scores merge float64 population moments in chunks of 4,096 TRAIN rows.
PCA uses the TRAIN covariance matrix and `numpy.linalg.eigh`, not a dense SVD
of all samples. It retains at most 14 eigenvectors above
`feature_width * float64_epsilon * largest_eigenvalue`, with deterministic signs.
Numerically degenerate PCA directions can differ from the previous dense-SVD
solver. The solver and fitted transforms are recorded in `selection.json`.
Both fitters read every TRAIN row. Neither fits on validation or held-out rows.

Fusion creates features only for the requested head batch. Head evaluation
reduces logits to class counters batch by batch. It does not concatenate the
full `samples × classes` output. LMDB sample plans store packed UTF-8 IDs and
integer labels instead of copying complete manifest-row dictionaries to workers.
The image loader still uses the configured spawn workers; cached-head training
uses zero workers. Sample plans, shuffled row indices, and provenance metadata
still require memory proportional to the number of samples. Feature storage
requires disk space proportional to the selected dataset.

Caches open as NumPy read-only memory maps. Compare and evaluate close their
maps on success or failure. For direct `extract()` callers, use the returned
cache as a context manager or call `close()` before removing its files,
especially on Windows. Do not use array views after closing their cache.
Use a new output directory for this schema; old caches and fitted projections
fail the source-identity check rather than being silently reused.

Projection fitting and alpha
search costs are separate from extraction component wall times. These times
are not deployment latency or a cached-head inference benchmark. Complete
inference would include backbone, projection, CPU transfer, PH, transforms,
and classification.

## Run the independent one-seed loss ablation

This study trains three backbones from scratch, not three cached PH heads.
It holds architecture, 1,000 selected classes, seed 7, preprocessing,
augmentation, batch 64, 20-epoch budget, and cosine scheduler fixed.

```powershell
python scripts/run_experiments.py --config configs/experiment/loss_ablation_seed7.yaml --dry-run
python scripts/run_experiments.py --config configs/experiment/loss_ablation_seed7.yaml
```

The variants are CosFace control, ordinary linear logits with cross-entropy
(`classification_head: softmax`), and ArcFace. All use cross-entropy on their
respective training logits. CosFace margin .1 is a cosine offset; ArcFace
margin .1 is in radians, not an assertion of equal margin strength. The shared
margin warmup applies to the two angular heads only. This is one declared
pilot setting, not a tuned comparison of each loss's best possible result.

The config reuses the existing `casia_hwdb/images.lmdb`, CUDA, four spawn
workers and validation-only evaluation. No GNT preparation is required.
For a larger GPU, override batch and
reference batch together for all three jobs. Do not combine the PH result and
this independent loss change into one causal claim without an interaction
experiment. Full extraction/training sweeps are not started by the smoke commands.
