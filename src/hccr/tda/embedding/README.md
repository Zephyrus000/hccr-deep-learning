# Persistent homology on 320D embedding neighborhoods

This package tests whether local embedding-space topology adds recognition
information beyond the same frozen CNN features. It is separate from the
scalar-map/cubical-PH study in the parent package. It does not train the
backbone, alter existing checkpoints, or overwrite the old study's artifacts.

## Geometry and controls

Each image contributes **one point in R^320**. Its local point cloud contains
the query and `neighbors` nearest reference points. A 320-element vector is
never reshaped into a 2D image or 320 one-dimensional points.

References come only from TRAIN, with a seeded, up-to-`bank_per_class` quota
for each selected class. Sampling is stable under source row reordering.
Search is exhaustive within this sampled bank, **not all TRAIN samples**.
Embeddings are L2-normalized in float64 for Euclidean search and PH only.
Search uses bounded query/reference tiles and breaks distance ties by sample
ID. A TRAIN query cannot retrieve itself; exclusion uses sample ID, not zero
distance. Other images with identical embeddings remain valid references.

For each cloud, GUDHI builds the complete Euclidean Vietoris–Rips 2-skeleton
and computes H0/H1 over F2. There is no finite edge cutoff. Each dimension
contributes finite positive lifetime count, total, mean, population standard
deviation, maximum, natural-log entropy, and essential-class count: 14
features total. The extra dimension is necessary to obtain H1 deaths; see
the [GUDHI Rips reference](https://gudhi.inria.fr/python/latest/rips_complex_ref.html).
No features are learned through the Rips computation.

The four controls share source rows, candidate classes, frozen raw CNN
embeddings, CosFace implementation, head initialization seed, optimizer,
training budgets, and validation selection rule:

| Control | Input / prediction |
| --- | --- |
| `cnn` | TRAIN-standardized raw CNN vector plus 14 zero coordinates |
| `cnn_knn` | The exact selected CNN head mixed with uniform TRAIN-neighbor votes |
| `distance` | CNN vector plus 14 statistics of query-to-neighbor distances |
| `ph` | CNN vector plus 14 H0/H1 lifetime summaries |

Distance statistics are mean, standard deviation, mean absolute value,
skewness, excess kurtosis, and quantiles 0/5/10/25/50/75/90/95/100 percent.
They control for neighborhood scale without using persistence.

All heads have 334 input coordinates. CNN and auxiliary blocks are
standardized using TRAIN only. Distance and PH share the same `alpha_grid`
for auxiliary weighting and first-best validation-top1 selection, including
epoch selection. CNN+kNN reuses the selected CNN checkpoint; only its
probability-mixture coefficient is selected. Its grid must include zero;
there is no distance-weighting or temperature search advantage. The report
records unequal trial counts explicitly: one CNN head and one head per
alpha for each auxiliary branch. A positive PH claim should beat the CNN,
CNN+kNN **and** distance controls, not merely the historical checkpoint.

## Input contract

Install the existing optional dependencies if needed:

```powershell
python -m pip install -e ".[tda]"
```

`source_root` must contain the completed frozen-embedding cache directories
`train/` and `validation/`. Each must have signed `metadata.json`,
`cnn.npy` (float32, N × 320), and integer `targets.npy`. Metadata must include
checkpoint identity, class mapping, manifest identity, preprocessing,
extraction precision, split and ordered unique sample IDs. Checkpoint,
mapping and shared extraction provenance must agree across splits.
Both consumed arrays are checksum-verified. Unused scalar-PH arrays are not
loaded or hashed. Zero/nonfinite selected embeddings and overlapping sample
IDs are rejected.

The supplied YAML reuses the existing LMDB-derived embedding caches. This
module does **not** directly read images, LMDB, GNT, or the checkpoint to
extract new embeddings. Generate those caches with the parent pipeline
before using a different source. A fresh-image extraction adapter is outside
this implementation's scope.

## Run development, then evaluate explicitly

Run from the repository root. The default is a 200-class, seed-7 development
pilot with 32 references per class and 32 neighbors. No experiment starts
automatically when importing the package.

```powershell
python -m hccr.tda.embedding run --config configs/experiment/embedding_ph_seed7.yaml
```

`run` extracts TRAIN/validation features and selects all four controls. It
does not open, stat, or evaluate test/ICDAR caches. To separate work:

```powershell
python -m hccr.tda.embedding prepare --config configs/experiment/embedding_ph_seed7.yaml
python -m hccr.tda.embedding extract --config configs/experiment/embedding_ph_seed7.yaml
python -m hccr.tda.embedding compare --config configs/experiment/embedding_ph_seed7.yaml
```

Only after selection, evaluate existing holdout embedding caches explicitly:

```powershell
python -m hccr.tda.embedding evaluate --config configs/experiment/embedding_ph_seed7.yaml --splits test
# Add --splits test icdar only if both compatible caches exist.
```

The holdout gate verifies all four selections, source/config/implementation
identity, selected head checksums, reference bank, and development caches
before opening a holdout. Selection signatures detect accidental edits,
not malicious modification by someone able to rewrite all artifacts.
No scaler, bank, alpha, checkpoint, or kNN mixture is fitted on holdouts.
Changed inputs/config/code require a new `output_dir`; completed comparisons
are not silently overwritten. Published feature caches are reused after
integrity checks. An interrupted unpublished split is recomputed.

For an inexpensive implementation check, without any real dataset:

```powershell
python -m hccr.tda.embedding smoke --output-dir artifacts/embedding-ph-smoke
```

The smoke output directory must be empty. Its accuracy is synthetic and
cannot support a research claim.

## Outputs and limits

Artifacts are under `output_dir/seed-7/`:

- `study.json`: source identities, class mapping, protocol/config and code hashes.
- `bank/`: normalized TRAIN vectors, targets and IDs with integrity metadata.
- `features/{train,validation,test,icdar}/`: raw CNN, PH/distance features,
  neighbor indices, selected source indices and IDs. Holdout directories are
  created only by explicit evaluation. Inspection arrays store one example
  cloud and its H0/H1 diagrams per split.
- `heads/`: selected refitted head checkpoints.
- `selection.json`: validation metrics, epoch/alpha/mix trials, actual bank
  support, checkpoint checksums and selection signature.
- `heldout.json`: final per-control top1 and class support/error counts.
- `extraction_*.json`: cached-feature extraction time. This excludes backbone
  inference and is **not deployment latency**.

Numeric features remain memory-mapped; fusion and classifier inference read
batches rather than materializing N × 334 or N × classes matrices. Sample
ID lists, selected indices/targets and bank IDs still use memory proportional
to sample count. Rips runs per query on CPU; its simplex count grows cubically
with cloud size. The config caps neighbors at 64. Exhaustive float64 search
and PH add computation and reference-bank storage at inference; this is an
accuracy experiment, not a latency optimization. Start with the pilot before
considering full-class extraction.

Writer IDs are not present in these inputs: sample-ID separation is checked,
but `writer_disjoint_verified` remains unknown. Source holdouts previously
viewed in the cubical study are exploratory, not a newly untouched benchmark.
The historical original-checkpoint accuracy discrepancy remains unresolved;
this module does not treat that historical score as its accuracy reference.
Use matched refitted controls and additional seeds/untouched evaluation to
substantiate any later paper claim.

## Accelerated full-class execution

`search_backend` selects `reference`, `cpu_exact`, or `cuda_exact`. The matrix
backends still search every bank vector in float64. They retain candidates
inside a conservative rounding-error interval and rerank them with the original
fixed-order CPU expression. Distance ties keep sample-ID order. A large candidate
set falls back to the reference implementation without truncation. This is not
FP32 geometry or approximate nearest-neighbor search.

The error interval assumes IEEE FP64 arithmetic. With unit roundoff `u=2^-53`,
dimension `D=320`, and `gamma(n)=n*u/(1-n*u)`, the implementation uses
`E=32*gamma(2D+8)*max(1, max_reference_squared_norm)`. Keeping the closed GPU
distance band through `kth_distance + 2E` includes every legacy top-k candidate;
the CPU rerank decides membership. Extra top-k slots only avoid an interval scan
when their boundary certifies the same condition. They do not approximate it.

The full query-distance matrix is limited to 1 GiB by reducing query tile size,
not reference coverage. The reference bank itself also occupies device memory.
`ph_workers` enables persistent spawn-based PH processes, and `ph_chunk_size`
sets the target chunk size. Each worker opens the bank read-only. The parent
overlaps the next search batch with current PH tasks, restores input order,
and remains the only cache writer. Pending work is bounded by query-batch size.
Extraction resources close before head training.

The optimized runner sets native worker math threads to one before spawn.
The supplied config uses six PH workers. CPU matrix search uses six Torch
threads, so overlapping search and PH stay within twelve compute threads.
After extraction, head training restores twelve Torch threads. This limits
thread counts; it does not pin Windows processor affinity to particular cores.

To benchmark the old full bank and then run the unchanged full-class study:

```powershell
python -m scripts.run_optimized_embedding_ph --config configs/experiment/embedding_ph_full_optimized_seed7.yaml --bank artifacts/embedding-ph-full-seed7-20261020/seed-7/bank
```

The runner compares reference, CPU-BLAS and CUDA extraction on the same 1,024
seeded TRAIN queries, covering two production query batches and their overlap.
Five alternating trials include search, Rips, ordered
cache writes and hash publication. Warmup and a separate reference-search
profile are excluded from the reported timings. All published array hashes
must match. The runner selects the faster matrix backend and refuses the long
restart if neither has a measured gain above 5 percent. This subset benchmark
does not measure full training time or deployment latency.

`benchmark/benchmark.json` records samples, ranges, stage times, numerical
diagnostics and the CPU profile summary. `resolved_config.json` records the
selected backend without editing the requested YAML. `dispatch_status.json`
tracks the benchmark gate; `seed-7/status.json` tracks extraction, matched-head
training and final TEST after the gate passes. Both logs are retained.

The optimized study writes to a new output directory because source/config
fingerprints change. Old incomplete split staging is retained but not resumed
or mixed into the new cache. Once the new study starts, the existing runner's
explicit `--resume` can reuse completed compatible splits. A partial split is
still recomputed rather than treated as valid data.
