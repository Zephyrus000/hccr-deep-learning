# `hccr.evidence`

This package creates a reviewer-facing, versioned view over historical run
artifacts. Source files are never rewritten: JSON, CSV, plots, and text reports
are copied byte-for-byte and recorded with SHA-256 checksums. Checkpoints are
not copied by default, but their checksums remain in each run record.

Each bundle contains `evidence_manifest.json`, canonical
`runs/<run-id>/run_evidence.json` records, canonical sweep records, and the
original files under `raw/`. Missing legacy measurements are represented as
`null` with `availability: not_measured`; the normalizer never estimates them.

Build and validate a bundle:

```bash
python scripts/build_evidence_bundle.py \
  --experiments experiments \
  --derived-root artifacts/experiments_20260915/profiling-20260918-rtx3060 \
  --derived-root artifacts/experiments_20260915/profiling-20260918-rtx3060-multi-branch \
  --derived-root artifacts/experiments_20260915/profiling-20260919-rtx3060-all-runs \
  --output artifacts/paper-evidence-v1-full

python scripts/build_evidence_bundle.py \
  --validate artifacts/paper-evidence-v1-full
```

Use `--run-id` and `--sweep-id` repeatedly to publish only runs referenced by
the paper. Use `--derived-root` repeatedly for post-hoc profiling collections;
their raw files retain checksums while each resource profile receives the same
canonical view as a training run. Build into a new empty directory so an older
bundle cannot be silently mixed with new evidence.

The repository ignores `artifacts/`. Publish a completed bundle as a versioned
release asset or immutable research deposit and cite its URL/DOI, source Git
commit, and `evidence_manifest.json` SHA-256 in the artifact appendix.
