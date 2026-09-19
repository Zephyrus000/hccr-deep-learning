# CLI commands

This package keeps command-line parsing and dispatch separate from domain logic.
The top-level [`hccr.cli`](../cli.py) module only builds the root parser and
routes the parsed namespace to one command handler.

| Module | Status | Responsibility |
| --- | --- | --- |
| `train.py` | Executable | Owns training options, maps them to `TrainingConfig`, and starts training. |
| `benchmark.py` | Executable | Builds the deploy-form model and measures inference latency. |
| `compare_runs.py` | Executable | Applies the accuracy/latency quality gate. |
| `validate.py` | Scaffold | Reserves the standalone checkpoint-validation boundary. |
| `deploy.py` | Scaffold | Reserves the deploy-artifact boundary. |

Each command module exposes `NAME`, `HELP`, `configure_parser(parser)`, and
`run(arguments)`. New workflow options belong in their command module, not in
`hccr.cli`. Domain computation should remain in the corresponding package such
as `training`, `evaluation`, `models`, or `inference`.

The legacy `scripts/benchmark_inference.py` entrypoint is a compatibility wrapper
over `hccr.commands.benchmark`; `hccr benchmark` is the preferred interface.

## Reproduction entry points

Reviewers should use the repository scripts for paper matrices and the CLI for
one-off inspection:

```bash
# Expand and validate every paper command without training.
python scripts/run_experiments.py \
  --config configs/experiment/thesis_model_comparison.yaml \
  --dry-run

# Continue the materialized matrix on its declared hardware.
python scripts/run_experiments.py \
  --config configs/experiment/thesis_model_comparison.yaml \
  --resume --show-output

# Inspect one checkpoint under an explicitly declared benchmark protocol.
hccr benchmark --mode eager --checkpoint experiments/<run-id>/checkpoint.pt ...
```

The CLI is intentionally not the paper protocol by itself. Exact paper options
live in the checked-in YAML and the expanded `plan.json`. Moving parsing or
dispatch among command modules is paper-neutral; changing argument defaults is
safe for old paper artifacts only because their effective values are persisted.
Any change to argument-to-`TrainingConfig` mapping requires a dry-run plan diff
and a fixed-run equivalence test.
