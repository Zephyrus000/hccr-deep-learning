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
