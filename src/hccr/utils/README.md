# `hccr.utils`

This package contains small cross-cutting helpers with no dependency on CLI
dispatch or training business logic.

## Package exports

| Symbol | Purpose |
| --- | --- |
| `resolve_device(preferred="auto")` | Resolve `auto`, `cpu`, or `cuda` with explicit errors for unavailable CUDA. |
| `configure_logging(output_dir)` | Create the run logger with console and UTF-8 `run.log` handlers. |
| `close_logging(logger)` | Flush, detach, and close handlers so Windows releases artifact files. |

```python
from pathlib import Path

from hccr.utils import close_logging, configure_logging, resolve_device

device = resolve_device("auto")
logger = configure_logging(Path("experiments/example"))
try:
    logger.info("resolved device=%s", device)
finally:
    close_logging(logger)
```

Progress-log records under `hccr.train` are excluded from the console handler so
they do not interrupt tqdm, but they remain available in `run.log`.

## Experiment artifact helpers

`utils.experiment` is imported directly by orchestration modules and provides:

| Symbol | Purpose |
| --- | --- |
| `write_json(path, payload)` | Create parents and write stable UTF-8 indented JSON; dataclasses and `Path` values are supported. |
| `new_run_id()` | Generate a UTC timestamp plus short random suffix. |
| `initialize_run(...)` | Write effective config and metadata including Git/environment information. |
| `write_curves(output_dir, epochs)` | Store the epoch list under the `epochs` key. |

Device resolution imports PyTorch lazily. This keeps CPU-only bootstrap and
configuration inspection usable even when PyTorch is absent; explicitly
requesting CUDA still fails clearly.

## Artifact provenance

`metadata.json` is the reviewer-facing provenance record. It binds a run to its
Git commit, dirty working-tree digest, environment, device, and manifest
digest. Keep it beside the resolved config and checkpoint; a checkpoint copied
without provenance cannot establish that it corresponds to a paper row.

Refactors of metadata helpers must preserve stable JSON types and existing
field meanings. Adding a versioned field is safe; silently reinterpreting or
overwriting a historical metric is not. Post-hoc audit scripts should write new
artifacts or update only fields they actually recomputed, leaving unavailable
measurements explicit rather than replacing them with empty values.
