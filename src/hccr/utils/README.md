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

## Run persistence

Run metadata, JSON serialization, and checkpoints belong to
[`hccr.artifacts`](../artifacts/README.md).
