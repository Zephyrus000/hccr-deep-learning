# `hccr.utils`

Shared helpers that avoid trainer/CLI-specific behavior.

| File | API | Purpose |
| --- | --- | --- |
| `device.py` | `resolve_device` | Resolves `auto`, `cpu`, or `cuda` and fails clearly for unavailable CUDA. |
| `experiment.py` | `new_run_id`, `initialize_run`, `write_json`, `write_curves` | Run IDs, environment/Git metadata and structured artifacts. |
| `logging.py` | `configure_logging`, `close_logging` | Console plus `run.log`, with handlers closed to release Windows file locks. |

Use `write_json` for run artifacts: it creates parent directories, serializes
dataclasses and handles `Path` values safely.
