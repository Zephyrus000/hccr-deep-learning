# `hccr.config`

Configuration helpers are intentionally small and have no CLI dependency.

| File | API | Purpose |
| --- | --- | --- |
| `loader.py` | `load_yaml(path)` | Loads a YAML document and requires a mapping root. |
| `schema.py` | `DataConfig` | Manifest path, image size and seed contract. |
| `schema.py` | `ModelConfig` | Generic model name, class count and channel contract. |
| `schema.py` | `ExperimentConfig` | Run name, output directory and requested device. |

`configs/experiment/baseline.yaml` documents the current baseline. The train
CLI remains the executable source of configuration and writes the final
effective values to each run's `config.json`.
