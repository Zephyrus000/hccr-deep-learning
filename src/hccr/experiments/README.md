# Experiments

`spec.py` defines `ExperimentSpec`, variants, jobs, YAML loading, and hardware
requirements. `runner.py` contains `ExperimentRunner`, which executes the ordered
jobs and persists resumable status. `summary.py` computes seed distributions,
Student-t confidence intervals, paired deltas, and latency comparisons.
`comparison.py` implements the CSV accuracy/latency gate.

```bash
python -m hccr.experiments --config configs/experiment/thesis_model_comparison.yaml --dry-run
```

`scripts/run_experiments.py` delegates to the same CLI. Python callers import
`ExperimentRunner` from `hccr.experiments.runner` and `ExperimentSpec` from
`hccr.experiments.spec`. The old `hccr.experiment_runner` module was removed.
The public `from hccr.experiments import compare_runs` import is unchanged.

The runner profiles the checkpoint selected on validation, including the
BatchNorm-recalibrated variant when selected. It reuses a training profile only
when the checkpoint SHA-256, runtime/hardware metadata, measurement counts,
and analytical class-count target match. A different device or protocol creates
a new profile under the sweep directory. It does not benchmark a newly initialized
full-vocabulary head; `full_class_num_classes` controls analytical complexity only.

Sweep comparisons use validation metrics. Held-out test and ICDAR measurements
stay in separate record fields and are never substituted for selection metrics.
Sweep summary schema 3 omits the retired optimized and synthetic-head aggregates.
