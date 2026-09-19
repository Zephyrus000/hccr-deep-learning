# `hccr.evaluation`

This package evaluates frozen dataset splits and produces scalar metrics plus
diagnostic artifacts. Evaluation always runs with `torch.inference_mode()` and
switches the model to evaluation mode.

## Package API

The package root exports:

| Symbol | Purpose |
| --- | --- |
| `classification_metrics(logits, targets)` | Compute top-1 and top-5 accuracy for an in-memory batch. |
| `save_learning_curves(output_dir, epochs, ...)` | Render training/evaluation curves. |

The training workflow calls `evaluator.evaluate` directly. It evaluates the
validation loader during training and the test loader only after selecting the
best validation checkpoint. Its loader must yield `(images, targets, metadata)`
batches, matching `HCCRDataset`.

```python
from hccr.evaluation.evaluator import evaluate

metrics = evaluate(model, test_loader, device="cuda", evaluation_name="test")
print(metrics["top1"], metrics["top5"])
```

When `output_dir` is omitted, `evaluate` returns aggregate accuracy and
support-tier recall only. Supplying `output_dir`, `source_root`, labels, and
training class support enables streaming diagnostics without retaining every
logit tensor in memory.

## Files

| File | Responsibility |
| --- | --- |
| `metrics.py` | Stateless top-k batch metrics. |
| `evaluator.py` | Evaluation loop and aggregate recall calculation. |
| `diagnostics.py` | Streaming errors, per-class recall, support tiers, ECE, calibration, and galleries. |
| `reports.py` | Learning curves, confusion views, and confidence plots. |
| `analysis.py` | Lightweight error/confusion CSV helpers. |

Typical diagnostic output includes `per_class_metrics.csv`,
`<evaluation>_errors.csv`, `confusion_pairs.csv`, `class_tiers.json`,
`calibration_bins.json`, `reliability_diagram.png`, and matching health/error
gallery artifacts. The final diagnostic pass uses `evaluation_name="test"`.

Support tiers are derived from active training support: the bottom 20% of
classes are `tail`, the top 20% are `head`, and the remainder are `mid`, with
class ID used for deterministic tie-breaking. Avoid rendering a dense full
7,186-class confusion matrix; use the ranked pair and per-class artifacts.

## Paper reporting contract

Architecture choices, scheduling, early stopping, BatchNorm-variant selection,
and ablations use validation only. The test loader is not constructed under
`validation_only`; `final_test` evaluates the already selected checkpoint once.
Paper tables must label the split and must not substitute the best test result
across epochs, seeds, or checkpoint variants.

For each reported aggregate, retain seed-level `metrics.json` files and the
sweep `summary.json`. Report the mean and sample variation over the declared
seeds, and keep top-1, top-5, macro recall, tail recall, and calibration metrics
under their stored definitions. A refactor that changes class support tiers,
aggregation, tie-breaking, or test access changes the paper protocol and
requires recomputing affected tables.
