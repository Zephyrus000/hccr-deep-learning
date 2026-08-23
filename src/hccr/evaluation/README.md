# `hccr.evaluation`

This package evaluates frozen validation splits and produces scalar metrics plus
diagnostic artifacts. Evaluation always runs with `torch.inference_mode()` and
switches the model to evaluation mode.

## Package API

The package root exports:

| Symbol | Purpose |
| --- | --- |
| `classification_metrics(logits, targets)` | Compute top-1 and top-5 accuracy for an in-memory batch. |
| `save_learning_curves(output_dir, epochs, ...)` | Render training/validation curves. |

The training workflow calls `evaluator.evaluate` directly. Its loader must
yield `(images, targets, metadata)` batches, matching `HCCRDataset`.

```python
from hccr.evaluation.evaluator import evaluate

metrics = evaluate(model, validation_loader, device="cuda")
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
| `evaluator.py` | Validation loop and aggregate recall calculation. |
| `diagnostics.py` | Streaming errors, per-class recall, support tiers, ECE, calibration, and galleries. |
| `reports.py` | Learning curves, confusion views, and confidence plots. |
| `analysis.py` | Lightweight error/confusion CSV helpers. |

Typical diagnostic output includes `per_class_metrics.csv`,
`validation_errors.csv`, `confusion_pairs.csv`, `class_tiers.json`,
`calibration_bins.json`, `reliability_diagram.png`, and validation health/error
gallery artifacts.

Support tiers are derived from active training support: the bottom 20% of
classes are `tail`, the top 20% are `head`, and the remainder are `mid`, with
class ID used for deterministic tie-breaking. Avoid rendering a dense full
7,186-class confusion matrix; use the ranked pair and per-class artifacts.
