# `hccr.evaluation`

Evaluation is inference-only and aggregates logits over a frozen validation
split. It produces both scalar metrics and artifacts for model debugging.

| File | Responsibility |
| --- | --- |
| `metrics.py` | Computes top-1 and top-5 accuracy. |
| `evaluator.py` | Runs model evaluation and connects diagnostics to a run directory. |
| `diagnostics.py` | Per-class accuracy, top-k errors, calibration/ECE, reliability diagram, health JSON and error gallery. |
| `reports.py` | Learning curves, selected-class confusion matrix and confidence distribution plots. |
| `analysis.py` | Lightweight per-sample error and confusion-pair CSV helper. |

Do not render a full 7k-class confusion matrix. Use `per_class_metrics.csv`,
`validation_errors.csv` and selected confusion pairs to focus investigation.
