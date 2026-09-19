# `hccr.inference`

`hccr.inference` exposes `Predictor`, a small top-k wrapper for an already
constructed model and ordered label list.

## Contract

```python
from hccr.inference import Predictor

predictor = Predictor(model, labels, device="cuda")
predictions = predictor.predict(image_tensor, top_k=5)
```

- `image_tensor` is one preprocessed sample with shape `[1, H, W]`.
- `labels[index]` must describe the model output at the same index.
- The model is moved to the requested device and put into evaluation mode.
- The return value is a descending list of `(label, probability)` tuples.
- `top_k` is capped at the number of labels.

Build `labels` from the run's `labels.json`; do not sort labels independently,
because class-subset training may remap original class IDs.

## Deliberate boundaries

`Predictor` does not load checkpoints, parse checkpoint metadata, open image
files, or apply preprocessing. Deployment code owns those steps so it can
validate artifact compatibility and choose CPU/CUDA policy explicitly.

For deploy benchmarking, optimize a loaded model before constructing the
predictor:

```python
from hccr.models import optimize_model_for_inference

model = optimize_model_for_inference(model)
predictor = Predictor(model, labels, device="cpu")
```

The high-level `hccr predict` CLI command is currently a placeholder; this
Python API is the implemented inference surface.

## Reviewer checks

Prediction is reproducible only when four artifacts agree: checkpoint weights,
checkpoint model metadata, ordered `labels.json`, and evaluation preprocessing.
Before accepting a refactored loader or deploy path, compare eager logits and
top-k indices on fixed preprocessed tensors and reject missing/unexpected state
keys. Optimized inference requires a second logit-equivalence check and must be
reported separately from the eager paper comparison.
