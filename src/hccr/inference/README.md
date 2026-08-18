# `hccr.inference`

`predictor.py` contains `Predictor`, a small top-k prediction wrapper.

```python
predictor = Predictor(model, labels, device="cuda")
results = predictor.predict(image_tensor, top_k=5)
```

`image_tensor` is a single preprocessed `[1, H, W]` tensor. `labels` must be
ordered by the model output index; use the run's `labels.json` artifact when
building that list. The predictor deliberately does not load checkpoints or
apply preprocessing itself, so serving code can own those deployment choices.
