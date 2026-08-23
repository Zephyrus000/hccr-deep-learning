# `hccr.models`

This package contains the retained compact grayscale CNN and its deployment
optimization. Architectures removed by the ablation process—attention blocks,
cross-stage routes, CSP stages, and directional input adapters—are not part of
the current API.

## Public API

| Symbol | Purpose |
| --- | --- |
| `EfficientHCCRNet` | Three-stage depthwise-separable CNN for one-channel character images. |
| `build_model(name, **kwargs)` | Factory accepting the single name `efficient_hccr`. |
| `optimize_model_for_inference(model)` | Return a frozen eval copy with safe inference transformations. |

Internal building blocks in `efficient_hccr.py` are:

- `ConvNormAct`: 3×3 convolution, BatchNorm, and SiLU.
- `DepthwiseSeparableBlock`: depthwise spatial convolution, pointwise
  projection, residual/identity skip, and SiLU.
- `AngularMarginClassifier`: normalized CosFace or ArcFace head.

## Model contract

```python
import torch

from hccr.models import EfficientHCCRNet

model = EfficientHCCRNet(
    num_classes=1000,
    width=64,
    stage_depths=(1, 2, 2),
    stem_stride=2,
    reparameterize_depthwise=True,
    classification_head="cosface",
)
logits = model(torch.rand(8, 1, 64, 64))
```

Inputs must be BCHW grayscale tensors; multi-channel or non-4D inputs are
rejected. `width` defines stage channels `(width, 2×width, 4×width)`, and
`stage_depths` must contain three positive integers. The stem downsamples once;
stages 2 and 3 downsample at their first block.

`forward_features(inputs, return_stages=True)` returns the final feature map and
named `stage1`/`stage2`/`stage3` outputs. The flat `features.*` layout is kept so
checkpoint keys remain stable while diagnostics can inspect logical stages.

`stem_stride=1` preserves the full input resolution through stage 1 for the
thin-stroke ablation; the default `2` keeps the low-latency path. Enabling
`reparameterize_depthwise` trains parallel depthwise `3×3`, `1×3`, and `3×1`
branches followed by the existing pointwise `1×1` projection. Deployment pads
and sums the asymmetric kernels into one `3×3` depthwise convolution.

The training CLI and `TrainingConfig` promote the multi-branch form by default;
`--no-reparameterize-depthwise` remains available for matched control runs.
The low-level `EfficientHCCRNet` constructor retains its explicit single-branch
default so older checkpoint reconstruction does not silently change state keys.

## Angular heads

CosFace and ArcFace normalize embeddings and class weights. Plain `forward`
returns target-free scaled cosine logits for validation/inference.
`training_logits(inputs, targets, margin_multiplier)` applies the target margin
only during training; the multiplier must be between zero and one and supports
margin warm-up.

## Inference optimization

`optimize_model_for_inference` deep-copies the model, switches it to eval mode,
folds Conv-BatchNorm pairs, collapses reparameterized depthwise branches,
replaces eval dropout with identity, caches normalized classifier weights, and
disables gradients. The source training model and checkpoint remain unchanged.

Always verify optimized/eager logit equivalence on the target device. The
training resource profile performs this check and records both benchmark sets.
Compare model candidates using validation accuracy and matched batch-1 p95
latency, not parameter count alone.
