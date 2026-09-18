# `hccr.models`

This package contains the proposed compact grayscale CNN, fixed reference
baselines, and deployment optimization. Architectures removed by the ablation
process—attention blocks, cross-stage routes, CSP stages, and directional input
adapters—are not part of the proposed-model API.

## Public API

| Symbol | Purpose |
| --- | --- |
| `EfficientHCCRNet` | Three-stage depthwise-separable CNN for one-channel character images. |
| `build_model(name, **kwargs)` | Factory accepting `efficient_hccr`, `resnet18`, `mobilenet_v3_small`, `shufflenet_v2_x1_0`, or `efficientnet_b0`. |
| `optimize_model_for_inference(model)` | Return a frozen eval copy with safe inference transformations. |

Internal building blocks in `efficient_hccr.py` are:

- `ConvNormAct`: 3×3 convolution, BatchNorm, and SiLU.
- `DepthwiseSeparableBlock`: depthwise spatial convolution, pointwise
  projection, residual/identity skip, and SiLU.
- `AngularMarginClassifier`: normalized CosFace or ArcFace head.

## Reference baselines

`resnet18`, `mobilenet_v3_small`, `shufflenet_v2_x1_0`, and `efficientnet_b0`
are standard torchvision architectures initialized from scratch
(`weights=None`). All replace the RGB stem with a one-channel convolution for
grayscale HCCR input. Their normal softmax classifier remains the default, while
`classification_head="cosface"` or `"arcface"` replaces only the final logits
layer and retains each architecture's penultimate projection. This permits a
matched angular-margin head across proposed and reference models.

For the stricter baseline-fairness study, set `embedding_dim=320` on each
torchvision baseline. This removes its original classifier prefix and applies a
learned `Linear(native_backbone_dim, 320, bias=False)` before the common
`CosFace(320, num_classes)` classifier. EfficientHCCRNet is left unchanged;
leaving `embedding_dim` unset preserves legacy baseline checkpoints exactly.

Each baseline exposes `backbone` and `classifier` separately. Resource reports
therefore measure its classifier and project the last logits layer to the
full-class setting without treating the whole model as an opaque block.

## Model contract

```python
import torch

from hccr.models import EfficientHCCRNet

model = EfficientHCCRNet(
    num_classes=1000,
    width=64,
    stage_depths=(1, 2, 2),
    stem_stride=2,
    backbone_output_channels=320,
    embedding_dim=192,
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

## Decoupled classifier capacity

The final stage width and the angular-classifier dimension are separate. By
default, both retain the legacy `4 × width` value, so existing checkpoints can
be reconstructed without a structural change. New runs may use
`backbone_output_channels` to add a late 1×1 Conv-BatchNorm-SiLU expansion and
`embedding_dim` to project pooled features before the classifier:

```
stages -> late 1×1 backbone expansion -> global average pool
       -> linear embedding projection -> CosFace / ArcFace classifier
```

This lets the final feature extractor gain capacity while constraining the
full-class classifier, whose weight count is `num_classes × embedding_dim`.
For example, 7,186 classes at embedding dimensions 192, 160, and 128 require
about 1.38M, 1.15M, and 0.92M classifier weights respectively. The resource
profile reports backbone, embedding-projection, classifier, and combined-head
costs separately; latency remains the decision metric.

## Angular heads

CosFace and ArcFace normalize embeddings and class weights. Plain `forward`
returns target-free scaled cosine logits for validation/inference.
`training_logits(inputs, targets, margin_multiplier)` applies the target margin
only during training; the multiplier must be between zero and one and supports
margin warm-up. The same head contract applies to EfficientHCCRNet and all four
reference baselines; `classification_head="softmax"` keeps a plain classifier.

## Inference optimization

`optimize_model_for_inference` deep-copies the model, switches it to eval mode,
folds Conv-BatchNorm pairs, collapses reparameterized depthwise branches,
folds the optional late pointwise expansion, replaces eval dropout with identity,
caches normalized classifier weights, and disables gradients. The source
training model and checkpoint remain unchanged.

Always verify optimized/eager logit equivalence on the target device. The
training resource profile performs this check and records both benchmark sets.
Compare model candidates using validation accuracy and matched batch-1 p95
latency, not parameter count alone.
