# `hccr.models`

The target architecture is a compact grayscale CNN built for a joint accuracy
and inference-latency objective.

| Symbol | Role |
| --- | --- |
| `ConvNormAct` | 3×3 convolution, batch normalization and SiLU activation. |
| `DepthwiseSeparableBlock` | Depthwise convolution, pointwise projection and residual skip. |
| `CrossStageAdd` | Project-defined stage-2→stage-3 projected additive bridge. |
| `CrossStageCBAM` | Paper-inspired cross-stage route using parallel CAM/SAM on the stage-2 source. |
| `CrossStagePartialStage` | CSP split/transform/bypass/merge stage. |
| `AngularMarginClassifier` | Target-free cosine inference with train-only CosFace/ArcFace margins. |
| `DirectionalInputAdapter` | Fixed grayscale+Sobel or grayscale+four-orientation-Gabor input features. |
| `EfficientHCCRNet` | Stem, three configurable feature stages, global average pool and linear classifier. |
| `build_model(name, **kwargs)` | Factory; currently accepts `efficient_hccr`. |

Inputs are `[batch, 1, image_size, image_size]`; outputs are class logits.
The retained model always consumes one grayscale channel.
Directional modes always retain raw grayscale as channel 0 and compute fixed
features inside the model so training, validation, and inference share exactly
the same implementation.
`width` controls channels and `stage_depths` controls stage capacity. Compare
architecture variants using accuracy plus batch-1 p95 latency from
`resource_profile.json`.

`EfficientHCCRNet.forward_features(..., return_stages=True)` exposes logical
`stage1`/`stage2`/`stage3` outputs without changing legacy `features.*`
checkpoint keys. Training accepts exactly three positive depths through
`--stage-depths`, for example `--stage-depths 1 2 3`.

Width is a positive integer, and `stage_depths` accepts any three positive
integers. Classifier choices are CosFace and ArcFace.
candidate promotion still depends on matched GPU/CPU latency and validation
accuracy rather than parameter count alone.

Cross-stage variants are disabled by default and must be ablated separately:

- `--cross-stage projected_residual` enables the projected residual route.
- `--cross-stage c_cbam` enables the reference-inspired stage-2→stage-3
  adaptation. It uses the reference's middle-stage parallel CAM/SAM pattern,
  but does not claim to reproduce the paper's complete SqueezeNext topology.
- `--csp-stages 3 --csp-split-ratio 0.5` replaces stage 3 with a CSP stage.

Do not combine these mechanisms for their first screen. Compare each candidate
against the same class subset, seed, resolution, and CPU/CUDA benchmark
protocol before considering a combined model.

Resource profiles report total, backbone, and classifier-head parameters and
MACs separately. Use backbone values when comparing architectures across
different class counts because the 7,186-class head is much larger than a
1,000-class subset head.

For deployment, call `optimize_model_for_inference(model)` after loading the
checkpoint and moving it to the target device. It returns an eval-only copy
that caches the normalized CosFace/ArcFace class weights, folds Conv-BatchNorm
pairs, removes the eval-time dropout hop, and uses the sequential feature fast
path. These transformations do not require retraining and resource profiles
record both eager and optimized benchmarks plus a logit-equivalence check.
