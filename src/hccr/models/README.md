# `hccr.models`

The target architecture is a compact grayscale CNN built for a joint accuracy
and inference-latency objective.

| Symbol | Role |
| --- | --- |
| `ConvNormAct` | 3×3 convolution, batch normalization and SiLU activation. |
| `DepthwiseSeparableBlock` | Depthwise convolution, pointwise projection and residual skip. |
| `EfficientHCCRNet` | Stem, three configurable feature stages, global average pool and linear classifier. |
| `build_model(name, **kwargs)` | Factory; currently accepts `efficient_hccr`. |

Inputs are `[batch, 1, image_size, image_size]`; outputs are class logits.
`width` controls channels and `stage_depths` controls stage capacity. Compare
architecture variants using accuracy plus batch-1 p95 latency from
`resource_profile.json`.
