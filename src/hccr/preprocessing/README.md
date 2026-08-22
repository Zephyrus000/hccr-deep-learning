# `hccr.preprocessing`

Preprocessing converts one PIL image into a square grayscale image. The dataset
then converts it to a normalized `torch.Tensor` of shape `[1, H, W]`.

| File | API | Behavior |
| --- | --- | --- |
| `pipeline.py` | `EvalPreprocessor` | Deterministic grayscale crop, resize, and padding. |
| `pipeline.py` | `TrainPreprocessor` | Eval normalization followed by the retained random affine and blur augmentation. |
| `gallery.py` | `save_gallery` | Writes raw-versus-transformed contact sheets for visual QA. |

Validation and inference must use `EvalPreprocessor`; random augmentation is
train-only. The preprocessing path is deliberately fixed after ablation.
ablation options, not unconditional defaults, because they can damage fine
strokes in clean CASIA images.

Random affine and blur preserve the baseline augmentation used by the retained
affine pipeline remains the control. Enabled runs save
`augmentation_gallery.png`; each epoch records actual application counts and
rates returned by DataLoader workers.

Images use the canonical black-on-white representation,
then the final image is inverted when requested. Erosion and dilation therefore
keep the same meaning for both output polarities.

Deterministic Sobel/Gabor features are implemented by the model-side input
adapter rather than Pillow. This keeps raw grayscale as channel 0 and guarantees
the same fixed Torch kernels are used during training, validation, benchmarking,
and deployment. Directional runs also save `directional_input_gallery.png`
with one column per effective input channel.
