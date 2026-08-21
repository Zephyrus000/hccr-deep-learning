# `hccr.preprocessing`

Preprocessing converts one PIL image into a square grayscale image. The dataset
then converts it to a normalized `torch.Tensor` of shape `[1, H, W]`.

| File | API | Behavior |
| --- | --- | --- |
| `pipeline.py` | `EvalPreprocessor` | Deterministic grayscale, optional filtering/binarization, crop, resize, padding, centroid centering and output polarity. |
| `pipeline.py` | `TrainPreprocessor` | Eval normalization followed by random affine, optional blur, elastic deformation, and mutually exclusive black-foreground erosion/dilation. |
| `gallery.py` | `save_gallery` | Writes raw-versus-transformed contact sheets for visual QA. |

Validation and inference must use `EvalPreprocessor`; random augmentation is
train-only. Otsu binarization, median filtering and centroid centering are
ablation options, not unconditional defaults, because they can damage fine
strokes in clean CASIA images.

Elastic and morphology probabilities default to zero so the existing random
affine pipeline remains the control. Enabled runs save
`augmentation_gallery.png`; each epoch records actual application counts and
rates returned by DataLoader workers.

`input_polarity` accepts `black_on_white` (default) and `white_on_black`.
Normalization and morphology run in the canonical black-on-white representation,
then the final image is inverted when requested. Erosion and dilation therefore
keep the same meaning for both output polarities.

Deterministic Sobel/Gabor features are implemented by the model-side input
adapter rather than Pillow. This keeps raw grayscale as channel 0 and guarantees
the same fixed Torch kernels are used during training, validation, benchmarking,
and deployment. Directional runs also save `directional_input_gallery.png`
with one column per effective input channel.
