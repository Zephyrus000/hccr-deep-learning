# `hccr.preprocessing`

Preprocessing converts one PIL image into a square grayscale image. The dataset
then converts it to a normalized `torch.Tensor` of shape `[1, H, W]`.

| File | API | Behavior |
| --- | --- | --- |
| `pipeline.py` | `EvalPreprocessor` | Deterministic grayscale, optional filtering/binarization, crop, resize, padding and optional centroid centering. |
| `pipeline.py` | `TrainPreprocessor` | Eval normalization followed by random rotation, translation, scale and optional blur. |
| `gallery.py` | `save_gallery` | Writes raw-versus-transformed contact sheets for visual QA. |

Validation and inference must use `EvalPreprocessor`; random augmentation is
train-only. Otsu binarization, median filtering and centroid centering are
ablation options, not unconditional defaults, because they can damage fine
strokes in clean CASIA images.
