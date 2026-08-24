# `hccr.preprocessing`

This package converts PIL images into the canonical square grayscale images
consumed by `HCCRDataset`. Tensor conversion and `[0, 1]` scaling happen in the
dataset, not in the preprocessing classes.

## Public API

| Symbol | Behavior |
| --- | --- |
| `EvalPreprocessor` | Deterministic grayscale conversion, foreground crop, aspect-preserving resize, and centered white padding. |
| `TrainPreprocessor` | Evaluation normalization followed by random scale, translation, rotation, and optional Gaussian blur. |

`gallery.save_gallery` is an artifact helper used by training to compare raw
and transformed samples visually.

## Evaluation pipeline

```python
from hccr.preprocessing import EvalPreprocessor

transform = EvalPreprocessor(image_size=64, margin=4)
prepared = transform(pil_image)
```

The transform:

1. Converts the image to Pillow mode `L`.
2. Detects foreground pixels darker than 250 and crops to their bounding box.
3. Fits the crop inside `image_size - 2 × margin` without changing aspect ratio.
4. Centers it on a white `image_size × image_size` canvas.

Use this deterministic transform for validation, diagnostics, and inference.

## Training augmentation

`TrainPreprocessor` inherits the evaluation pipeline. Its retained defaults are:

| Parameter | Default |
| --- | --- |
| `rotation_degrees` | `8.0` (sampled symmetrically) |
| `translate_ratio` | `0.08` of image size per axis |
| `scale_min`, `scale_max` | `0.9`, `1.1` |
| `blur_probability` | `0.1` with radius `0.5` |

```python
from hccr.preprocessing import TrainPreprocessor

transform = TrainPreprocessor(image_size=64, blur_probability=0.1)
augmented = transform(pil_image)
```

The current transform writes the transforms actually applied to each sample into
the Pillow metadata; `HCCRDataset` propagates that field as a string for a stable
batch metadata contract. Do not use the training transform for validation or
inference.

Polarity switching, Otsu/median filtering, morphology, elastic deformation,
centroid centering, and Sobel/Gabor channels are not supported by the retained
pipeline.
