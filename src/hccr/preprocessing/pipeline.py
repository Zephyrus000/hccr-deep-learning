"""Stable grayscale preprocessing for the retained experiment family."""

from __future__ import annotations

from dataclasses import dataclass
from random import uniform

from PIL import Image, ImageFilter


@dataclass(frozen=True)
class EvalPreprocessor:
    image_size: int = 64
    margin: int = 4

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        bbox = grayscale.point(lambda value: 255 if value < 250 else 0).getbbox()
        if bbox is not None:
            grayscale = grayscale.crop(bbox)
        inner = self.image_size - 2 * self.margin
        grayscale.thumbnail((inner, inner), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (self.image_size, self.image_size), color=255)
        offset = ((self.image_size - grayscale.width) // 2, (self.image_size - grayscale.height) // 2)
        canvas.paste(grayscale, offset)
        return canvas


@dataclass(frozen=True)
class TrainPreprocessor(EvalPreprocessor):
    rotation_degrees: float = 8.0
    translate_ratio: float = 0.08
    scale_min: float = 0.9
    scale_max: float = 1.1
    blur_probability: float = 0.1

    def __call__(self, image: Image.Image) -> Image.Image:
        prepared = super().__call__(image)
        scale = uniform(self.scale_min, self.scale_max)
        resized = prepared.resize((round(self.image_size * scale), round(self.image_size * scale)), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.image_size, self.image_size), color=255)
        offset = ((self.image_size - resized.width) // 2, (self.image_size - resized.height) // 2)
        canvas.paste(resized, offset)
        translated = (uniform(-self.translate_ratio, self.translate_ratio) * self.image_size, uniform(-self.translate_ratio, self.translate_ratio) * self.image_size)
        transformed = canvas.rotate(uniform(-self.rotation_degrees, self.rotation_degrees), resample=Image.Resampling.BILINEAR, translate=translated, fillcolor=255)
        if uniform(0, 1) < self.blur_probability:
            transformed = transformed.filter(ImageFilter.GaussianBlur(radius=0.5))
        transformed.info["applied_augmentations"] = ()
        return transformed
