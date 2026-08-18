"""Pillow-based image pipelines that never read files or CLI configuration."""

from __future__ import annotations

from dataclasses import dataclass
from random import uniform


def _pillow():
    try:
        from PIL import Image, ImageChops, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Pillow must be installed for image preprocessing"
        ) from error
    return Image, ImageChops, ImageOps


@dataclass(frozen=True)
class EvalPreprocessor:
    image_size: int = 64
    margin: int = 4
    invert: bool = False
    center_by_centroid: bool = False
    otsu_binarize: bool = False
    median_filter_size: int | None = None

    def __call__(self, image):
        Image, ImageChops, ImageOps = _pillow()
        grayscale = image.convert("L")
        if self.invert:
            grayscale = ImageOps.invert(grayscale)
        if self.median_filter_size is not None:
            from PIL import ImageFilter

            grayscale = grayscale.filter(
                ImageFilter.MedianFilter(self.median_filter_size)
            )
        if self.otsu_binarize:
            grayscale = grayscale.point(_otsu_threshold(grayscale))
        foreground = ImageChops.difference(
            grayscale, Image.new("L", grayscale.size, 255)
        )
        bbox = foreground.getbbox()
        cropped = grayscale.crop(bbox) if bbox else grayscale
        padded = ImageOps.expand(cropped, border=self.margin, fill=255)
        padded.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (self.image_size, self.image_size), 255)
        offset = [
            (self.image_size - padded.width) // 2,
            (self.image_size - padded.height) // 2,
        ]
        if self.center_by_centroid:
            centroid = _foreground_centroid(padded)
            if centroid is not None:
                offset[0] += round(self.image_size / 2 - (offset[0] + centroid[0]))
                offset[1] += round(self.image_size / 2 - (offset[1] + centroid[1]))
        canvas.paste(padded, tuple(offset))
        return canvas


@dataclass(frozen=True)
class TrainPreprocessor(EvalPreprocessor):
    rotation_degrees: float = 8.0
    translate_pixels: int = 3
    scale_range: tuple[float, float] = (0.9, 1.1)
    blur_radius: float | None = None

    def __call__(self, image):
        Image, _, _ = _pillow()
        prepared = super().__call__(image)
        angle = uniform(-self.rotation_degrees, self.rotation_degrees)
        translate = (
            round(uniform(-self.translate_pixels, self.translate_pixels)),
            round(uniform(-self.translate_pixels, self.translate_pixels)),
        )
        if self.rotation_degrees or self.translate_pixels:
            prepared = prepared.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                translate=translate,
                fillcolor=255,
            )
        scale = uniform(*self.scale_range)
        if scale != 1:
            prepared = _scale_centered(prepared, scale)
        if self.blur_radius is not None:
            from PIL import ImageFilter

            prepared = prepared.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return prepared


def _otsu_threshold(image):
    histogram = image.histogram()
    total = sum(histogram)
    total_intensity = sum(index * count for index, count in enumerate(histogram))
    background_count = 0
    background_intensity = 0
    best_threshold, best_variance = 0, -1.0
    for threshold, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        background_intensity += threshold * count
        background_mean = background_intensity / background_count
        foreground_mean = (total_intensity - background_intensity) / foreground_count
        variance = (
            background_count
            * foreground_count
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_threshold, best_variance = threshold, variance
    return lambda value: 0 if value <= best_threshold else 255


def _foreground_centroid(image):
    weights = [255 - value for value in image.get_flattened_data()]
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    x = (
        sum(index % image.width * weight for index, weight in enumerate(weights))
        / total_weight
    )
    y = (
        sum(index // image.width * weight for index, weight in enumerate(weights))
        / total_weight
    )
    return x, y


def _scale_centered(image, scale: float):
    Image, _, _ = _pillow()
    scaled_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    scaled = image.resize(scaled_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", image.size, 255)
    canvas.paste(
        scaled,
        ((image.width - scaled.width) // 2, (image.height - scaled.height) // 2),
    )
    return canvas
