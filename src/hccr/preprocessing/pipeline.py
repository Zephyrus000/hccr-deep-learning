"""Pillow-based image pipelines that never read files or CLI configuration."""

from __future__ import annotations

from dataclasses import dataclass
from random import random, uniform


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
    input_polarity: str = "black_on_white"
    center_by_centroid: bool = False
    otsu_binarize: bool = False
    median_filter_size: int | None = None

    def __call__(self, image):
        prepared = self._prepare_black_on_white(image)
        return self._apply_input_polarity(prepared)

    def _prepare_black_on_white(self, image):
        Image, ImageChops, ImageOps = _pillow()
        if self.input_polarity not in {"black_on_white", "white_on_black"}:
            raise ValueError(
                "input_polarity must be one of: black_on_white, white_on_black"
            )
        grayscale = image.convert("L")
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

    def _apply_input_polarity(self, image):
        if self.input_polarity == "white_on_black":
            _, _, ImageOps = _pillow()
            return ImageOps.invert(image)
        return image


@dataclass(frozen=True)
class TrainPreprocessor(EvalPreprocessor):
    rotation_degrees: float = 8.0
    translate_pixels: int = 3
    scale_range: tuple[float, float] = (0.9, 1.1)
    blur_radius: float | None = None
    elastic_probability: float = 0.0
    elastic_displacement_ratio: float = 0.015
    erosion_probability: float = 0.0
    dilation_probability: float = 0.0
    morphology_kernel_size: int = 3

    def __call__(self, image):
        Image, _, _ = _pillow()
        self._validate_augmentation_options()
        prepared = self._prepare_black_on_white(image)
        applied: list[str] = []
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
            applied.append("gaussian_blur")
        if random() < self.elastic_probability:
            prepared = _elastic_mesh_transform(
                prepared, self.elastic_displacement_ratio
            )
            applied.append("elastic")
        morphology_draw = random()
        if morphology_draw < self.erosion_probability:
            from PIL import ImageFilter

            prepared = prepared.filter(
                ImageFilter.MaxFilter(self.morphology_kernel_size)
            )
            applied.append("erosion")
        elif morphology_draw < (self.erosion_probability + self.dilation_probability):
            from PIL import ImageFilter

            prepared = prepared.filter(
                ImageFilter.MinFilter(self.morphology_kernel_size)
            )
            applied.append("dilation")
        prepared = self._apply_input_polarity(prepared)
        prepared.info["applied_augmentations"] = tuple(applied)
        return prepared

    def _validate_augmentation_options(self) -> None:
        probabilities = (
            self.elastic_probability,
            self.erosion_probability,
            self.dilation_probability,
        )
        if any(not 0 <= probability <= 1 for probability in probabilities):
            raise ValueError("augmentation probabilities must be between 0 and 1")
        if self.erosion_probability + self.dilation_probability > 1:
            raise ValueError("erosion and dilation probabilities must sum to at most 1")
        if not 0 <= self.elastic_displacement_ratio <= 0.015:
            raise ValueError("elastic_displacement_ratio must be in [0, 0.015]")
        if self.morphology_kernel_size < 3 or self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be an odd value >= 3")


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


def _elastic_mesh_transform(image, displacement_ratio: float):
    Image, _, _ = _pillow()
    if displacement_ratio == 0:
        return image
    cell_size = max(4, round(min(image.size) / 4))
    x_coordinates = list(range(0, image.width, cell_size)) + [image.width]
    y_coordinates = list(range(0, image.height, cell_size)) + [image.height]
    maximum_displacement = min(image.size) * displacement_ratio
    displaced = {
        (x, y): (
            min(
                image.width,
                max(0, x + uniform(-maximum_displacement, maximum_displacement)),
            ),
            min(
                image.height,
                max(0, y + uniform(-maximum_displacement, maximum_displacement)),
            ),
        )
        for y in y_coordinates
        for x in x_coordinates
    }
    mesh = []
    for y0, y1 in zip(y_coordinates, y_coordinates[1:], strict=False):
        for x0, x1 in zip(x_coordinates, x_coordinates[1:], strict=False):
            upper_left = displaced[(x0, y0)]
            lower_left = displaced[(x0, y1)]
            lower_right = displaced[(x1, y1)]
            upper_right = displaced[(x1, y0)]
            mesh.append(
                (
                    (x0, y0, x1, y1),
                    (*upper_left, *lower_left, *lower_right, *upper_right),
                )
            )
    return image.transform(
        image.size,
        Image.Transform.MESH,
        mesh,
        resample=Image.Resampling.BILINEAR,
        fillcolor=255,
    )
